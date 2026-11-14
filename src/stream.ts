import { createHash } from "node:crypto";
import {
  calculateCost,
  createAssistantMessageEventStream,
  getCurrentSystemPrompt,
  getCurrentTools,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type FetchFunction,
  type ImageContent,
  type JsonObject,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ThinkingContent,
  type Tool,
  type ToolCall,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { PROVIDER_ID, buildHeaders, endpointsFor, isRecord, parseApiKey, redactSecrets, stableUuid } from "./client.ts";
import { getThinkingConfig, resolveRuntimeModel, siblingRuntimeModels } from "./models.ts";

const DEFAULT_STREAM_TIMEOUT_MS = 180_000;
const CONTINUATION_TEXT = "Continue the active task from the conversation above.";
const OMITTED_IMAGE_TEXT = "[Image omitted: model does not support image input]";
const BASE64_SIG = /^[A-Za-z0-9+/]+={0,2}$/;

type GeminiPart =
  | { text: string; thought?: boolean; thoughtSignature?: string }
  | { inlineData: { mimeType: string; data: string } }
  | { functionCall: { name: string; args: Record<string, unknown>; id?: string }; thoughtSignature?: string }
  | { functionResponse: { name: string; response: { output?: string; error?: string }; id?: string } };

export type GeminiContent = { role: "user" | "model"; parts: GeminiPart[] };

let toolIdCounter = 0;

const isValidSig = (sig?: string): sig is string => Boolean(sig && sig.length % 4 === 0 && BASE64_SIG.test(sig));
const usesCustomBridge = (modelId: string, runtimeModel: string): boolean =>
  /^(claude|gpt-oss)-/.test(modelId) || /^(claude|gpt-oss)-/.test(runtimeModel);
const isModernGemini = (runtimeModel: string): boolean =>
  runtimeModel.startsWith("gemini-") && !/^gemini-[12](?:[.-]|$)/.test(runtimeModel);

function sanitizeToolCallId(id: string, fallbackName?: string, usedIds?: Set<string>): string {
  const raw = id.replace(/[^a-zA-Z0-9_-]/g, "_");
  let candidate = "";
  if (/[a-zA-Z0-9]/.test(raw)) {
    candidate =
      raw.length <= 64
        ? raw
        : `${raw.slice(0, 48)}_${createHash("sha1").update(id).digest("hex").slice(0, 15)}`;
  }
  if (!candidate || usedIds?.has(candidate)) {
    const prefix = (fallbackName ?? "tool").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 28) || "tool";
    do {
      candidate = `${prefix}_${Date.now().toString(36)}_${++toolIdCounter}`;
    } while (usedIds?.has(candidate));
  }
  usedIds?.add(candidate);
  return candidate;
}

function toImagePart(block: ImageContent, allowImages: boolean): GeminiPart | undefined {
  if (!block.data) return undefined;
  if (!allowImages) return { text: OMITTED_IMAGE_TEXT };
  const m = block.data.match(/^data:([^;]+);base64,(.+)$/s);
  const data = (m ? m[2]! : block.data).trim();
  return data ? { inlineData: { mimeType: block.mimeType || m?.[1] || "image/png", data } } : undefined;
}

function appendTurn(contents: GeminiContent[], role: "user" | "model", parts: GeminiPart[]): void {
  if (parts.length === 0) return;
  const prev = contents.at(-1);
  if (prev?.role === role) prev.parts.push(...parts);
  else contents.push({ role, parts: [...parts] });
}

type PendingCall = { name: string; wireId?: string; degradedArgs?: string };

export function convertMessages(
  model: Model<Api>,
  messages: readonly Message[],
  runtimeModel: string,
): GeminiContent[] {
  const contents: GeminiContent[] = [];
  const requiresSig = isModernGemini(runtimeModel);
  const isClaude = model.id.startsWith("claude-") || runtimeModel.startsWith("claude-");
  const needsId = usesCustomBridge(model.id, runtimeModel) || requiresSig;
  const allowImages = model.input.includes("image");
  const usedIds = new Set<string>();
  const pendingByKey = new Map<string, PendingCall[]>();
  const pendingOrder: PendingCall[] = [];

  const enqueuePending = (key: string, entry: PendingCall) => {
    const list = pendingByKey.get(key) ?? [];
    list.push(entry);
    pendingByKey.set(key, list);
    pendingOrder.push(entry);
  };

  const dequeuePending = (rawId: string, toolName: string): PendingCall | undefined => {
    const key = rawId ? `id:${rawId}` : `name:${toolName}`;
    const list = pendingByKey.get(key);
    const entry = list?.shift();
    if (list && list.length === 0) pendingByKey.delete(key);
    if (entry) {
      const idx = pendingOrder.indexOf(entry);
      if (idx >= 0) pendingOrder.splice(idx, 1);
    }
    return entry;
  };

  const flushUnanswered = () => {
    if (pendingOrder.length === 0) return;
    const synthParts: GeminiPart[] = pendingOrder.map((p) =>
      p.degradedArgs !== undefined
        ? { text: `[Observation from ${p.degradedArgs === "{}" ? p.name : `${p.name} (${p.degradedArgs})`}:\nNo result provided]` }
        : {
            functionResponse: {
              name: p.name,
              response: { error: "No result provided" },
              ...(p.wireId ? { id: p.wireId } : {}),
            },
          },
    );
    pendingOrder.length = 0;
    pendingByKey.clear();
    appendTurn(contents, "user", synthParts);
  };

  for (const msg of messages) {
    if (msg.role === "system") continue;

    if (msg.role === "user") {
      flushUnanswered();
      const parts: GeminiPart[] =
        typeof msg.content === "string"
          ? msg.content.length > 0
            ? [{ text: msg.content.toWellFormed() }]
            : []
          : msg.content.flatMap((item) => {
              if (item.type === "text") return item.text.length > 0 ? [{ text: item.text.toWellFormed() }] : [];
              const img = toImagePart(item, allowImages);
              return img ? [img] : [];
            });
      appendTurn(contents, "user", parts);
      continue;
    }

    if (msg.role === "assistant") {
      if (msg.stopReason === "error" || msg.stopReason === "aborted") continue;
      flushUnanswered();
      const parts: GeminiPart[] = [];
      const isSameModel = msg.provider === PROVIDER_ID && msg.model === model.id;
      const toolCalls = msg.content.filter((b): b is ToolCall => b.type === "toolCall");
      const groupSigned =
        isSameModel &&
        toolCalls.length > 0 &&
        isValidSig(toolCalls[0]?.thoughtSignature) &&
        toolCalls.every((tc) => !tc.thoughtSignature || isValidSig(tc.thoughtSignature));
      const turnHasSig =
        isSameModel &&
        msg.content.some(
          (b) =>
            (b.type === "thinking" && isValidSig(b.thinkingSignature)) ||
            (b.type === "text" && isValidSig(b.textSignature)) ||
            (b.type === "toolCall" && isValidSig(b.thoughtSignature)),
        );

      for (const block of msg.content) {
        if (block.type === "text") {
          const sig = isSameModel && isValidSig(block.textSignature) ? block.textSignature : undefined;
          if (!block.text.trim() && !sig) continue;
          parts.push({ text: block.text.toWellFormed(), ...(sig ? { thoughtSignature: sig } : {}) });
        } else if (block.type === "thinking") {
          const sig = isSameModel && isValidSig(block.thinkingSignature) ? block.thinkingSignature : undefined;
          if (!isSameModel || (!block.thinking.trim() && !sig)) continue;
          if ((requiresSig && !sig && !turnHasSig) || (isClaude && !sig)) continue;
          parts.push({ thought: true, text: block.thinking.toWellFormed(), ...(sig ? { thoughtSignature: sig } : {}) });
        } else if (block.type === "toolCall") {
          const rawId = block.id || "";
          const key = rawId ? `id:${rawId}` : `name:${block.name}`;
          if (requiresSig && !groupSigned) {
            enqueuePending(key, { name: block.name, degradedArgs: JSON.stringify(block.arguments ?? {}) });
          } else {
            const wireId = needsId ? sanitizeToolCallId(rawId, block.name, usedIds) : undefined;
            const callSig = isSameModel && isValidSig(block.thoughtSignature) ? block.thoughtSignature : undefined;
            enqueuePending(key, { name: block.name, wireId });
            parts.push({
              functionCall: {
                name: block.name,
                args: structuredClone((block.arguments ?? {}) as Record<string, unknown>),
                ...(wireId ? { id: wireId } : {}),
              },
              ...(callSig ? { thoughtSignature: callSig } : {}),
            });
          }
        }
      }

      if (parts.length === 0 && requiresSig && !groupSigned && toolCalls.length > 0) {
        parts.push({ text: "[Earlier tool call omitted because its signature is unavailable.]" });
      }
      appendTurn(contents, "model", parts);
      continue;
    }

    if (msg.role === "toolResult") {
      const text = msg.content
        .filter((c): c is TextContent => c.type === "text")
        .map((c) => c.text.toWellFormed())
        .join("\n");
      const responseText = text || (msg.isError ? "Tool failed" : "");
      const images = msg.content
        .filter((c): c is ImageContent => c.type === "image")
        .map((img) => toImagePart(img, allowImages))
        .filter((p): p is GeminiPart => p !== undefined);

      const rawId = msg.toolCallId || "";
      const matched = dequeuePending(rawId, msg.toolName);
      if (matched?.degradedArgs !== undefined) {
        const label = matched.degradedArgs === "{}" ? msg.toolName : `${msg.toolName} (${matched.degradedArgs})`;
        appendTurn(contents, "user", [{ text: `[Observation from ${label}:\n${responseText}]` }, ...images]);
      } else {
        const wireId = needsId ? (matched?.wireId ?? sanitizeToolCallId(rawId, msg.toolName, usedIds)) : undefined;
        appendTurn(contents, "user", [
          {
            functionResponse: {
              name: msg.toolName,
              response: msg.isError ? { error: responseText } : { output: responseText },
              ...(wireId ? { id: wireId } : {}),
            },
          },
          ...images,
        ]);
      }
    }
  }

  flushUnanswered();

  if (contents[0]?.role === "model") {
    contents.unshift({ role: "user", parts: [{ text: CONTINUATION_TEXT }] });
  }
  for (let i = 0; i < contents.length; i++) {
    if (contents[i]!.role === "model" && contents[i]!.parts.some((p) => "functionCall" in p) && contents[i - 1]?.role !== "user") {
      contents.splice(i, 0, { role: "user", parts: [{ text: CONTINUATION_TEXT }] });
      i++;
    }
  }

  if (contents.length > 0 && !contents.some((t) => t.role === "user" && t.parts.some((p) => "text" in p && Boolean(p.text.trim())))) {
    const firstUser = contents.find((t) => t.role === "user");
    if (firstUser) firstUser.parts.push({ text: CONTINUATION_TEXT });
    else contents.unshift({ role: "user", parts: [{ text: CONTINUATION_TEXT }] });
  }

  const last = contents.at(-1);
  if (last?.role === "model") appendTurn(contents, "user", [{ text: CONTINUATION_TEXT }]);
  return contents;
}

const SCHEMA_MAPS = new Set(["properties", "patternProperties", "dependentSchemas", "dependencies"]);
const LITERAL_KEYS = new Set(["enum", "default", "examples"]);
const META_KEYS = new Set(["$schema", "$id", "$anchor", "$dynamicAnchor", "$vocabulary", "$comment", "$defs", "definitions"]);
const CUSTOM_TOOL_ALLOW = new Set(["type", "description", "properties", "required", "items", "enum"]);
const UNSUPPORTED_STRICT_KEYS = [
  "$ref", "$defs", "definitions", "allOf", "oneOf", "patternProperties", "dependentSchemas",
  "dependencies", "unevaluatedProperties", "propertyNames", "contains", "prefixItems", "not", "if", "then", "else",
];

function resolvePointer(ref: string, root: unknown): unknown {
  if (ref === "#") return root;
  if (!ref.startsWith("#/")) return undefined;
  let cur: unknown = root;
  for (const raw of ref.slice(2).split("/")) {
    let decoded: string;
    try {
      decoded = decodeURIComponent(raw);
    } catch {
      return undefined;
    }
    const key = decoded.replace(/~1/g, "/").replace(/~0/g, "~");
    if (Array.isArray(cur)) {
      const idx = Number(key);
      if (!/^(0|[1-9]\d*)$/.test(key) || !Number.isSafeInteger(idx) || idx >= cur.length) return undefined;
      cur = cur[idx];
    } else if (isRecord(cur) && Object.hasOwn(cur, key)) {
      cur = cur[key];
    } else {
      return undefined;
    }
  }
  return cur;
}

const isNullSchema = (b: Record<string, unknown>): boolean =>
  b.type === "null" || b.const === null || (Array.isArray(b.enum) && b.enum.length === 1 && b.enum[0] === null);

function toCustomToolSchema(schema: unknown): unknown {
  if (!schema || typeof schema !== "object") return schema;
  if (Array.isArray(schema)) return schema.map(toCustomToolSchema);
  const rec = schema as Record<string, unknown>;

  const branches = Array.isArray(rec.anyOf) ? rec.anyOf : Array.isArray(rec.oneOf) ? rec.oneOf : undefined;
  if (branches) {
    const nonNull = branches.filter((b): b is Record<string, unknown> => isRecord(b) && !isNullSchema(b));
    if (
      nonNull.length > 0 &&
      nonNull.every((b) => Array.isArray(b.enum) && b.enum.length > 0 && b.enum.every((x) => typeof x === "string"))
    ) {
      return {
        type: "string",
        enum: [...new Set(nonNull.flatMap((b) => b.enum as string[]))],
        ...(typeof rec.description === "string" ? { description: rec.description } : {}),
      };
    }
    if (nonNull.length > 0) {
      const first = toCustomToolSchema(nonNull[0]);
      return isRecord(first) && typeof rec.description === "string" && !first.description
        ? { ...first, description: rec.description }
        : first;
    }
  }

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(rec)) {
    if (!CUSTOM_TOOL_ALLOW.has(k)) continue;
    if (k === "type") {
      const scalar = typeof v === "string" ? v : Array.isArray(v) ? v.find((x): x is string => typeof x === "string" && x !== "null") : undefined;
      if (scalar && scalar !== "null") out.type = scalar;
    } else if (k === "properties" && isRecord(v)) {
      out.properties = Object.fromEntries(Object.entries(v).map(([p, s]) => [p, toCustomToolSchema(s)]));
    } else if (k === "items") {
      if (isRecord(v)) out.items = toCustomToolSchema(v);
      else if (Array.isArray(v)) {
        const objItem = v.find(isRecord);
        out.items = objItem ? toCustomToolSchema(objItem) : { type: "string" };
      }
    } else if (k === "enum") {
      if (Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === "string")) out.enum = [...v];
    } else {
      out[k] = toCustomToolSchema(v);
    }
  }
  if (out.type === "array" && out.items === undefined) out.items = { type: "string" };
  if (out.properties && Array.isArray(out.required)) {
    const valid = (out.required as unknown[]).filter(
      (r): r is string => typeof r === "string" && Object.hasOwn(out.properties as object, r),
    );
    if (valid.length > 0) out.required = valid;
    else delete out.required;
  }
  if (!out.type) out.type = out.properties ? "object" : "string";
  return out;
}

class StrictSchemaError extends Error {}

function schemaAllowsNull(s: unknown): boolean {
  if (!isRecord(s)) return false;
  if (s.type === "null" || (Array.isArray(s.type) && s.type.includes("null"))) return true;
  if (s.const === null || (Array.isArray(s.enum) && s.enum.includes(null))) return true;
  return Array.isArray(s.anyOf) && s.anyOf.some(schemaAllowsNull);
}

function makeStrictSchema(root: unknown): Record<string, unknown> {
  const cloned = structuredClone(root);
  const visit = (s: unknown): void => {
    if (!isRecord(s)) throw new StrictSchemaError("boolean schemas are unsupported");
    for (const key of UNSUPPORTED_STRICT_KEYS) {
      if (s[key] !== undefined) throw new StrictSchemaError(`${key} schemas are unsupported`);
    }
    if (s.anyOf !== undefined) {
      if (!Array.isArray(s.anyOf) || s.anyOf.length === 0) throw new StrictSchemaError("anyOf must contain at least one schema");
      for (const v of s.anyOf) {
        if (isRecord(v) && (v.type === "object" || v.type === "array" || v.properties !== undefined || v.items !== undefined)) {
          throw new StrictSchemaError("object and array unions are unsupported");
        }
        visit(v);
      }
    }
    if (s.items !== undefined) {
      if (Array.isArray(s.items)) throw new StrictSchemaError("tuple schemas are unsupported");
      visit(s.items);
    }
    if (s.type !== "object") {
      if (s.properties !== undefined) throw new StrictSchemaError("properties require type object");
      return;
    }
    if (s.additionalProperties !== undefined && s.additionalProperties !== false) {
      throw new StrictSchemaError("schema-valued or true additionalProperties is unsupported");
    }
    const props = isRecord(s.properties) ? s.properties : {};
    const propNames = Object.keys(props);
    const req = new Set(Array.isArray(s.required) ? (s.required as string[]) : []);
    if ([...req].some((k) => !propNames.includes(k))) throw new StrictSchemaError("required contains an unknown property");
    for (const [k, prop] of Object.entries(props)) {
      visit(prop);
      if (!req.has(k) && !schemaAllowsNull(prop)) props[k] = { anyOf: [prop, { type: "null" }] };
    }
    s.required = propNames;
    s.additionalProperties = false;
  };
  if (!isRecord(cloned) || cloned.type !== "object") throw new StrictSchemaError("root schema must have type object");
  visit(cloned);
  return cloned;
}

export function convertTools(tools: readonly Tool[] | undefined, useCustomBridge = false, supportsStrict = false) {
  if (!tools?.length) return undefined;
  const declarations = tools.map((tool) => {
    const cs = (tool as { constrainedSampling?: { type?: string; strict?: string } }).constrainedSampling;
    let baseParams: unknown = tool.parameters;
    if (cs?.type === "json_schema") {
      if (!supportsStrict || useCustomBridge) {
        if (cs.strict === "require") {
          throw new Error(`Tool "${tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.`);
        }
      } else {
        try {
          baseParams = makeStrictSchema(tool.parameters);
        } catch (err) {
          if (cs.strict === "require") {
            throw new Error(`Tool "${tool.name}" requires JSON-schema constrained sampling, but ${(err as Error).message}.`);
          }
        }
      }
    }

    let nodes = 0;
    const clean = (schema: unknown, refs = new Set<string>(), objs = new Set<object>(), depth = 0): unknown => {
      if (depth > 64 || ++nodes > 10_000) throw new Error(`Tool "${tool.name}" schema limit exceeded`);
      if (!schema || typeof schema !== "object") return schema;
      if (Array.isArray(schema)) return schema.map((v) => clean(v, refs, objs, depth + 1));
      if (objs.has(schema)) throw new Error(`Tool "${tool.name}" circular schema`);
      const nextObjs = new Set(objs).add(schema);
      const rec = schema as Record<string, unknown>;

      if (typeof rec.$ref === "string") {
        if (refs.has(rec.$ref)) throw new Error(`Tool "${tool.name}" circular $ref: ${rec.$ref}`);
        const target = resolvePointer(rec.$ref, baseParams);
        if (target === undefined) throw new Error(`Tool "${tool.name}" unresolved $ref: ${rec.$ref}`);
        const resolved = clean(target, new Set(refs).add(rec.$ref), nextObjs, depth + 1);
        const { $ref: _, ...siblings } = rec;
        if (Object.keys(siblings).length === 0) return resolved;
        const cleanedSiblings = clean(siblings, refs, nextObjs, depth + 1);
        return isRecord(resolved) && isRecord(cleanedSiblings) ? { ...resolved, ...cleanedSiblings } : resolved;
      }

      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(rec)) {
        if (META_KEYS.has(k)) continue;
        if (k === "const") {
          if (v === null) {
            out.type = "null";
          } else {
            out.enum = [structuredClone(v)];
            if ((typeof v === "string" || typeof v === "number" || typeof v === "boolean") && rec.type === undefined) {
              out.type = typeof v;
            }
          }
        } else if (SCHEMA_MAPS.has(k) && isRecord(v)) {
          out[k] = Object.fromEntries(Object.entries(v).map(([p, s]) => [p, clean(s, refs, nextObjs, depth + 1)]));
        } else if (LITERAL_KEYS.has(k)) {
          out[k] = structuredClone(v);
        } else {
          out[k] = clean(v, refs, nextObjs, depth + 1);
        }
      }
      return out;
    };

    const inlined = clean(baseParams);
    const rootObj = isRecord(inlined)
      ? inlined.type
        ? inlined
        : { ...inlined, type: "object", properties: inlined.properties ?? {} }
      : { type: "object", properties: {} };
    return {
      name: tool.name,
      description: tool.description,
      ...(useCustomBridge ? { parameters: toCustomToolSchema(rootObj) } : { parametersJsonSchema: rootObj }),
    };
  });
  return [{ functionDeclarations: declarations }];
}

const i64 = (bytes: Uint8Array): string => String(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigInt64(0, true));

function sessionIdentity(messages: readonly Message[], sessionId?: string) {
  const explicit = sessionId?.trim();
  const toSessionInt = (seed?: string): string => {
    if (!seed) return i64(crypto.getRandomValues(new Uint8Array(8)));
    if (/^-?\d+$/.test(seed)) {
      const v = BigInt(seed);
      if (v >= -(1n << 63n) && v <= (1n << 63n) - 1n) return seed;
    }
    return i64(createHash("sha256").update(`antigravity:session:${seed}`).digest());
  };

  let seed = explicit ? `session:${explicit}` : undefined;
  if (!seed) {
    const first = messages.find((m) => m.role !== "system");
    if (!first) {
      return { conversationId: crypto.randomUUID(), trajectoryId: crypto.randomUUID(), sessionId: toSessionInt() };
    }
    const head = typeof first.content === "string" ? first.content.slice(0, 64) : JSON.stringify(first.content[0] ?? "").slice(0, 64);
    seed = `${first.role}:${first.timestamp}:${head}`;
  }
  return {
    conversationId: stableUuid(`antigravity:conv:${seed}`),
    trajectoryId: stableUuid(`antigravity:traj:${seed}`),
    sessionId: toSessionInt(explicit ?? seed),
  };
}

export function buildRequest(
  model: Model<Api>,
  context: TranscriptContext,
  projectId: string,
  options: SimpleStreamOptions | undefined,
  runtimeModel: string,
  modelEnum?: string,
) {
  const systemPrompt = getCurrentSystemPrompt(context.messages)
    ?.replace(/running inside pi(?:, a coding agent harness)?/gi, "running inside your development environment")
    .replace(/inside pi, a coding agent harness/gi, "inside your development environment");

  const contents = convertMessages(model, context.messages, runtimeModel);
  if (contents.length === 0) {
    contents.push({
      role: "user",
      parts: [{ text: systemPrompt ? "Apply the active system instructions." : CONTINUATION_TEXT }],
    });
  }

  const isClaude = model.id.startsWith("claude-") || runtimeModel.startsWith("claude-");
  const thinking = getThinkingConfig(model, runtimeModel, options?.reasoning);
  const rawMax = options?.maxTokens !== undefined ? Math.min(options.maxTokens, model.maxTokens) : model.maxTokens;
  const maxOutputTokens =
    isClaude && thinking?.includeThoughts ? Math.max(rawMax, thinking.thinkingBudget + 256) : rawMax;
  const allowTemp = options?.temperature !== undefined && (!isClaude || !thinking?.includeThoughts || options.temperature === 1);
  const generationConfig = {
    ...(allowTemp ? { temperature: options!.temperature } : {}),
    ...(thinking ? { thinkingConfig: thinking } : {}),
    maxOutputTokens,
  };
  const isCustom = usesCustomBridge(model.id, runtimeModel);
  const isNonGemini = isCustom || (!model.id.startsWith("gemini-") && !runtimeModel.startsWith("gemini-"));
  const supportsStrict = !isCustom && isModernGemini(runtimeModel);
  const activeTools = getCurrentTools(context.messages);
  const tools = convertTools(activeTools, isCustom, supportsStrict);
  const hasStrictTool =
    supportsStrict &&
    Boolean(
      activeTools?.some((t) => {
        const cs = (t as { constrainedSampling?: { type?: string } }).constrainedSampling;
        if (cs?.type !== "json_schema") return false;
        try {
          makeStrictSchema(t.parameters);
          return true;
        } catch {
          return false;
        }
      }),
    );
  const toolChoice = (options as { toolChoice?: string } | undefined)?.toolChoice;
  const fcMode =
    toolChoice === "none"
      ? ("NONE" as const)
      : toolChoice && toolChoice !== "auto"
        ? ("ANY" as const)
        : hasStrictTool
          ? ("VALIDATED" as const)
          : undefined;

  const step = Math.max(1, contents.length);
  const requestIndex = context.messages.filter(
    (m) => m.role === "assistant" && m.stopReason !== "error" && m.stopReason !== "aborted",
  ).length;
  const { conversationId, trajectoryId, sessionId } = sessionIdentity(context.messages, options?.sessionId);

  return {
    project: projectId,
    model: runtimeModel,
    request: {
      contents,
      ...(systemPrompt ? { systemInstruction: { role: "user" as const, parts: [{ text: systemPrompt.toWellFormed() }] } } : {}),
      generationConfig,
      ...(tools ? { tools, ...(fcMode ? { toolConfig: { functionCallingConfig: { mode: fcMode } } } : {}) } : {}),
      sessionId,
      labels: {
        last_step_index: String(Math.max(0, contents.length - 1)),
        request_id: `${trajectoryId}-${requestIndex}`,
        trajectory_id: trajectoryId,
        used_claude: isClaude ? "true" : "false",
        used_claude_conservative: isClaude ? "true" : "false",
        used_non_gemini_model: isNonGemini ? "true" : "false",
        ...(modelEnum ? { model_enum: modelEnum } : {}),
        ...(step > 1 ? { last_execution_id: stableUuid(`antigravity:exec:${trajectoryId}:${step - 1}`) } : {}),
      },
    },
    requestType: "agent" as const,
    userAgent: "antigravity" as const,
    requestId: `agent/${conversationId}/${Date.now()}/${trajectoryId}/${step}`,
  };
}

const isRateLimit429 = (t: string): boolean =>
  /rate.?limit|too many requests|per minute|requests per|tokens per minute/i.test(t);

const isQuotaWall429 = (t: string): boolean =>
  !isRateLimit429(t) &&
  (/Individual quota reached|Resets? in |AI credits balance is too low|INSUFFICIENT_G1_CREDITS/i.test(t) ||
    /quota exceeded|exceeded your|limit reached|reached your|daily limit/i.test(t));

const isGenericCapacity429 = (t: string): boolean =>
  !isQuotaWall429(t) && !isRateLimit429(t) && /resource has been exhausted|RESOURCE_EXHAUSTED/i.test(t);

const OVERFLOW_RE =
  /exceeds the maximum|context[_ ]length[_ ]exceeded|prompt is too long|too many tokens|maximum context length|request payload size exceeds/i;

function trimTrailingPunct(s: string): string {
  let end = s.length;
  while (end > 0 && ".,;)".includes(s[end - 1]!)) end--;
  return s.slice(0, end);
}

export function formatApiError(status: number | undefined, rawText: string): string {
  let extracted = rawText;
  try {
    const p: unknown = JSON.parse(rawText);
    if (isRecord(p) && isRecord(p.error) && typeof p.error.message === "string") extracted = p.error.message;
  } catch {
    // plain text
  }
  const msg = redactSecrets(extracted).trim();
  const short = msg.slice(0, 500);

  if (status === 429) {
    if (isQuotaWall429(short)) {
      const wait = short.match(/Resets? in ([^.\n]+)/i)?.[1]?.trim();
      return `Antigravity quota reached${wait ? ` (resets in ${wait})` : ""}. Switch models or retry after reset.`;
    }
    if (isGenericCapacity429(short)) {
      const clean = short.replace(/429|RESOURCE_EXHAUSTED|ResourceExhausted/gi, "exhausted");
      return `Antigravity model unavailable right now (${clean || "capacity exhausted"}). Switch models or retry later.`;
    }
    const safeThrottle = short
      .replace(/quota exceeded|exceeded your|too many tokens/gi, "throttled")
      .slice(0, 200);
    return `429 Rate limited by Antigravity${safeThrottle ? `: ${safeThrottle}` : ""}`;
  }

  if (status === 413 || OVERFLOW_RE.test(msg)) {
    return status === 413 ? `413: ${short || "Payload too large"}` : `context_length_exceeded: ${short}`;
  }
  if (status === 401) return "Antigravity authentication failed. Run /login antigravity and retry.";
  if (status === 403) {
    const rawLink = rawText.slice(0, 4096).match(/https:\/\/accounts\.google\.com\/[^\s"'<>\\]{1,500}/)?.[0];
    const safeLink = rawLink ? redactSecrets(trimTrailingPunct(rawLink)) : undefined;
    return `Antigravity access denied: ${short || "forbidden"}.${safeLink ? ` Verify account: ${safeLink}` : ""}`;
  }
  if (status === 408) return `Antigravity request timed out (408): ${short || "timeout"}`;
  if (status !== undefined && status >= 500) return `Antigravity server error (${status}): ${short || "unavailable"}`;
  return status !== undefined ? `Antigravity request failed (${status}): ${short || "unknown error"}` : short || "Antigravity request failed";
}

type StreamPayload = {
  responseId?: string;
  modelVersion?: string;
  candidates?: Array<{
    finishReason?: string;
    content?: {
      parts?: Array<{
        text?: string;
        thought?: boolean;
        thoughtSignature?: string;
        functionCall?: { id?: string; name?: string; args?: unknown };
      }>;
    };
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    thoughtsTokenCount?: number;
    cachedContentTokenCount?: number;
    totalTokenCount?: number;
  };
};
type StreamChunk = { error?: { code?: number; message?: string }; response?: StreamPayload } & StreamPayload;

async function fetchWithHeaderDeadline(
  fetchFn: FetchFunction,
  url: string,
  init: RequestInit,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<Response> {
  if (timeoutMs <= 0) return fetchFn(url, { ...init, signal });
  const headerAbort = new AbortController();
  const combined = signal ? AbortSignal.any([signal, headerAbort.signal]) : headerAbort.signal;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      fetchFn(url, { ...init, signal: combined }),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          const err = new Error(`Antigravity request timed out after ${timeoutMs}ms waiting for headers`);
          reject(err);
          headerAbort.abort(err);
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function readWithDeadline(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  if (signal?.aborted) throw new Error("Request was aborted");
  let timer: ReturnType<typeof setTimeout> | undefined;
  const onAbort = () => void reader.cancel().catch(() => {});
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const read = reader.read();
    if (timeoutMs <= 0) return await read;
    return await Promise.race([
      read,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          void reader.cancel().catch(() => {});
          reject(new Error(`Antigravity stream timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

async function readTextWithDeadline(response: Response, timeoutMs: number, signal?: AbortSignal): Promise<string> {
  if (!response.body) return response.text();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const deadline = timeoutMs > 0 ? Date.now() + timeoutMs : 0;
  let out = "";
  let completed = false;
  try {
    while (true) {
      if (deadline > 0 && Date.now() >= deadline) {
        throw new Error(`Antigravity error response timed out after ${timeoutMs}ms`);
      }
      const rem = deadline > 0 ? Math.max(1, deadline - Date.now()) : 0;
      const { done, value } = await readWithDeadline(reader, rem, signal);
      if (done) break;
      if (value instanceof Uint8Array && out.length < 65_536) out += decoder.decode(value, { stream: true });
    }
    out += decoder.decode();
    completed = true;
    return out;
  } finally {
    if (!completed) void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function consumeSseStream(
  response: Response,
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  model: Model<Api>,
  timeoutMs: number,
  options?: SimpleStreamOptions,
): Promise<void> {
  if (!response.body) throw new Error("Antigravity returned an empty response body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const blocks = output.content;
  const usedIds = new Set<string>();
  let buffer = "";
  let skipLf = false;
  let dataLines: string[] = [];
  let stray = "";
  let started = false;
  let hasContent = false;
  let sawFinishReason = false;
  let completed = false;
  let active: TextContent | ThinkingContent | null = null;

  const ensureStarted = () => {
    if (!started) {
      stream.push({ type: "start", partial: output });
      started = true;
    }
  };

  const finishActive = () => {
    if (!active) return;
    const contentIndex = blocks.length - 1;
    stream.push(
      active.type === "text"
        ? { type: "text_end", contentIndex, content: active.text, partial: output }
        : { type: "thinking_end", contentIndex, content: active.thinking, partial: output },
    );
    active = null;
  };

  const handleLine = async (line: string) => {
    if (options?.signal?.aborted) throw new Error("Request was aborted");
    if (!line.startsWith("data:")) {
      const trimmed = line.trim();
      if (!trimmed) {
        stray = "";
        dataLines = [];
        return;
      }
      if (trimmed.startsWith(":") || /^(?:event|id|retry):/i.test(trimmed)) return;
      if (stray.length < 4096) stray = stray ? `${stray}\n${trimmed}` : trimmed;
      const start = Math.min(
        stray.indexOf("{") >= 0 ? stray.indexOf("{") : Infinity,
        stray.indexOf("[") >= 0 ? stray.indexOf("[") : Infinity,
      );
      if (!Number.isFinite(start)) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(stray.slice(start));
      } catch {
        return;
      }
      const errItem = Array.isArray(parsed) ? parsed[0] : parsed;
      if (isRecord(errItem) && isRecord(errItem.error)) {
        const code = typeof errItem.error.code === "number" ? errItem.error.code : undefined;
        throw new Error(formatApiError(code, JSON.stringify({ error: errItem.error })));
      }
      stray = "";
      return;
    }

    stray = "";
    const partLine = line.slice(5).replace(/^ /, "");
    if (partLine.trim() === "[DONE]") {
      dataLines = [];
      return;
    }
    if (dataLines.length < 256) dataLines.push(partLine);
    const json = dataLines.join("\n").trim();
    if (!json) return;
    let rawChunk: unknown;
    try {
      rawChunk = JSON.parse(json);
    } catch {
      return;
    }
    dataLines = [];
    if (!isRecord(rawChunk)) return;
    const chunk = rawChunk as StreamChunk;

    await options?.onProviderStreamEvent?.(chunk, model);
    if (options?.signal?.aborted) throw new Error("Request was aborted");
    if (chunk.error) throw new Error(formatApiError(chunk.error.code, JSON.stringify({ error: chunk.error })));

    const payload = isRecord(chunk.response) ? chunk.response : chunk;
    if (payload.modelVersion) output.responseModel = payload.modelVersion;
    if (payload.responseId) output.responseId = payload.responseId;
    const candidate = payload.candidates?.[0];

    for (const part of candidate?.content?.parts ?? []) {
      if (part.text !== undefined) {
        const isThinking = part.thought === true;
        hasContent = true;
        const kind = isThinking ? "thinking" : "text";
        if (!active || active.type !== kind) {
          finishActive();
          active = isThinking ? { type: "thinking", thinking: "" } : { type: "text", text: "" };
          blocks.push(active);
          ensureStarted();
          stream.push({ type: isThinking ? "thinking_start" : "text_start", contentIndex: blocks.length - 1, partial: output });
        }
        const contentIndex = blocks.length - 1;
        if (active.type === "thinking") {
          active.thinking += part.text;
          if (part.thoughtSignature) active.thinkingSignature = part.thoughtSignature;
          if (part.text) stream.push({ type: "thinking_delta", contentIndex, delta: part.text, partial: output });
        } else {
          active.text += part.text;
          if (part.thoughtSignature) active.textSignature = part.thoughtSignature;
          if (part.text) stream.push({ type: "text_delta", contentIndex, delta: part.text, partial: output });
        }
      }

      if (part.functionCall) {
        hasContent = true;
        finishActive();
        const args: JsonObject = isRecord(part.functionCall.args) ? (part.functionCall.args as JsonObject) : {};
        const toolCall: ToolCall = {
          type: "toolCall",
          id: sanitizeToolCallId(part.functionCall.id ?? "", part.functionCall.name, usedIds),
          name: part.functionCall.name ?? "",
          arguments: args,
          ...(part.thoughtSignature ? { thoughtSignature: part.thoughtSignature } : {}),
        };
        blocks.push(toolCall);
        const contentIndex = blocks.length - 1;
        ensureStarted();
        stream.push({ type: "toolcall_start", contentIndex, partial: output });
        stream.push({ type: "toolcall_delta", contentIndex, delta: JSON.stringify(args), partial: output });
        stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
      }
    }

    if (candidate?.finishReason) {
      sawFinishReason = true;
      output.rawStopReason = candidate.finishReason;
      const mapped = candidate.finishReason === "STOP" ? "stop" : candidate.finishReason === "MAX_TOKENS" ? "length" : "error";
      output.stopReason = mapped === "stop" && blocks.some((b) => b.type === "toolCall") ? "toolUse" : mapped;
      if (output.stopReason === "length") ensureStarted();
    }

    if (payload.usageMetadata) {
      const prompt = payload.usageMetadata.promptTokenCount ?? 0;
      const cacheRead = payload.usageMetadata.cachedContentTokenCount ?? 0;
      const thoughts = payload.usageMetadata.thoughtsTokenCount ?? 0;
      output.usage.input = Math.max(0, prompt - cacheRead);
      output.usage.output = (payload.usageMetadata.candidatesTokenCount ?? 0) + thoughts;
      output.usage.reasoning = thoughts;
      output.usage.cacheRead = cacheRead;
      output.usage.totalTokens = payload.usageMetadata.totalTokenCount ?? 0;
      calculateCost(model, output.usage);
    }
  };

  const appendDecoded = (chunk: string) => {
    if (skipLf && chunk.startsWith("\n")) chunk = chunk.slice(1);
    skipLf = chunk.endsWith("\r");
    buffer += chunk.replace(/\r\n|\r/g, "\n");
  };

  try {
    while (true) {
      const { done, value } = await readWithDeadline(reader, timeoutMs, options?.signal);
      if (options?.signal?.aborted) throw new Error("Request was aborted");
      if (done) break;
      if (!(value instanceof Uint8Array)) continue;
      appendDecoded(decoder.decode(value, { stream: true }));
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) await handleLine(line);
    }
    appendDecoded(decoder.decode());
    if (buffer.length > 0) await handleLine(buffer);
    if (options?.signal?.aborted) throw new Error("Request was aborted");
    completed = true;
    finishActive();
  } finally {
    if (!completed) void reader.cancel().catch(() => {});
    reader.releaseLock();
  }

  if (!sawFinishReason) throw new Error("Antigravity stream ended without a finish reason");
  if (!hasContent && output.stopReason === "stop") {
    throw new Error("Antigravity stream ended without any content");
  }
}

export function createStreamAntigravity(
  getModelEnum?: (runtimeModel: string) => string | undefined,
  defaultFetch?: FetchFunction,
) {
  return function streamAntigravity(
    model: Model<Api>,
    context: TranscriptContext,
    options?: SimpleStreamOptions,
  ): AssistantMessageEventStream {
    const stream = createAssistantMessageEventStream();

    void (async () => {
      const zero = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
      const output: AssistantMessage = {
        role: "assistant",
        content: [],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: { ...zero, totalTokens: 0, cost: { ...zero, total: 0 } },
        stopReason: "pending",
        timestamp: Date.now(),
      };

      try {
        if (options?.signal?.aborted) throw new Error("Request was aborted");
        const creds = parseApiKey(options?.apiKey);
        const fetchFn = options?.fetch ?? defaultFetch ?? fetch;
        const timeoutMs = options?.timeoutMs ?? DEFAULT_STREAM_TIMEOUT_MS;
        const headers = buildHeaders(creds.token, model.headers, options?.headers);
        const candidates = [resolveRuntimeModel(model, options?.reasoning)];
        const endpoints = endpointsFor(model.baseUrl);

        let response: Response | undefined;
        let primaryStatus: number | undefined;
        let primaryBody = "";
        let primaryTransportError: Error | undefined;

        for (let i = 0; i < candidates.length; i++) {
          const runtimeModel = candidates[i]!;
          response = undefined;
          let sawCapacity429 = false;
          let tierStatus: number | undefined;
          let tierBody = "";

          for (let epIdx = 0; epIdx < endpoints.length; epIdx++) {
            if (options?.signal?.aborted) throw new Error("Request was aborted");
            let payload: unknown = buildRequest(model, context, creds.projectId, options, runtimeModel, getModelEnum?.(runtimeModel));
            const rewritten = await options?.onPayload?.(payload, model);
            if (rewritten !== undefined) payload = rewritten;

            try {
              response = await fetchWithHeaderDeadline(
                fetchFn,
                `${endpoints[epIdx]!}/v1internal:streamGenerateContent?alt=sse`,
                { method: "POST", headers, body: JSON.stringify(payload) },
                options?.signal,
                timeoutMs,
              );
            } catch (err) {
              if (options?.signal?.aborted) throw err;
              const transportErr = err instanceof Error ? err : new Error(String(err));
              if (i === 0) primaryTransportError ??= transportErr;
              continue;
            }

            try {
              await options?.onResponse?.(
                { status: response.status, headers: Object.fromEntries(response.headers.entries()) },
                model,
              );
            } catch (err) {
              void response.body?.cancel().catch(() => {});
              throw err;
            }

            if (response.ok) break;
            const status = response.status;
            const bodyText = await readTextWithDeadline(response, timeoutMs, options?.signal).catch(() => "");

            if (epIdx === 0) {
              tierStatus = status;
              tierBody = bodyText;
            }
            if (status === 429 && isGenericCapacity429(bodyText)) {
              sawCapacity429 = true;
              continue;
            }
            if (![403, 404, 500, 502, 503, 504].includes(status)) {
              tierStatus = status;
              tierBody = bodyText;
              break;
            }
          }

          if (response?.ok) break;
          if (i === 0 && tierStatus !== undefined) {
            primaryStatus = tierStatus;
            primaryBody = tierBody;
          }

          if (sawCapacity429 && candidates.length < 4) {
            const nextSibling = siblingRuntimeModels(model, runtimeModel).find((id) => !candidates.includes(id));
            if (nextSibling) {
              candidates.splice(i + 1, 0, nextSibling);
              continue;
            }
          }
          break;
        }

        if (!response?.ok) {
          if (primaryStatus === undefined && primaryTransportError) throw primaryTransportError;
          throw new Error(formatApiError(primaryStatus ?? response?.status, primaryBody));
        }

        await consumeSseStream(response, stream, output, model, timeoutMs, options);
        if (options?.signal?.aborted) throw new Error("Request was aborted");

        if (output.stopReason === "error" || output.stopReason === "aborted") {
          output.errorMessage ||= output.rawStopReason ? `Provider stopped with: ${output.rawStopReason}` : "Antigravity generation failed";
          stream.push({ type: "error", reason: output.stopReason, error: output });
        } else {
          stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
        }
        stream.end();
      } catch (err) {
        output.stopReason = options?.signal?.aborted ? "aborted" : "error";
        output.errorMessage = redactSecrets(err instanceof Error ? err.message : String(err));
        stream.push({ type: "error", reason: output.stopReason, error: output });
        stream.end();
      }
    })();

    return stream;
  };
}
