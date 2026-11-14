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

function sanitizeToolCallId(id: string, fallbackName?: string): string {
  const cleaned = id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  if (cleaned) return cleaned;
  const prefix = (fallbackName ?? "tool").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 32) || "tool";
  return `${prefix}_${++toolIdCounter}`;
}

function toImagePart(block: ImageContent): GeminiPart | undefined {
  if (!block.data) return undefined;
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

export function convertMessages(
  model: Model<Api>,
  messages: readonly Message[],
  runtimeModel: string,
): GeminiContent[] {
  const contents: GeminiContent[] = [];
  const requiresSig = runtimeModel.startsWith("gemini-") && !/^gemini-[12](?:[.-]|$)/.test(runtimeModel);
  const needsId = usesCustomBridge(model.id, runtimeModel);
  const droppedById = new Map<string, string>();
  const droppedEmptyByName = new Map<string, string[]>();
  const synthIdByName = new Map<string, string[]>();

  for (const msg of messages) {
    if (msg.role === "system") continue;

    if (msg.role === "user") {
      const parts: GeminiPart[] =
        typeof msg.content === "string"
          ? msg.content.trim()
            ? [{ text: msg.content.toWellFormed() }]
            : []
          : msg.content.flatMap((item) => {
              if (item.type === "text") return item.text.trim() ? [{ text: item.text.toWellFormed() }] : [];
              const img = toImagePart(item);
              return img ? [img] : [];
            });
      appendTurn(contents, "user", parts);
      continue;
    }

    if (msg.role === "assistant") {
      if (msg.stopReason === "error" || msg.stopReason === "aborted") continue;
      const parts: GeminiPart[] = [];
      const isSameModel = msg.provider === PROVIDER_ID && msg.model === model.id;
      const toolCalls = msg.content.filter((b): b is ToolCall => b.type === "toolCall");
      const groupSigned =
        isSameModel &&
        toolCalls.length > 0 &&
        isValidSig(toolCalls[0]?.thoughtSignature) &&
        toolCalls.every((tc) => !tc.thoughtSignature || isValidSig(tc.thoughtSignature));

      for (const block of msg.content) {
        if (block.type === "text") {
          const sig = isSameModel && isValidSig(block.textSignature) ? block.textSignature : undefined;
          if (!block.text.trim() && !sig) continue;
          parts.push({ text: block.text.toWellFormed(), ...(sig ? { thoughtSignature: sig } : {}) });
        } else if (block.type === "thinking") {
          const sig = isSameModel && isValidSig(block.thinkingSignature) ? block.thinkingSignature : undefined;
          if (!isSameModel || (!block.thinking.trim() && !sig) || (requiresSig && !sig && !groupSigned)) continue;
          parts.push({ thought: true, text: block.thinking.toWellFormed(), ...(sig ? { thoughtSignature: sig } : {}) });
        } else if (block.type === "toolCall") {
          const rawId = block.id || "";
          if (requiresSig && !groupSigned) {
            const argsJson = JSON.stringify(block.arguments ?? {});
            if (rawId) droppedById.set(rawId, argsJson);
            else droppedEmptyByName.set(block.name, [...(droppedEmptyByName.get(block.name) ?? []), argsJson]);
          } else {
            if (rawId) droppedById.delete(rawId);
            else droppedEmptyByName.delete(block.name);
            let wireId: string | undefined;
            if (needsId) {
              wireId = sanitizeToolCallId(rawId, block.name);
              if (!rawId.replace(/[^a-zA-Z0-9_-]/g, "")) {
                synthIdByName.set(block.name, [...(synthIdByName.get(block.name) ?? []), wireId]);
              }
            }
            parts.push({
              functionCall: {
                name: block.name,
                args: (block.arguments ?? {}) as Record<string, unknown>,
                ...(wireId ? { id: wireId } : {}),
              },
              ...(block.thoughtSignature ? { thoughtSignature: block.thoughtSignature } : {}),
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
        .map(toImagePart)
        .filter((p): p is GeminiPart => p !== undefined);

      const rawId = msg.toolCallId || "";
      const droppedArgs = requiresSig ? (rawId ? droppedById.get(rawId) : droppedEmptyByName.get(msg.toolName)?.shift()) : undefined;
      if (droppedArgs !== undefined) {
        if (rawId) droppedById.delete(rawId);
        const label = droppedArgs === "{}" ? msg.toolName : `${msg.toolName} (${droppedArgs})`;
        appendTurn(contents, "user", [{ text: `[Observation from ${label}:\n${responseText}]` }, ...images]);
      } else {
        const wireId = needsId
          ? (!rawId.replace(/[^a-zA-Z0-9_-]/g, "") ? synthIdByName.get(msg.toolName)?.shift() : undefined) ??
            sanitizeToolCallId(rawId, msg.toolName)
          : undefined;
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
  if (last?.role === "model") {
    if (last.parts.some((p) => "functionCall" in p)) {
      throw new Error("Antigravity request is missing tool result(s) for the final assistant tool call.");
    }
    appendTurn(contents, "user", [{ text: CONTINUATION_TEXT }]);
  }

  return contents;
}

const SCHEMA_MAPS = new Set(["properties", "patternProperties", "dependentSchemas", "dependencies"]);
const LITERAL_KEYS = new Set(["enum", "const", "default", "examples"]);
const META_KEYS = new Set(["$schema", "$id", "$anchor", "$dynamicAnchor", "$vocabulary", "$comment", "$defs", "definitions"]);
const CUSTOM_TOOL_ALLOW = new Set(["type", "description", "properties", "required", "items", "enum"]);

function resolvePointer(ref: string, root: unknown): unknown {
  if (ref === "#") return root;
  if (!ref.startsWith("#/")) return undefined;
  let cur: unknown = root;
  for (const raw of ref.slice(2).split("/")) {
    const key = raw.replace(/~1/g, "/").replace(/~0/g, "~");
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

function toCustomToolSchema(schema: unknown): unknown {
  if (!schema || typeof schema !== "object") return schema;
  if (Array.isArray(schema)) return schema.map(toCustomToolSchema);
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(schema)) {
    if (!CUSTOM_TOOL_ALLOW.has(k)) continue;
    if (k === "type") {
      const scalar = typeof v === "string" ? v : Array.isArray(v) ? v.find((x): x is string => typeof x === "string" && x !== "null") : undefined;
      if (scalar) out.type = scalar;
    } else if (k === "properties" && isRecord(v)) {
      out.properties = Object.fromEntries(Object.entries(v).map(([p, s]) => [p, toCustomToolSchema(s)]));
    } else if (k === "enum") {
      if (Array.isArray(v) && v.every((x) => typeof x === "string")) out.enum = v;
    } else {
      out[k] = toCustomToolSchema(v);
    }
  }
  return out;
}

export function convertTools(tools: readonly Tool[] | undefined, useCustomBridge = false) {
  if (!tools?.length) return undefined;
  const declarations = tools.flatMap((tool) => {
    let nodes = 0;
    const clean = (schema: unknown, refs = new Set<string>(), objs = new Set<object>(), depth = 0): unknown => {
      if (depth > 64 || ++nodes > 10_000) throw new Error("schema limit exceeded");
      if (!schema || typeof schema !== "object") return schema;
      if (Array.isArray(schema)) return schema.map((v) => clean(v, refs, objs, depth + 1));
      if (objs.has(schema)) throw new Error("circular schema");
      const nextObjs = new Set(objs).add(schema);
      const rec = schema as Record<string, unknown>;

      if (typeof rec.$ref === "string") {
        if (refs.has(rec.$ref)) throw new Error("circular $ref");
        const target = resolvePointer(rec.$ref, tool.parameters);
        if (target === undefined) throw new Error("unresolved $ref");
        const resolved = clean(target, new Set(refs).add(rec.$ref), nextObjs, depth + 1);
        const { $ref: _, ...siblings } = rec;
        if (Object.keys(siblings).length === 0) return resolved;
        const cleanedSiblings = clean(siblings, refs, nextObjs, depth + 1);
        return isRecord(resolved) && isRecord(cleanedSiblings) ? { ...resolved, ...cleanedSiblings } : resolved;
      }

      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(rec)) {
        if (META_KEYS.has(k)) continue;
        if (SCHEMA_MAPS.has(k) && isRecord(v)) {
          out[k] = Object.fromEntries(Object.entries(v).map(([p, s]) => [p, clean(s, refs, nextObjs, depth + 1)]));
        } else if (!LITERAL_KEYS.has(k)) {
          out[k] = clean(v, refs, nextObjs, depth + 1);
        } else {
          out[k] = v;
        }
      }
      return out;
    };

    let inlined: unknown;
    try {
      inlined = clean(tool.parameters);
    } catch {
      return [];
    }
    const rootObj = isRecord(inlined)
      ? inlined.type
        ? inlined
        : { ...inlined, type: "object", properties: inlined.properties ?? {} }
      : { type: "object", properties: {} };
    return [
      {
        name: tool.name,
        description: tool.description,
        ...(useCustomBridge ? { parameters: toCustomToolSchema(rootObj) } : { parametersJsonSchema: rootObj }),
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : undefined;
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
  if (contents.length === 0 && systemPrompt) {
    contents.push({ role: "user", parts: [{ text: "Apply the active system instructions." }] });
  }

  const thinking = getThinkingConfig(model, runtimeModel, options?.reasoning);
  const generationConfig = {
    ...(options?.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(thinking ? { thinkingConfig: thinking } : {}),
    maxOutputTokens: options?.maxTokens !== undefined ? Math.min(options.maxTokens, model.maxTokens) : model.maxTokens,
  };

  const isClaude = model.id.startsWith("claude-") || runtimeModel.startsWith("claude-");
  const isCustom = usesCustomBridge(model.id, runtimeModel);
  const isNonGemini = isCustom || (!model.id.startsWith("gemini-") && !runtimeModel.startsWith("gemini-"));
  const tools = convertTools(getCurrentTools(context.messages), isCustom);
  const toolChoice = (options as { toolChoice?: string } | undefined)?.toolChoice;

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
      ...(tools ? { tools } : {}),
      ...(toolChoice && toolChoice !== "auto"
        ? { toolConfig: { functionCallingConfig: { mode: toolChoice === "none" ? ("NONE" as const) : ("ANY" as const) } } }
        : {}),
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

const isQuotaWall429 = (t: string): boolean =>
  /Individual quota reached|Resets? in |AI credits balance is too low|INSUFFICIENT_G1_CREDITS/i.test(t) ||
  (!/rate.?limit/i.test(t) && /quota exceeded|exceeded your|limit reached|reached your|daily limit/i.test(t));

const isGenericCapacity429 = (t: string): boolean =>
  !isQuotaWall429(t) && !/rate.?limit|too many requests/i.test(t) && /resource has been exhausted|RESOURCE_EXHAUSTED/i.test(t);

const OVERFLOW_RE =
  /exceeds the maximum|context[_ ]length[_ ]exceeded|prompt is too long|too many tokens|maximum context length|request payload size exceeds/i;

export function formatApiError(status: number | undefined, rawText: string): string {
  let extracted = rawText;
  try {
    const p: unknown = JSON.parse(rawText);
    if (isRecord(p) && isRecord(p.error) && typeof p.error.message === "string") extracted = p.error.message;
  } catch {
    // plain text
  }
  const msg = redactSecrets(extracted).trim();
  if (status === 413 || OVERFLOW_RE.test(msg)) return status === 413 ? `413: ${msg || "Payload too large"}` : msg;
  const short = msg.slice(0, 500);
  if (status === 401) return "Antigravity authentication failed. Run /login antigravity and retry.";
  if (status === 403) {
    const rawLink = rawText.match(/https:\/\/accounts\.google\.com\/[^\s"'<>\\]+/)?.[0];
    const safeLink = rawLink ? redactSecrets(rawLink.replace(/[.,;)]+$/, "")) : undefined;
    return `Antigravity access denied: ${short || "forbidden"}.${safeLink ? ` Verify account: ${safeLink}` : ""}`;
  }
  if (status === 408) return `Antigravity request timed out (408): ${short || "timeout"}`;
  if (status === 429) {
    if (/rate.?limit|too many requests/i.test(short)) return `429 Rate limited by Antigravity: ${short}`;
    const wait = short.match(/Resets? in ([^.\n]+)/i)?.[1]?.trim();
    if (isQuotaWall429(short)) return `Antigravity quota reached${wait ? ` (resets in ${wait})` : ""}. Switch models or retry after reset.`;
    const clean = short.replace(/429|RESOURCE_EXHAUSTED|ResourceExhausted/gi, "exhausted");
    return `Antigravity model unavailable right now (${clean || "capacity exhausted"}). Switch models or retry later.`;
  }
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
          headerAbort.abort();
          reject(new Error(`Antigravity request timed out after ${timeoutMs}ms waiting for headers`));
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
  let buffer = "";
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
    if (!line.startsWith("data:")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(":")) return;
      stray = stray ? `${stray}\n${trimmed}` : trimmed;
      const brace = stray.indexOf("{");
      if (brace < 0) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(stray.slice(brace));
      } catch {
        return;
      }
      if (isRecord(parsed) && "error" in parsed) {
        const code = isRecord(parsed.error) && typeof parsed.error.code === "number" ? parsed.error.code : undefined;
        throw new Error(formatApiError(code, stray.slice(brace)));
      }
      return;
    }

    const json = line.slice(5).trim();
    if (!json || json === "[DONE]") return;
    let chunk: StreamChunk;
    try {
      chunk = JSON.parse(json) as StreamChunk;
    } catch {
      return;
    }

    await options?.onProviderStreamEvent?.(chunk, model);
    if (chunk.error) throw new Error(formatApiError(chunk.error.code, JSON.stringify({ error: chunk.error })));

    const payload = chunk.response ?? chunk;
    if (payload.modelVersion) output.responseModel = payload.modelVersion;
    if (payload.responseId) output.responseId = payload.responseId;
    const candidate = payload.candidates?.[0];

    for (const part of candidate?.content?.parts ?? []) {
      if (part.text !== undefined) {
        hasContent = true;
        const isThinking = part.thought === true;
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
          stream.push({ type: "thinking_delta", contentIndex, delta: part.text, partial: output });
        } else {
          active.text += part.text;
          if (part.thoughtSignature) active.textSignature = part.thoughtSignature;
          stream.push({ type: "text_delta", contentIndex, delta: part.text, partial: output });
        }
      }

      if (part.functionCall) {
        hasContent = true;
        finishActive();
        const args: JsonObject = isRecord(part.functionCall.args) ? (part.functionCall.args as JsonObject) : {};
        const toolCall: ToolCall = {
          type: "toolCall",
          id: sanitizeToolCallId(part.functionCall.id ?? "", part.functionCall.name),
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

  try {
    while (true) {
      const { done, value } = await readWithDeadline(reader, timeoutMs, options?.signal);
      if (options?.signal?.aborted) throw new Error("Request was aborted");
      if (done) break;
      if (!(value instanceof Uint8Array)) continue;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) await handleLine(line.replace(/\r$/, ""));
    }
    buffer += decoder.decode();
    if (buffer.length > 0) await handleLine(buffer.replace(/\r$/, ""));
    completed = true;
  } finally {
    finishActive();
    if (!completed) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }

  if (!sawFinishReason) throw new Error("Antigravity stream ended without a finish reason");
  if (!hasContent && output.stopReason !== "error" && output.stopReason !== "aborted") {
    throw new Error("Antigravity stream ended without any content");
  }
}

export function createStreamAntigravity(getModelEnum?: (runtimeModel: string) => string | undefined) {
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
        const fetchFn = options?.fetch ?? fetch;
        const timeoutMs = options?.timeoutMs ?? DEFAULT_STREAM_TIMEOUT_MS;
        const headers = buildHeaders(creds.token, model.headers, options?.headers);
        const candidates = [resolveRuntimeModel(model, options?.reasoning)];
        const endpoints = endpointsFor(model.baseUrl);

        let response: Response | undefined;
        let lastStatus: number | undefined;
        let lastBody = "";

        for (let i = 0; i < candidates.length; i++) {
          const runtimeModel = candidates[i]!;
          response = undefined;
          let lastTransportError: Error | undefined;

          for (const endpoint of endpoints) {
            if (options?.signal?.aborted) throw new Error("Request was aborted");
            let payload: unknown = buildRequest(model, context, creds.projectId, options, runtimeModel, getModelEnum?.(runtimeModel));
            const rewritten = await options?.onPayload?.(payload, model);
            if (rewritten !== undefined) payload = rewritten;

            try {
              response = await fetchWithHeaderDeadline(
                fetchFn,
                `${endpoint}/v1internal:streamGenerateContent?alt=sse`,
                { method: "POST", headers, body: JSON.stringify(payload) },
                options?.signal,
                timeoutMs,
              );
            } catch (err) {
              if (options?.signal?.aborted) throw err;
              lastTransportError = err instanceof Error ? err : new Error(String(err));
              continue;
            }

            if (response.ok) break;
            lastStatus = response.status;
            lastBody = await response.text();
            if (response.status === 429 && isGenericCapacity429(lastBody)) continue;
            if (![403, 404, 500, 502, 503, 504].includes(response.status)) break;
          }

          if (!response && lastTransportError) throw lastTransportError;
          if (response?.ok) break;

          if (response?.status === 429 && isGenericCapacity429(lastBody) && candidates.length < 4) {
            const nextSibling = siblingRuntimeModels(model, runtimeModel).find((id) => !candidates.includes(id));
            if (nextSibling) {
              candidates.splice(i + 1, 0, nextSibling);
              continue;
            }
          }
          break;
        }

        if (response) {
          try {
            await options?.onResponse?.({ status: response.status, headers: Object.fromEntries(response.headers.entries()) }, model);
          } catch (err) {
            await response.body?.cancel().catch(() => {});
            throw err;
          }
        }

        if (!response || !response.ok) throw new Error(formatApiError(lastStatus ?? response?.status, lastBody));

        await consumeSseStream(response, stream, output, model, timeoutMs, options);

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
