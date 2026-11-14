import assert from "node:assert/strict";
import test from "node:test";
import { isContextOverflow, normalizeContext, type AssistantMessage, type AssistantMessageEvent, type ToolResultMessage } from "@earendil-works/pi-ai";
import { encodeApiKey } from "../src/client.ts";
import { FALLBACK_MODELS } from "../src/models.ts";
import { buildRequest, convertMessages, convertTools, createStreamAntigravity, formatApiError } from "../src/stream.ts";

const hydrogen = FALLBACK_MODELS.find((m) => m.id === "gemini-4-hydrogen")!;
const sonnet = FALLBACK_MODELS.find((m) => m.id === "claude-sonnet-5-5")!;
const gptOss = FALLBACK_MODELS.find((m) => m.id === "gpt-oss-120b")!;
const zeroUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };

const asst = (over: Partial<AssistantMessage> & Pick<AssistantMessage, "content">): AssistantMessage => ({
  role: "assistant",
  provider: "antigravity",
  model: "gemini-4-hydrogen",
  api: "antigravity-api",
  stopReason: "toolUse",
  timestamp: 2,
  usage: zeroUsage,
  ...over,
});

const toolRes = (toolCallId: string, text: string, toolName = "bash"): ToolResultMessage => ({
  role: "toolResult",
  toolCallId,
  toolName,
  content: [{ type: "text", text }],
  isError: false,
  timestamp: 3,
});

test("buildRequest sanitizes Pi system prompt and populates Antigravity envelope", () => {
  const ctx = normalizeContext({
    systemPrompt: "You are a coding assistant running inside pi, a coding agent harness.",
    messages: [{ role: "user", content: "hi", timestamp: 1 }],
    tools: [
      {
        name: "ping",
        description: "ping host",
        parameters: {
          $schema: "https://json-schema.org/draft/2020-12/schema",
          $defs: { Target: { type: "string" } },
          type: "object",
          properties: { definitions: { $ref: "#/$defs/Target" } },
        },
      },
    ],
  });

  const req = buildRequest(hydrogen, ctx, "proj-1", { reasoning: "high", sessionId: "sess-1" }, "gemini-4-hydrogen-high", "M525");
  assert.equal(req.project, "proj-1");
  assert.equal(req.model, "gemini-4-hydrogen-high");
  assert.equal(req.request.systemInstruction?.parts[0]?.text, "You are a coding assistant running inside your development environment.");
  assert.equal(req.request.labels.model_enum, "M525");
  const decl = req.request.tools?.[0]?.functionDeclarations[0];
  assert(decl && "parametersJsonSchema" in decl);
  assert.deepEqual(decl.parametersJsonSchema, {
    type: "object",
    properties: { definitions: { type: "string" } },
  });
});

test("convertTools normalizes schemas for Claude and GPT-OSS and skips cyclic $ref tools", () => {
  const tools = convertTools(
    [
      {
        name: "good",
        description: "ok",
        parameters: {
          type: "object",
          additionalProperties: false,
          properties: { mode: { type: ["string", "null"], enum: ["fast", "slow"], format: "uri" }, badEnum: { type: "number", enum: [1, 2] } },
          required: ["mode"],
        },
      },
      { name: "cyclic", description: "loop", parameters: { $defs: { A: { $ref: "#/$defs/A" } }, $ref: "#/$defs/A" } },
    ],
    true,
  );
  assert.equal(tools?.[0]?.functionDeclarations.length, 1);
  const decl = tools?.[0]?.functionDeclarations[0];
  assert(decl && "parameters" in decl);
  assert.deepEqual(decl.parameters, {
    type: "object",
    properties: { mode: { type: "string", enum: ["fast", "slow"] }, badEnum: { type: "number" } },
    required: ["mode"],
  });

  const ctx = normalizeContext({
    messages: [{ role: "user", content: "go", timestamp: 1 }],
    tools: [{ name: "t", description: "d", parameters: { type: "object", nullable: true, properties: {} } }],
  });
  const gptReq = buildRequest(gptOss, ctx, "p", { reasoning: "medium" }, "gpt-oss-120b-medium");
  assert("parameters" in (gptReq.request.tools?.[0]?.functionDeclarations[0] ?? {}));
});

test("convertMessages handles signed/unsigned Gemini calls, empty IDs, and paired Claude tool IDs", () => {
  const sig = "QUJDREVGR0hJSktMTU5PUA==";
  const signed = convertMessages(
    hydrogen,
    [
      { role: "user", content: "ls", timestamp: 1 },
      asst({ content: [{ type: "thinking", thinking: "", thinkingSignature: sig }, { type: "toolCall", id: "c1", name: "bash", arguments: { cmd: "ls" }, thoughtSignature: sig }] }),
      toolRes("c1", "file.txt"),
    ],
    "gemini-4-hydrogen-high",
  );
  assert.equal(signed.length, 3);
  assert("thought" in signed[1]!.parts[0]!);
  assert("functionCall" in signed[1]!.parts[1]!);

  // Unsigned empty-ID Gemini tool calls degrade with their matching empty-ID toolResults
  const foreignEmpty = convertMessages(
    hydrogen,
    [
      { role: "user", content: "ls", timestamp: 1 },
      asst({ provider: "openai", model: "gpt-6.1-sol", content: [{ type: "toolCall", id: "", name: "bash", arguments: { cmd: "pwd" } }] }),
      toolRes("", "/tmp"),
    ],
    "gemini-4-hydrogen-high",
  );
  assert("text" in foreignEmpty[2]!.parts[0]! && foreignEmpty[2]!.parts[0]!.text.includes("[Observation from bash"));

  // Claude empty-ID historical calls pair synthetic IDs across call and response
  const claudeTurns = convertMessages(
    sonnet,
    [
      { role: "user", content: "hi", timestamp: 1 },
      asst({ model: "claude-sonnet-5-5", content: [{ type: "toolCall", id: "", name: "bash", arguments: {} }] }),
      toolRes("", "ok"),
    ],
    "claude-sonnet-5-5-high",
  );
  const callPart = claudeTurns[1]!.parts[0] as { functionCall: { id: string } };
  const respPart = claudeTurns[2]!.parts[0] as { functionResponse: { id: string } };
  assert.equal(callPart.functionCall.id, respPart.functionResponse.id);
});

test("formatApiError preserves Pi overflow detection, redacts 403 links, and classifies 429s", () => {
  const err413 = formatApiError(413, "Request Entity Too Large");
  assert(isContextOverflow(asst({ content: [], stopReason: "error", errorMessage: err413 }), 1_048_576));
  assert(formatApiError(400, '{"error":{"message":"input token count exceeds the maximum"}}').includes("exceeds the maximum"));

  const err403 = formatApiError(403, "Verify https://accounts.google.com/signin/continue?access_token=ya29.secret123.");
  assert(err403.includes("https://accounts.google.com/signin/continue?access_token=[REDACTED]"));
  assert(!err403.includes("ya29.secret123"));

  assert(!formatApiError(429, "Individual quota reached. Resets in 2h.").includes("429"));
  assert(!/429|ResourceExhausted/i.test(formatApiError(429, "Resource has been exhausted.")));
  assert(formatApiError(429, "Too many requests, rate limit").includes("429"));
});

test("streamAntigravity streams SSE, preserves MAX_TOKENS/SAFETY stopReasons, fails over siblings, and honors timeoutMs: 0", async () => {
  const sse = (chunks: string[]) => {
    const enc = new TextEncoder();
    return new Response(
      new ReadableStream({
        start(ctrl) {
          for (const c of chunks) ctrl.enqueue(enc.encode(c));
          ctrl.close();
        },
      }),
      { status: 200, headers: { "x-ag-test": "ok" } },
    );
  };

  const attempted: string[] = [];
  let responseStatus = 0;
  let rawEvents = 0;
  const streamFn = createStreamAntigravity(() => "M527");
  const ctx = normalizeContext({ messages: [{ role: "user", content: "hi", timestamp: 1 }] });

  const s = streamFn(hydrogen, ctx, {
    apiKey: encodeApiKey("tok", "proj-1"),
    reasoning: "low",
    timeoutMs: 0,
    fetch: async (_url, init) => {
      const body = JSON.parse(String(init?.body)) as { model: string };
      attempted.push(body.model);
      if (body.model === "gemini-4-hydrogen-low") {
        return new Response('{"error":{"message":"Resource has been exhausted"}}', { status: 429 });
      }
      return sse([
        'data: {"response":{"modelVersion":"gemini-4-hydrogen-high","candidates":[{"content":{"parts":[{"thought":true,"text":"plan ","thoughtSignature":"U0lHMTIzNDU2Nzg="}]}}]}}\n\n',
        'data: {"response":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"thought":true,"text":"more"},{"text":"ok"}]}}],"usageMetadata":{"promptTokenCount":50,"cachedContentTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":3,"totalTokenCount":58}}}\n\n',
      ]);
    },
    onResponse: (r) => {
      responseStatus = r.status;
    },
    onProviderStreamEvent: () => {
      rawEvents++;
    },
  });

  const events: AssistantMessageEvent[] = [];
  for await (const ev of s) events.push(ev);
  const final = await s.result();
  assert.equal(responseStatus, 200);
  assert.equal(rawEvents, 2);
  assert.deepEqual([attempted[0], attempted.at(-1)], ["gemini-4-hydrogen-low", "gemini-4-hydrogen-high"]);
  assert.equal(final.stopReason, "stop");
  assert.deepEqual(final.content, [
    { type: "thinking", thinking: "plan more", thinkingSignature: "U0lHMTIzNDU2Nzg=" },
    { type: "text", text: "ok" },
  ]);

  // MAX_TOKENS with truncated toolCall stays "length", not "toolUse"
  const trunc = await streamFn(hydrogen, ctx, {
    apiKey: encodeApiKey("tok", "proj-1"),
    fetch: async () =>
      sse(['data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS","content":{"parts":[{"functionCall":{"name":"bash","args":{"cmd":"ls"}}}]}}]}}\n\n']),
  }).result();
  assert.equal(trunc.stopReason, "length");

  // Zero-part SAFETY finishReason surfaces rawStopReason without "ended without" retry trigger
  const safety = await streamFn(hydrogen, ctx, {
    apiKey: encodeApiKey("tok", "proj-1"),
    fetch: async () => sse(['data: {"response":{"candidates":[{"finishReason":"SAFETY"}]}}\n\n']),
  }).result();
  assert.equal(safety.stopReason, "error");
  assert.equal(safety.errorMessage, "Provider stopped with: SAFETY");
});
