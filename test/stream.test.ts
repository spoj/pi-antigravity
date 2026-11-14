import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { isContextOverflow, isRetryableAssistantError, type Message, type TranscriptContext } from "@earendil-works/pi-ai";
import { encodeApiKey } from "../src/client.ts";
import { FALLBACK_MODELS } from "../src/models.ts";
import { buildRequest, convertMessages, convertTools, createStreamAntigravity, formatApiError } from "../src/stream.ts";

const tx = (messages: Message[]): TranscriptContext => ({ messages }) as unknown as TranscriptContext;

const hydrogen = FALLBACK_MODELS.find((m) => m.id === "gemini-4-hydrogen")!;
const claude = FALLBACK_MODELS.find((m) => m.id === "claude-sonnet-5-5")!;
const gptOss = FALLBACK_MODELS.find((m) => m.id === "gpt-oss-120b")!;

describe("stream request & SSE handling", () => {
  it("converts $ref, const->enum, and Type.Union schemas for Gemini and Claude", () => {
    const tools = [
      {
        name: "edit",
        description: "Edit file",
        parameters: {
          $schema: "http://json-schema.org/draft-07/schema#",
          $defs: { Edit: { type: "object", properties: { old: { type: "string" } }, required: ["old"] } },
          type: "object",
          properties: {
            mode: { const: "replace" },
            target: { anyOf: [{ type: "string" }, { type: "null" }], description: "path" },
            edits: { type: "array", items: { $ref: "#/$defs/Edit" } },
          },
        } as never,
      },
    ];
    const geminiTools = convertTools(tools, false);
    const geminiProps = (
      geminiTools?.[0]?.functionDeclarations[0] as unknown as { parametersJsonSchema: { properties: Record<string, unknown> } }
    ).parametersJsonSchema.properties;
    assert.deepEqual(geminiProps.mode, { enum: ["replace"], type: "string" });

    const claudeTools = convertTools(tools, true);
    const claudeProps = (
      claudeTools?.[0]?.functionDeclarations[0] as unknown as { parameters: { properties: Record<string, unknown> } }
    ).parameters.properties;
    assert.deepEqual(claudeProps.target, { type: "string", description: "path" });
  });

  it("synthesizes missing tool results on interrupted parallel batches, strips foreign sigs, and omits unsupported images", () => {
    const history = [
      { role: "user" as const, content: "run two tools", timestamp: 1 },
      {
        role: "assistant" as const,
        api: "antigravity-api" as const,
        provider: "antigravity",
        model: "gemini-4-hydrogen",
        content: [
          { type: "toolCall" as const, id: "tc1", name: "bash", arguments: { cmd: "ls" }, thoughtSignature: "AAAA" },
          { type: "toolCall" as const, id: "tc2", name: "bash", arguments: { cmd: "pwd" } },
        ],
        usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: "toolUse" as const,
        timestamp: 2,
      },
      {
        role: "toolResult" as const,
        toolCallId: "tc1",
        toolName: "bash",
        content: [{ type: "text" as const, text: "file.txt" }],
        isError: false,
        timestamp: 3,
      },
      {
        role: "user" as const,
        content: [
          { type: "text" as const, text: "now check" },
          { type: "image" as const, data: "AAAA", mimeType: "image/png" },
        ],
        timestamp: 4,
      },
    ];

    const gemTurns = convertMessages(hydrogen, history, "gemini-4-hydrogen-low");
    assert.equal(gemTurns.length, 3);
    const userResponses = gemTurns[2]!.parts.filter((p) => "functionResponse" in p);
    assert.equal(userResponses.length, 2);

    const gptTurns = convertMessages(gptOss, history, "gpt-oss-120b-medium");
    const callPart = gptTurns[1]!.parts[0] as { functionCall: unknown; thoughtSignature?: string };
    assert.equal(callPart.thoughtSignature, undefined);
    assert.match(JSON.stringify(gptTurns[2]!.parts), /Image omitted/);
  });

  it("clones toolCall arguments so onPayload mutations do not corrupt history", () => {
    const args = { cmd: "pwd" };
    const ctx = tx([
        { role: "user" as const, content: "go", timestamp: 1 },
        {
          role: "assistant" as const,
          api: "antigravity-api" as const,
          provider: "antigravity",
          model: "claude-sonnet-5-5",
          content: [{ type: "toolCall" as const, id: "c1", name: "bash", arguments: args }],
          usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
          stopReason: "toolUse" as const,
          timestamp: 2,
        },
        { role: "toolResult" as const, toolCallId: "c1", toolName: "bash", content: [{ type: "text" as const, text: "/tmp" }], isError: false, timestamp: 3 },
      ]);
    const req = buildRequest(claude, ctx, "proj", undefined, "claude-sonnet-5-5-low");
    const wireCall = (req.request.contents[1]!.parts[0] as { functionCall: { args: Record<string, unknown> } }).functionCall;
    wireCall.args.cmd = "mutated";
    assert.equal(args.cmd, "pwd");
  });

  it("aligns formatApiError with Pi 1.5 retry and context-overflow classifiers", () => {
    const rpm = formatApiError(429, "Quota exceeded for quota metric requests per minute (rate limit)");
    const tpm = formatApiError(429, "Too many tokens per minute");
    const empty429 = formatApiError(429, "");
    const quota = formatApiError(429, "Individual quota reached. Resets in 3h 20m.");
    const cap = formatApiError(429, '{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}');
    const overflow = formatApiError(400, "Request payload size exceeds the limit: 20971520 bytes");

    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: rpm } as never), true);
    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: tpm } as never), true);
    assert.equal(isContextOverflow({ stopReason: "error", errorMessage: tpm, usage: { input: 0, cacheRead: 0 } } as never), false);
    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: empty429 } as never), true);
    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: quota } as never), false);
    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: cap } as never), false);
    assert.equal(isContextOverflow({ stopReason: "error", errorMessage: overflow, usage: { input: 0, cacheRead: 0 } } as never), true);
  });

  it("fails over across sibling tier even when sandbox returns 403 and prod returns 404", async () => {
    const attemptedModels: string[] = [];
    const streamFn = createStreamAntigravity(() => "MODEL_PLACEHOLDER_M527");

    const s = streamFn(
      hydrogen,
      tx([{ role: "system", content: "You are running inside pi, a coding agent harness.", timestamp: 1 }, { role: "user", content: "hi", timestamp: 2 }]),
      {
        apiKey: encodeApiKey("tok_1", "proj_1"),
        fetch: async (url, init) => {
          const u = String(url);
          const body = JSON.parse(String(init?.body)) as { model: string };
          attemptedModels.push(`${new URL(u).host}:${body.model}`);
          if (body.model === "gemini-4-hydrogen-low") {
            if (u.includes("sandbox")) return new Response("Forbidden", { status: 403 });
            if (u.includes("https://cloudcode-pa")) return new Response("Not found", { status: 404 });
            return new Response('{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}', { status: 429 });
          }
          const sse = [
            'data: {"response":{"modelVersion":"gemini-4-hydrogen-high","candidates":[{"content":{"parts":[{"text":"think","thought":true,"thoughtSignature":"BBBB"},{"text":"ok"}]}}]}}',
            'data: {"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"thoughtsTokenCount":3,"totalTokenCount":15}}}',
            "",
          ].join("\n");
          return new Response(sse, { status: 200, headers: { "Content-Type": "text/event-stream" } });
        },
      },
    );

    const msg = await s.result();
    assert.equal(msg.stopReason, "stop");
    assert.ok(attemptedModels.includes("daily-cloudcode-pa.googleapis.com:gemini-4-hydrogen-high"));
  });

  it("does not hang when reader.cancel() never settles on mid-stream error", async () => {
    const streamFn = createStreamAntigravity();
    const s = streamFn(
      hydrogen,
      tx([{ role: "user", content: "hi", timestamp: 1 }]),
      {
        apiKey: encodeApiKey("tok_1", "proj_1"),
        timeoutMs: 50,
        fetch: async () => {
          let sent = false;
          const body = new ReadableStream<Uint8Array>({
            pull(ctrl) {
              if (!sent) {
                sent = true;
                ctrl.enqueue(new TextEncoder().encode('data: {"error":{"code":500,"message":"boom"}}\n'));
              }
            },
            cancel: () => new Promise<void>(() => {}),
          });
          return new Response(body, { status: 200 });
        },
      },
    );
    const res = await s.result();
    assert.equal(res.stopReason, "error");
    assert.match(res.errorMessage ?? "", /500/);
  });
});
