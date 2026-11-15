import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { isContextOverflow, isRetryableAssistantError, type Message, type TranscriptContext } from "@earendil-works/pi-ai";
import { createAntigravityProvider } from "../src/index.ts";
import { encodeApiKey } from "../src/client.ts";
import { FALLBACK_MODELS } from "../src/models.ts";
import { buildRequest, convertMessages, convertTools, createStreamAntigravity, formatApiError } from "../src/stream.ts";

const tx = (messages: Message[]): TranscriptContext => ({ messages }) as unknown as TranscriptContext;
const hydrogen = FALLBACK_MODELS.find((m) => m.id === "gemini-4-hydrogen")!;
const claude = FALLBACK_MODELS.find((m) => m.id === "claude-sonnet-5-5")!;
const gptOss = FALLBACK_MODELS.find((m) => m.id === "gpt-oss-120b")!;

describe("stream request & SSE handling", () => {
  it("rewords rejected Pi boilerplate without dropping instructions, paths, or tools", () => {
    const system = {
      role: "system" as const,
      content: "",
      sections: {
        preamble: "You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.",
        docs: "Read /opt/pi-coding-agent/docs for codemode scripts and non-LLM models such as classifiers and image models (docs/codemode.md).",
        project_context: "Never send email on my behalf.",
      },
      toolsAdded: [{ name: "bash", description: "Execute commands", parameters: { type: "object", properties: {} } }],
      timestamp: 1,
    };
    const before = structuredClone(system);
    const context = tx([system, { role: "user", content: "hi", timestamp: 2 }] as Message[]);
    const request = buildRequest(hydrogen, context, "project", { reasoning: "high" }, "gemini-4-hydrogen-high").request;
    const prompt = request.systemInstruction!.parts[0]!.text;
    assert.ok(prompt.includes("You are a coding assistant. Help users inspect files, run commands, and modify code."));
    assert.ok(prompt.includes("Read /opt/pi-coding-agent/docs for codemode scripts, classifiers, and image models (docs/codemode.md)."));
    assert.ok(prompt.includes("Never send email on my behalf."));
    assert.equal(request.tools![0]!.functionDeclarations[0]!.name, "bash");
    assert.deepEqual(system, before);

    const custom = "Discuss Pi and non-LLM models. Keep /opt/pi-coding-agent/docs unchanged.";
    const unchanged = buildRequest(hydrogen, tx([{ role: "system", content: custom, timestamp: 1 }]), "project", undefined, "gemini-4-hydrogen-low");
    assert.equal(unchanged.request.systemInstruction!.parts[0]!.text, custom);
  });

  it("converts percent-encoded $ref, multi-branch unions, tuples, and strict sampling", () => {
    const tools = [
      {
        name: "edit",
        description: "Edit file",
        parameters: {
          $defs: { "My Edit": { type: "object", properties: { old: { type: "string" } }, required: ["old"] } },
          type: "object",
          properties: {
            mode: { const: "replace" },
            target: { anyOf: [{ type: "string" }, { const: null }], description: "path" },
            multi: { anyOf: [{ type: "string" }, { type: "array", items: { type: "string" } }] },
            pair: { type: "array", items: [{ type: "string" }, { type: "number" }] },
            edits: { type: "array", items: { $ref: "#/$defs/My%20Edit" } },
          },
        } as never,
      },
    ];
    const geminiProps = (
      convertTools(tools, false)?.[0]?.functionDeclarations[0] as unknown as {
        parametersJsonSchema: { properties: Record<string, unknown> };
      }
    ).parametersJsonSchema.properties;
    assert.deepEqual(geminiProps.mode, { enum: ["replace"], type: "string" });

    const claudeProps = (
      convertTools(tools, true)?.[0]?.functionDeclarations[0] as unknown as {
        parameters: { properties: Record<string, unknown> };
      }
    ).parameters.properties;
    assert.deepEqual(claudeProps.target, { type: "string", description: "path" });
    assert.deepEqual(claudeProps.multi, { type: "string" });
    assert.deepEqual(claudeProps.pair, { type: "array", items: { type: "string" } });

    const strictTool = [
      {
        name: "strictTool",
        description: "strict",
        parameters: { type: "object", properties: { x: { type: "string" } } } as never,
        constrainedSampling: { type: "json_schema" as const, strict: "require" as const },
      },
    ];
    assert.throws(() => convertTools(strictTool, true, false), /strict tools are unsupported/);
    const strictGem = convertTools(strictTool, false, true)?.[0]?.functionDeclarations[0] as unknown as {
      parametersJsonSchema: { required: string[]; additionalProperties: boolean };
    };
    assert.deepEqual(strictGem.parametersJsonSchema.required, ["x"]);
    assert.equal(strictGem.parametersJsonSchema.additionalProperties, false);
  });

  it("recovers both interrupted and trailing unanswered tool calls and handles whitespace-only prompts", () => {
    const trailing = [
      { role: "user" as const, content: "run tool", timestamp: 1 },
      {
        role: "assistant" as const,
        api: "antigravity-api" as const,
        provider: "antigravity",
        model: "claude-sonnet-5-5",
        content: [{ type: "toolCall" as const, id: "tc1", name: "bash", arguments: { cmd: "ls" } }],
        usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: "toolUse" as const,
        timestamp: 2,
      },
    ];
    const recovered = convertMessages(claude, trailing, "claude-sonnet-5-5-low");
    assert.equal(recovered.length, 3);
    assert.ok("functionResponse" in recovered[2]!.parts[0]!);

    const blankReq = buildRequest(hydrogen, tx([{ role: "user", content: "   ", timestamp: 1 }]), "proj", undefined, "gemini-4-hydrogen-low");
    assert.ok(blankReq.request.contents.length >= 1);

    const gptTurns = convertMessages(
      gptOss,
      [{ role: "user", content: [{ type: "image", data: "AAAA", mimeType: "image/png" }], timestamp: 1 }],
      "gpt-oss-120b-medium",
    );
    assert.match(JSON.stringify(gptTurns), /Image omitted/);
  });

  it("aligns formatApiError with Pi 1.5 retry and context-overflow classifiers", () => {
    const rpm = formatApiError(429, "Quota exceeded for quota metric requests per minute (rate limit)");
    const tpm = formatApiError(429, "Too many tokens per minute");
    const empty429 = formatApiError(429, "");
    const quota = formatApiError(429, "Individual quota reached. Resets in 3h 20m.");
    const overflow = formatApiError(400, "Request payload size exceeds the limit: 20971520 bytes");

    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: rpm } as never), true);
    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: tpm } as never), true);
    assert.equal(isContextOverflow({ stopReason: "error", errorMessage: tpm, usage: { input: 0, cacheRead: 0 } } as never), false);
    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: empty429 } as never), true);
    assert.equal(isRetryableAssistantError({ stopReason: "error", errorMessage: quota } as never), false);
    assert.equal(isContextOverflow({ stopReason: "error", errorMessage: overflow, usage: { input: 0, cacheRead: 0 } } as never), true);
  });

  it("preserves primary transport errors, fails over when fallback hits capacity 429, handles multi-line SSE, and preserves MAX_TOKENS", async () => {
    const prov = createAntigravityProvider(async (url) => {
      if (String(url).includes("daily-cloudcode-pa.googleapis.com")) throw new TypeError("fetch failed");
      return new Response("fallback routing error", { status: 403 });
    });
    const transportRes = await prov.streamSimple(hydrogen, tx([{ role: "user", content: "hi", timestamp: 1 }]), {
      apiKey: encodeApiKey("tok_1", "proj_1"),
    }).result();
    assert.equal(isRetryableAssistantError(transportRes), true);

    const streamFn = createStreamAntigravity();
    const sibRes = await streamFn(hydrogen, tx([{ role: "user", content: "hi", timestamp: 1 }]), {
      apiKey: encodeApiKey("tok_1", "proj_1"),
      fetch: async (url, init) => {
        const body = JSON.parse(String(init?.body)) as { model: string };
        if (body.model === "gemini-4-hydrogen-low") {
          if (String(url).includes("daily-cloudcode-pa.googleapis.com")) throw new TypeError("fetch failed");
          return new Response('{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}', { status: 429 });
        }
        return new Response('data: {"candidates":[{"content":{"parts":[{"text":"sib_ok"}]},"finishReason":"STOP"}]}\n\n', { status: 200 });
      },
    }).result();
    assert.equal(sibRes.stopReason, "stop");
    assert.deepEqual(sibRes.content, [{ type: "text", text: "sib_ok" }]);

    const lenRes = await streamFn(hydrogen, tx([{ role: "user", content: "hi", timestamp: 1 }]), {
      apiKey: encodeApiKey("tok_1", "proj_1"),
      fetch: async () => new Response('data: {"candidates":[{"finishReason":"MAX_TOKENS"}]}\n\n', { status: 200 }),
    }).result();
    assert.equal(lenRes.stopReason, "length");

    const multiSse = [
      'data: {"candidates":[',
      'data: {"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}',
      "",
      "",
    ].join("\r");
    const okRes = await streamFn(hydrogen, tx([{ role: "user", content: "hi", timestamp: 1 }]), {
      apiKey: encodeApiKey("tok_1", "proj_1"),
      fetch: async () => new Response(multiSse, { status: 200 }),
    }).result();
    assert.equal(okRes.stopReason, "stop");
    assert.deepEqual(okRes.content, [{ type: "text", text: "ok" }]);
  });

  it("cancels hung error bodies and wedged stream cancellations cleanly", async () => {
    let errorBodyCancelled = 0;
    const streamFn = createStreamAntigravity();
    await streamFn({ ...hydrogen, baseUrl: "https://custom.example.com" }, tx([{ role: "user", content: "hi", timestamp: 1 }]), {
      apiKey: encodeApiKey("tok_1", "proj_1"),
      timeoutMs: 15,
      fetch: async () =>
        new Response(
          new ReadableStream<Uint8Array>({
            pull: () => new Promise(() => {}),
            cancel: () => {
              errorBodyCancelled++;
            },
          }),
          { status: 401 },
        ),
    }).result();
    assert.equal(errorBodyCancelled, 1);
  });
});
