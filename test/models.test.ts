import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  buildCatalog,
  createCatalogManager,
  FALLBACK_MODELS,
  getThinkingConfig,
  resolveRuntimeModel,
  siblingRuntimeModels,
} from "../src/models.ts";

describe("models catalog & runtime routing", () => {
  it("groups compound suffixes (-extra-low / -x-high), compound display names, and strips image models", () => {
    const { models, enums } = buildCatalog({
      "gemini-3.5-flash-extra-low": { displayName: "Gemini 3.5 Flash", supportsThinking: true },
      "gemini-3.5-flash-low": { displayName: "Gemini 3.5 Flash (Low)", supportsThinking: true },
      "gemini-4-hydrogen-low": {
        displayName: "Gemini 4 Hydrogen (Thinking) (Low)",
        maxTokens: 1048576,
        maxOutputTokens: 262144,
        supportsThinking: true,
        supportsImages: true,
        model: "MODEL_PLACEHOLDER_M527",
      },
      "gemini-4-hydrogen-x-high": {
        displayName: "Gemini 4 Hydrogen",
        maxTokens: 1048576,
        maxOutputTokens: 262144,
        supportsThinking: true,
        supportsImages: true,
        model: "MODEL_PLACEHOLDER_M528",
      },
      "gemini-3-pro-image": { displayName: "Gemini 3 Pro Image", supportsImages: true },
    });

    assert.deepEqual(models.map((m) => m.id), ["gemini-3.5-flash", "gemini-4-hydrogen"]);
    const h = models[1]!;
    assert.equal(enums["gemini-4-hydrogen-low"], "MODEL_PLACEHOLDER_M527");
    assert.equal(resolveRuntimeModel(h, "xhigh"), "gemini-4-hydrogen-x-high");
    assert.equal(resolveRuntimeModel({ ...h, reasoning: false }, "off"), "gemini-4-hydrogen-low");
    assert.deepEqual(siblingRuntimeModels(h, "gemini-4-hydrogen-low"), ["gemini-4-hydrogen-x-high"]);
  });

  it("preserves non-thinking unsuffixed runtime for off and handles shared -tiered budgets", () => {
    const { models } = buildCatalog({
      "claude-sonnet-4-5": { displayName: "Claude Sonnet 4.5", supportsThinking: false },
      "claude-sonnet-4-5-thinking": { displayName: "Claude Sonnet 4.5 (Thinking)", supportsThinking: true },
      "gemini-4-shared-tiered": { displayName: "Gemini 4 Shared", supportsThinking: true },
    });
    const sonnet = models.find((m) => m.id === "claude-sonnet-4-5")!;
    assert.equal(resolveRuntimeModel(sonnet, "off"), "claude-sonnet-4-5");
    assert.equal(resolveRuntimeModel({ ...sonnet, reasoning: false }, "off"), "claude-sonnet-4-5");
    assert.deepEqual(siblingRuntimeModels(sonnet, "claude-sonnet-4-5-thinking"), ["claude-sonnet-4-5"]);

    const tiered = models.find((m) => m.id === "gemini-4-shared")!;
    assert.deepEqual(getThinkingConfig(tiered, "gemini-4-shared-tiered", "low"), { includeThoughts: true, thinkingBudget: 1000 });
    assert.deepEqual(getThinkingConfig(tiered, "gemini-4-shared-tiered", "medium"), { includeThoughts: true, thinkingBudget: 4000 });
    assert.deepEqual(getThinkingConfig(tiered, "gemini-4-shared-tiered", "high"), { includeThoughts: true, thinkingBudget: -1 });
  });

  it("rejects legacy pre-0.14 store entries without enums and invalidates on baseUrl change", async () => {
    let netCalls = 0;
    const mgr = createCatalogManager(async () => {
      netCalls++;
      return new Response(
        JSON.stringify({ models: { "gpt-oss-120b-medium": { displayName: "GPT-OSS 120B (Medium)", supportsImages: false } } }),
        { status: 200 },
      );
    });
    const publish = async ({ update }: { update?: () => void }) => {
      update?.();
      return true;
    };

    await mgr.refreshModels({
      stored: { models: [{ ...FALLBACK_MODELS[0]!, thinkingLevelMap: { low: "low" } as never }], checkedAt: Date.now() },
      allowNetwork: false,
      force: false,
      signal: new AbortController().signal,
      publish,
    });
    assert.equal(resolveRuntimeModel(mgr.getModels()[0]!, "low"), "gemini-4-hydrogen-low");

    const cred = { type: "oauth" as const, access: "tok", refresh: "ref", expires: Date.now() + 60_000, projectId: "proj-1" };
    const stored = {
      models: [...FALLBACK_MODELS],
      checkedAt: Date.now(),
      enums: { "gemini-4-hydrogen-low": "M527" },
      projectId: "proj-1",
      baseUrl: "https://daily-cloudcode-pa.googleapis.com",
    };
    await mgr.refreshModels({ stored, credential: cred, allowNetwork: true, force: false, signal: new AbortController().signal, publish });
    assert.equal(netCalls, 0);

    await mgr.refreshModels({
      stored,
      baseUrl: "https://custom.example.com",
      credential: cred,
      allowNetwork: true,
      force: false,
      signal: new AbortController().signal,
      publish,
    });
    assert.equal(netCalls, 1);
  });
});
