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
  it("groups variant suffixes, handles compound display names, and strips image models", () => {
    const { models, enums } = buildCatalog(
      {
        "gemini-4-hydrogen-low": {
          displayName: "Gemini 4 Hydrogen (Thinking) (Low)",
          maxTokens: 1048576,
          maxOutputTokens: 262144,
          supportsThinking: true,
          supportsImages: true,
          model: "MODEL_PLACEHOLDER_M527",
        },
        "gemini-4-hydrogen-high": {
          displayName: "Gemini 4 Hydrogen (Thinking)",
          maxTokens: 1048576,
          maxOutputTokens: 262144,
          supportsThinking: true,
          supportsImages: true,
          model: "MODEL_PLACEHOLDER_M525",
        },
        "gemini-3-pro-image": { displayName: "Gemini 3 Pro Image", supportsImages: true },
        "chat_internal": { displayName: "Internal" },
      },
      new Set(["gemini-4-hydrogen-low", "gemini-4-hydrogen-high"]),
    );

    assert.equal(models.length, 1);
    const h = models[0]!;
    assert.equal(h.id, "gemini-4-hydrogen");
    assert.equal(enums["gemini-4-hydrogen-low"], "MODEL_PLACEHOLDER_M527");
    assert.equal(resolveRuntimeModel(h, "off"), "gemini-4-hydrogen-low");
    assert.equal(resolveRuntimeModel(h, "high"), "gemini-4-hydrogen-high");
    assert.equal(resolveRuntimeModel({ ...h, reasoning: false }, "off"), "gemini-4-hydrogen-low");
    assert.deepEqual(siblingRuntimeModels(h, "gemini-4-hydrogen-low"), ["gemini-4-hydrogen-high"]);
    assert.equal(getThinkingConfig(h, "gemini-4-hydrogen-low", "off"), undefined);
    assert.deepEqual(getThinkingConfig(h, "gemini-4-hydrogen-high", "off"), {
      includeThoughts: true,
      thinkingBudget: -1,
    });
  });

  it("maps unsuffixed high-tier sibling budget on 429 failover", () => {
    const { models } = buildCatalog({
      "gemini-4-custom-low": { displayName: "Gemini 4 Custom (Low)", supportsThinking: true },
      "gemini-4-custom": { displayName: "Gemini 4 Custom (Thinking)", supportsThinking: true },
    });
    const m = models[0]!;
    assert.deepEqual(getThinkingConfig(m, "gemini-4-custom", "low"), {
      includeThoughts: true,
      thinkingBudget: -1,
    });
  });

  it("invalidates cached catalog when projectId or baseUrl changes", async () => {
    let netCalls = 0;
    const mgr = createCatalogManager(async () => {
      netCalls++;
      return new Response(
        JSON.stringify({ models: { "gpt-oss-120b-medium": { displayName: "GPT-OSS 120B (Medium)", supportsImages: false } } }),
        { status: 200 },
      );
    });
    const cred = { type: "oauth" as const, access: "tok", refresh: "ref", expires: Date.now() + 60_000, projectId: "proj-1" };
    const stored = {
      models: [...FALLBACK_MODELS],
      checkedAt: Date.now(),
      enums: { "gemini-4-hydrogen-low": "M527" },
      projectId: "proj-1",
      baseUrl: "https://daily-cloudcode-pa.googleapis.com",
    };

    const publish = async ({ update }: { update?: () => void }) => {
      update?.();
      return true;
    };
    await mgr.refreshModels({
      stored,
      credential: cred,
      allowNetwork: true,
      force: false,
      signal: new AbortController().signal,
      publish,
    });
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
