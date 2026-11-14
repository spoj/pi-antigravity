import assert from "node:assert/strict";
import test from "node:test";
import { getSupportedThinkingLevels, type ModelsStoreEntry } from "@earendil-works/pi-ai";
import { encodeApiKey } from "../src/client.ts";
import {
  FALLBACK_MODELS,
  buildCatalog,
  createCatalogManager,
  getThinkingConfig,
  resolveRuntimeModel,
  siblingRuntimeModels,
} from "../src/models.ts";

const hydrogen = FALLBACK_MODELS.find((m) => m.id === "gemini-4-hydrogen")!;
const flash38 = FALLBACK_MODELS.find((m) => m.id === "gemini-3.8-flash")!;
const opus = FALLBACK_MODELS.find((m) => m.id === "claude-opus-5-5")!;
const gptOss = FALLBACK_MODELS.find((m) => m.id === "gpt-oss-120b")!;

test("FALLBACK_MODELS and cold-start FALLBACK_ENUMS cover all 7 active 2026 models", () => {
  assert.deepEqual(
    FALLBACK_MODELS.map((m) => m.id),
    ["gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon", "gemini-3.8-flash", "claude-opus-5-5", "claude-sonnet-5-5", "gpt-oss-120b"],
  );
  assert.deepEqual(getSupportedThinkingLevels(hydrogen), ["low", "medium", "high", "xhigh"]);
  assert.equal(resolveRuntimeModel(hydrogen, "xhigh"), "gemini-4-hydrogen-xhigh");
  assert.equal(resolveRuntimeModel(hydrogen, undefined), "gemini-4-hydrogen-low");
  assert.equal(createCatalogManager().getModelEnum("gemini-4-hydrogen-high"), "MODEL_PLACEHOLDER_M525");
});

test("buildCatalog groups variants, filters internal/disabled/image entries, and takes min limits", () => {
  const built = buildCatalog(
    {
      "gemini-4-hydrogen-low": { displayName: "Gemini 4 Hydrogen (Low)", maxTokens: 1_048_576, maxOutputTokens: 262_144, supportsThinking: true, supportsImages: true, model: "M527" },
      "gemini-4-hydrogen-xhigh": { displayName: "Gemini 4 Hydrogen (XHigh)", maxTokens: 1_048_576, maxOutputTokens: 131_072, supportsThinking: true, supportsImages: true, model: "M528" },
      "gemini-4-hydrogen-disabled": { disabled: true },
      "gemini-3.1-flash-image": { displayName: "Gemini 3.1 Flash Image" },
      chat_internal: { isInternal: true },
      "retired-old-low": { displayName: "Retired (Low)", supportsThinking: true },
    },
    new Set(["gemini-4-hydrogen-low", "gemini-4-hydrogen-xhigh"]),
  );
  assert.equal(built.models.length, 1);
  const m = built.models[0]!;
  assert.equal(m.name, "Gemini 4 Hydrogen (Antigravity)");
  assert.equal(m.maxTokens, 131_072);
  assert.deepEqual(getSupportedThinkingLevels(m), ["low", "xhigh"]);
  assert.equal(built.enums["gemini-4-hydrogen-xhigh"], "M528");
  assert.deepEqual(siblingRuntimeModels(m, "gemini-4-hydrogen-low"), ["gemini-4-hydrogen-xhigh"]);
});

test("resolveRuntimeModel supports pre-0.14 thinkingLevelMap and getThinkingConfig clamps/guards", () => {
  const legacy = { ...opus, thinkingLevelMap: { off: null, minimal: null, low: "low", medium: "medium", high: "high", xhigh: null, max: null } };
  assert.equal(resolveRuntimeModel(legacy, "high"), "claude-opus-5-5-high");

  assert.equal(getThinkingConfig({ ...hydrogen, reasoning: false }, "gemini-4-hydrogen-low", "high"), undefined);
  assert.equal(getThinkingConfig(hydrogen, "gemini-4-hydrogen-low", undefined), undefined);
  assert.equal(getThinkingConfig(hydrogen, "gemini-4-hydrogen-low", "off"), undefined);
  assert.deepEqual(getThinkingConfig(flash38, "gemini-3.8-flash-low", "off"), { includeThoughts: false, thinkingBudget: 0 });
  assert.deepEqual(getThinkingConfig(hydrogen, "gemini-4-hydrogen-high", "low"), { includeThoughts: true, thinkingBudget: -1 });
  assert.deepEqual(getThinkingConfig(opus, "claude-opus-5-5-medium", "medium"), { includeThoughts: true, thinkingBudget: 1024 });
  assert.deepEqual(getThinkingConfig(gptOss, "gpt-oss-120b-medium", "medium"), { includeThoughts: true, thinkingBudget: 8192 });
});

test("createCatalogManager hydrates offline, respects 4h TTL, ignores malformed non-forced keys, and surfaces force errors", async () => {
  let hits = 0;
  let fail = false;
  const fakeFetch: typeof fetch = async () => {
    hits++;
    if (fail) return new Response("boom", { status: 500 });
    return Response.json({
      models: {
        "gemini-4-argon-high": { displayName: "Gemini 4 Argon (High)", maxTokens: 1_048_576, maxOutputTokens: 65_536, supportsThinking: true, model: "M417" },
      },
    });
  };

  const catalog = createCatalogManager(fakeFetch);
  let persisted: ModelsStoreEntry | undefined;
  const signal = new AbortController().signal;
  const refresh = (projectId: string, extra?: Partial<Parameters<typeof catalog.refreshModels>[0]>) =>
    catalog.refreshModels({
      credential: { type: "api_key", key: encodeApiKey("tok", projectId) },
      stored: persisted,
      allowNetwork: true,
      signal,
      async publish(pub) {
        if (pub.persist) persisted = pub.persist;
        pub.update?.();
        return true;
      },
      ...extra,
    });

  await refresh("proj-1");
  assert.equal(hits, 3);
  assert.equal(catalog.getModels()[0]?.id, "gemini-4-argon");
  assert.equal(catalog.getModelEnum("gemini-4-argon-high"), "M417");

  // Offline hydration on fresh instance
  const cold = createCatalogManager(fakeFetch);
  await cold.refreshModels({
    stored: persisted,
    allowNetwork: false,
    signal,
    async publish(pub) {
      pub.update?.();
      return true;
    },
  });
  assert.equal(cold.getModels()[0]?.id, "gemini-4-argon");
  assert.equal(hits, 3);

  // Within TTL + malformed plain api_key in background do not hit network or throw
  await refresh("proj-1");
  await refresh("proj-1", { credential: { type: "api_key", key: "plain-non-json" } });
  assert.equal(hits, 3);

  fail = true;
  await refresh("proj-2");
  assert.equal(catalog.getModels()[0]?.id, "gemini-4-argon");
  await assert.rejects(() => refresh("proj-2", { force: true }));
});
