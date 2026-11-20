import type { FetchFunction, Provider } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createAntigravityOAuth } from "./auth.ts";
import { DEFAULT_ENDPOINT, PROVIDER_ID, PROVIDER_NAME, type AntigravityApi } from "./client.ts";
import { createCatalogManager } from "./models.ts";
import { createStreamAntigravity } from "./stream.ts";

const isNeon = (id?: string) => id?.toLowerCase().includes("gemini-4-neon") ?? false;

export function createAntigravityProvider(fetchFn?: FetchFunction): Provider<AntigravityApi> {
  const catalog = createCatalogManager(fetchFn);
  const stream = createStreamAntigravity(catalog.getModelEnum, fetchFn);
  return {
    id: PROVIDER_ID,
    name: PROVIDER_NAME,
    baseUrl: DEFAULT_ENDPOINT,
    auth: { oauth: createAntigravityOAuth(fetchFn) },
    getModels: catalog.getModels,
    refreshModels: catalog.refreshModels,
    stream,
    streamSimple: stream,
  };
}

export default function antigravityExtension(pi: ExtensionAPI): void {
  pi.registerProvider(createAntigravityProvider());

  pi.registerTool({
    name: "stand_by",
    label: "Stand By",
    description: "End the current turn and wait for the next message or background completion.",
    parameters: { type: "object", properties: {} },
    exposure: "model-only",
    defaultActive: false,
    async execute() {
      return {
        content: [{ type: "text", text: "Standing by." }],
        details: undefined,
        terminate: true,
      };
    },
  });

  const syncStandBy = (modelId: string | undefined, mode: string) => {
    if (mode === "print" || mode === "json") return;
    const active = pi.getActiveTools();
    const has = active.includes("stand_by");
    const want = isNeon(modelId);
    if (want && !has) pi.setActiveTools([...active, "stand_by"]);
    else if (!want && has) pi.setActiveTools(active.filter((t) => t !== "stand_by"));
  };

  pi.on("session_start", (_e, ctx) => syncStandBy(ctx.model?.id, ctx.mode));
  pi.on("model_select", (e, ctx) => syncStandBy(e.model.id, ctx.mode));
}
