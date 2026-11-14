import type { FetchFunction, Provider } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createAntigravityOAuth } from "./auth.ts";
import { DEFAULT_ENDPOINT, PROVIDER_ID, PROVIDER_NAME, type AntigravityApi } from "./client.ts";
import { createCatalogManager } from "./models.ts";
import { createStreamAntigravity } from "./stream.ts";

export function createAntigravityProvider(fetchFn?: FetchFunction): Provider<AntigravityApi> {
  const catalog = createCatalogManager(fetchFn);
  const stream = createStreamAntigravity(catalog.getModelEnum);
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
}
