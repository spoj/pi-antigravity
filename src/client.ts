import { createHash } from "node:crypto";
import type { FetchFunction, ProviderHeaders } from "@earendil-works/pi-ai";

export const PROVIDER_ID = "antigravity";
export const PROVIDER_NAME = "Google Antigravity";
export const ANTIGRAVITY_API = "antigravity-api" as const;
export type AntigravityApi = typeof ANTIGRAVITY_API;

export const DEFAULT_ENDPOINT = "https://daily-cloudcode-pa.googleapis.com";
export const ENDPOINT_FALLBACKS = [
  DEFAULT_ENDPOINT,
  "https://daily-cloudcode-pa.sandbox.googleapis.com",
  "https://cloudcode-pa.googleapis.com",
] as const;

export const USER_AGENT =
  "antigravity/cli/1.5.5 (aidev_client; os_type=linux; arch=amd64; cl=982146307; auth_method=consumer)";

const DISCOVERY_TIMEOUT_MS = 8_000;

export type ModelInfoRaw = {
  displayName?: unknown;
  label?: unknown;
  modelName?: unknown;
  maxTokens?: unknown;
  maxOutputTokens?: unknown;
  supportsThinking?: unknown;
  supportsImages?: unknown;
  isInternal?: unknown;
  disabled?: unknown;
  apiProvider?: unknown;
  model?: unknown;
};

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function stableUuid(seed: string): string {
  const bytes = createHash("sha1").update(seed).digest().subarray(0, 16);
  bytes[6] = (bytes[6] & 0x0f) | 0x50;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const h = bytes.toString("hex");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

export const defaultProjectId = (seed = "antigravity-default"): string => stableUuid(`antigravity:${seed}`);
export const encodeApiKey = (token: string, projectId: string): string => JSON.stringify({ token, projectId });

export function parseApiKey(raw: string | undefined): { token: string; projectId: string } {
  if (!raw) throw new Error("No Antigravity credentials. Run /login antigravity.");
  const ok = (v: unknown): v is string => typeof v === "string" && /^[\x21-\x7e]+$/.test(v);
  try {
    const p: unknown = JSON.parse(raw);
    if (!isRecord(p) || !ok(p.token) || !ok(p.projectId)) throw new Error("invalid");
    return { token: p.token, projectId: p.projectId };
  } catch {
    throw new Error("Invalid Antigravity credentials. Run /login antigravity.");
  }
}

export function redactSecrets(text: string): string {
  return text
    .replace(/\b(?:Bearer|Basic)\s+[\x21-\x7e]+/gi, "Bearer [REDACTED]")
    .replace(/\bya29\.[A-Za-z0-9._~+/-]+=*/g, "[REDACTED]")
    .replace(/\b1\/\/[A-Za-z0-9._~+/-]+=*/g, "[REDACTED]")
    .replace(/\b4\/0A[A-Za-z0-9._~+/-]+/g, "[REDACTED]")
    .replace(/\bGOCSPX-[A-Za-z0-9._-]+/g, "[REDACTED]")
    .replace(/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g, "[REDACTED]")
    .replace(
      /(["']?(?:access_token|refresh_token|client_secret|id_token|code_verifier|api_key|x-goog-api-key|authorization)["']?\s*[:=]\s*)(?:"(?:\\.|[^"])*"|'[^']*'|[^&\s"',}]+)/gi,
      "$1[REDACTED]",
    )
    .replace(/([?&#]|^)((?:access_token|refresh_token|client_secret|id_token|code|code_verifier)=)[^&\s#"']+/gi, "$1$2[REDACTED]");
}

export function buildHeaders(
  token: string,
  ...overrides: Array<ProviderHeaders | Record<string, string> | undefined>
): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    "User-Agent": USER_AGENT,
  };
  for (const group of overrides) {
    if (!group) continue;
    for (const [name, value] of Object.entries(group)) {
      const lower = name.toLowerCase();
      if (lower === "authorization") continue;
      const prev = Object.keys(headers).find((k) => k.toLowerCase() === lower);
      if (prev) delete headers[prev];
      if (value !== null) headers[name] = value;
    }
  }
  return headers;
}

export function endpointsFor(baseUrl?: string): readonly string[] {
  const trimmed = baseUrl?.trim().replace(/\/+$/, "");
  return trimmed && trimmed !== DEFAULT_ENDPOINT ? [trimmed] : ENDPOINT_FALLBACKS;
}

export function withTimeout(signal: AbortSignal | undefined, ms: number): AbortSignal | undefined {
  if (ms <= 0) return signal;
  const timeout = AbortSignal.timeout(ms);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

function extractProjectId(data: unknown): string | undefined {
  if (!isRecord(data)) return undefined;
  const keys = [
    "antigravityProjectId", "projectId", "backendProjectId",
    "userDefinedCloudaicompanionProject", "cloudaicompanionProject", "project", "id",
  ];
  for (const key of keys) {
    const v = data[key];
    if (typeof v === "string" && v) return v;
    if (isRecord(v) && typeof v.id === "string" && v.id) return v.id;
  }
  for (const key of ["projects", "projectIds", "cloudaicompanionProjects"]) {
    const arr = data[key];
    if (!Array.isArray(arr)) continue;
    for (const item of arr) {
      const found = typeof item === "string" && item ? item : extractProjectId(item);
      if (found) return found;
    }
  }
  return undefined;
}

export async function loadCodeAssist(
  token: string,
  options?: { signal?: AbortSignal; fetch?: FetchFunction; baseUrl?: string },
): Promise<string | undefined> {
  const fetchFn = options?.fetch ?? fetch;
  for (const endpoint of endpointsFor(options?.baseUrl)) {
    if (options?.signal?.aborted) throw new Error("Request was aborted");
    try {
      const res = await fetchFn(`${endpoint}/v1internal:loadCodeAssist`, {
        method: "POST",
        headers: buildHeaders(token),
        body: JSON.stringify({ metadata: { ideType: "ANTIGRAVITY" } }),
        signal: withTimeout(options?.signal, DISCOVERY_TIMEOUT_MS),
      });
      if (!res.ok) {
        await res.body?.cancel().catch(() => {});
        continue;
      }
      const direct = extractProjectId(await res.json());
      if (direct) return direct;

      const listRes = await fetchFn(`${endpoint}/v1internal:listCloudAICompanionProjects`, {
        method: "POST",
        headers: buildHeaders(token),
        body: "{}",
        signal: withTimeout(options?.signal, DISCOVERY_TIMEOUT_MS),
      });
      if (listRes.ok) {
        const listed = extractProjectId(await listRes.json());
        if (listed) return listed;
      } else {
        await listRes.body?.cancel().catch(() => {});
      }
    } catch (err) {
      if (options?.signal?.aborted) throw err;
    }
  }
  return undefined;
}

export async function fetchAvailableModelsCatalog(
  token: string,
  projectId: string,
  options?: { signal?: AbortSignal; fetch?: FetchFunction; baseUrl?: string },
): Promise<{ models: Record<string, ModelInfoRaw>; recommendedIds?: Set<string> }> {
  const fetchFn = options?.fetch ?? fetch;
  const payloads = await Promise.all(
    endpointsFor(options?.baseUrl).map(async (endpoint) => {
      try {
        const res = await fetchFn(`${endpoint}/v1internal:fetchAvailableModels`, {
          method: "POST",
          headers: buildHeaders(token),
          body: JSON.stringify({ project: projectId }),
          signal: withTimeout(options?.signal, DISCOVERY_TIMEOUT_MS),
        });
        if (!res.ok) {
          await res.body?.cancel().catch(() => {});
          return undefined;
        }
        return (await res.json()) as unknown;
      } catch {
        return undefined;
      }
    }),
  );

  const models: Record<string, ModelInfoRaw> = {};
  const recommendedIds = new Set<string>();
  let matched = false;

  for (const raw of payloads) {
    if (!isRecord(raw) || !isRecord(raw.models)) continue;
    const entries = Object.entries(raw.models).filter(([, v]) => isRecord(v)) as Array<[string, ModelInfoRaw]>;
    if (entries.length === 0) continue;
    matched = true;
    for (const [id, info] of entries) models[id] = info;
    if (!Array.isArray(raw.agentModelSorts)) continue;
    for (const sort of raw.agentModelSorts) {
      if (!isRecord(sort) || !Array.isArray(sort.groups)) continue;
      for (const group of sort.groups) {
        if (!isRecord(group) || !Array.isArray(group.modelIds)) continue;
        for (const id of group.modelIds) {
          if (typeof id === "string" && id) recommendedIds.add(id);
        }
      }
    }
  }

  if (!matched) throw new Error("Antigravity fetchAvailableModels failed across all endpoints");
  return { models, ...(recommendedIds.size > 0 ? { recommendedIds } : {}) };
}
