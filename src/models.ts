import {
  clampThinkingLevel,
  type Api,
  type FetchFunction,
  type Model,
  type ModelCost,
  type ModelsStoreEntry,
  type ModelThinkingLevel,
  type RefreshModelsContext,
  type ThinkingLevel,
  type ThinkingLevelMap,
} from "@earendil-works/pi-ai";
import { credentialApiKey } from "./auth.ts";
import {
  ANTIGRAVITY_API,
  DEFAULT_ENDPOINT,
  PROVIDER_ID,
  fetchAvailableModelsCatalog,
  isRecord,
  parseApiKey,
  type AntigravityApi,
  type ModelInfoRaw,
} from "./client.ts";

export const REFRESH_TTL_MS = 4 * 60 * 60 * 1000;

const ZERO_COST: ModelCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const PREFIX_COSTS: ReadonlyArray<readonly [string, ModelCost]> = [
  ["gemini-4-hydrogen", { input: 1.5, output: 7.5, cacheRead: 0.075, cacheWrite: 1.5 }],
  ["gemini-4-", { input: 2, output: 10, cacheRead: 0.1, cacheWrite: 2 }],
  ["gemini-3.8-flash", { input: 0.75, output: 3.75, cacheRead: 0.075, cacheWrite: 0.75 }],
  ["claude-opus-", { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 }],
  ["claude-sonnet-", { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 }],
  ["gpt-oss-", { input: 0.6, output: 2.4, cacheRead: 0.15, cacheWrite: 0.6 }],
];

const FALLBACK_ENUMS: Readonly<Record<string, string>> = {
  "gemini-4-hydrogen-low": "MODEL_PLACEHOLDER_M527",
  "gemini-4-hydrogen-medium": "MODEL_PLACEHOLDER_M526",
  "gemini-4-hydrogen-high": "MODEL_PLACEHOLDER_M525",
  "gemini-4-hydrogen-xhigh": "MODEL_PLACEHOLDER_M528",
  "gemini-4-carbon-low": "MODEL_PLACEHOLDER_M438",
  "gemini-4-carbon-medium": "MODEL_PLACEHOLDER_M437",
  "gemini-4-carbon-high": "MODEL_PLACEHOLDER_M427",
  "gemini-4-argon-low": "MODEL_PLACEHOLDER_M419",
  "gemini-4-argon-medium": "MODEL_PLACEHOLDER_M418",
  "gemini-4-argon-high": "MODEL_PLACEHOLDER_M417",
  "gemini-3.8-flash-low": "MODEL_PLACEHOLDER_M320",
  "gemini-3.8-flash-medium": "MODEL_PLACEHOLDER_M319",
  "gemini-3.8-flash-high": "MODEL_PLACEHOLDER_M318",
  "claude-opus-5-5-low": "MODEL_PLACEHOLDER_M400",
  "claude-opus-5-5-medium": "MODEL_PLACEHOLDER_M401",
  "claude-opus-5-5-high": "MODEL_PLACEHOLDER_M402",
  "claude-sonnet-5-5-low": "MODEL_PLACEHOLDER_M403",
  "claude-sonnet-5-5-medium": "MODEL_PLACEHOLDER_M404",
  "claude-sonnet-5-5-high": "MODEL_PLACEHOLDER_M405",
  "gpt-oss-120b-medium": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
};

const UI_LEVELS: readonly ModelThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
const LEVEL_MAP: Readonly<Record<string, ThinkingLevel>> = {
  minimal: "minimal",
  low: "low",
  "extra-low": "low",
  medium: "medium",
  high: "high",
  thinking: "high",
  xhigh: "xhigh",
  "x-high": "xhigh",
  "extra-high": "xhigh",
};

function modelCostFor(publicId: string): ModelCost {
  return PREFIX_COSTS.find(([prefix]) => publicId.startsWith(prefix))?.[1] ?? ZERO_COST;
}

function buildThinkingLevelMap(
  variants: Partial<Record<ThinkingLevel, string>>,
  unsuffixed?: string,
  offRuntime?: string,
): ThinkingLevelMap {
  const hasVariants = Object.keys(variants).length > 0;
  return Object.fromEntries(
    UI_LEVELS.map((l) => [
      l,
      l === "off"
        ? (offRuntime ?? null)
        : l === "max"
          ? null
          : (variants[l] ?? (!hasVariants && l === "high" ? unsuffixed : undefined) ?? null),
    ]),
  ) as ThinkingLevelMap;
}

function seedModel(
  id: string,
  name: string,
  contextWindow: number,
  maxTokens: number,
  levels: readonly ThinkingLevel[],
  images = true,
): Model<AntigravityApi> {
  const variants = Object.fromEntries(levels.map((l) => [l, `${id}-${l}`])) as Partial<Record<ThinkingLevel, string>>;
  return {
    id,
    name: `${name} (Antigravity)`,
    api: ANTIGRAVITY_API,
    provider: PROVIDER_ID,
    baseUrl: DEFAULT_ENDPOINT,
    reasoning: true,
    thinkingLevelMap: buildThinkingLevelMap(variants),
    input: images ? ["text", "image"] : ["text"],
    cost: modelCostFor(id),
    contextWindow,
    maxTokens,
  };
}

export const FALLBACK_MODELS: readonly Model<AntigravityApi>[] = [
  seedModel("gemini-4-hydrogen", "Gemini 4 Hydrogen", 1_048_576, 262_144, ["low", "medium", "high", "xhigh"]),
  seedModel("gemini-4-carbon", "Gemini 4 Carbon", 1_048_576, 262_144, ["low", "medium", "high"]),
  seedModel("gemini-4-argon", "Gemini 4 Argon", 1_048_576, 65_536, ["low", "medium", "high"]),
  seedModel("gemini-3.8-flash", "Gemini 3.8 Flash", 1_048_576, 65_536, ["low", "medium", "high"]),
  seedModel("claude-opus-5-5", "Claude Opus 5.5", 1_000_000, 128_000, ["low", "medium", "high"]),
  seedModel("claude-sonnet-5-5", "Claude Sonnet 5.5", 1_000_000, 128_000, ["low", "medium", "high"]),
  seedModel("gpt-oss-120b", "GPT-OSS 120B", 131_072, 32_768, ["medium"], false),
];

function parseTrailingTier(displayName: string | undefined): { stripped: string; level?: ThinkingLevel } {
  if (!displayName) return { stripped: "" };
  const trimmed = displayName.trim();
  if (!trimmed.endsWith(")")) return { stripped: trimmed };
  const open = trimmed.lastIndexOf("(");
  if (open < 0 || trimmed.length - open > 32) return { stripped: trimmed };
  const key = trimmed.slice(open + 1, -1).trim().toLowerCase().replace(/[\s-]+/g, "-");
  const level = LEVEL_MAP[key];
  return level ? { stripped: trimmed.slice(0, open).trimEnd(), level } : { stripped: trimmed };
}

const minPos = (cur: number | undefined, v: unknown): number | undefined =>
  typeof v === "number" && Number.isSafeInteger(v) && v > 0 ? Math.min(cur ?? v, v) : cur;

export function buildCatalog(
  rawModels: Record<string, ModelInfoRaw>,
  recommendedIds?: ReadonlySet<string>,
  baseUrl = DEFAULT_ENDPOINT,
): { models: Model<AntigravityApi>[]; enums: Record<string, string> } {
  const enums: Record<string, string> = {};
  for (const [id, info] of Object.entries(rawModels)) {
    if (typeof info?.model === "string" && info.model) enums[id] = info.model;
  }

  type Group = {
    publicId: string;
    variants: Partial<Record<ThinkingLevel, string>>;
    unsuffixed?: string;
    nonThinkingRuntime?: string;
    tieredRuntime?: string;
    displayNames: string[];
    supportsThinking?: boolean;
    supportsImages?: boolean;
    contextWindow?: number;
    maxOutputTokens?: number;
  };

  const collect = (allowSet?: ReadonlySet<string>): Map<string, Group> => {
    const groups = new Map<string, Group>();
    for (const [runtimeId, info] of Object.entries(rawModels)) {
      const displayName = [info?.displayName, info?.label, info?.modelName]
        .find((v): v is string => typeof v === "string" && Boolean(v.trim()))
        ?.trim();
      if (
        !/^[a-z0-9][a-z0-9_.-]*$/i.test(runtimeId) ||
        /^MODEL_|^(chat_|tab_)/i.test(runtimeId) ||
        /image/i.test(runtimeId) ||
        /image/i.test(displayName ?? "") ||
        info?.isInternal === true ||
        info?.apiProvider === "API_PROVIDER_INTERNAL" ||
        info?.disabled === true ||
        runtimeId.endsWith("-disabled")
      ) {
        continue;
      }
      if (allowSet && !allowSet.has(runtimeId) && !/(?:^|-)(?:hydrogen|oxygen)(?:-|$)/i.test(runtimeId)) {
        continue;
      }

      const isTiered = runtimeId.endsWith("-tiered");
      const suffix = isTiered
        ? null
        : runtimeId.match(/^(.+?)-(extra-low|extra-high|x-high|xhigh|thinking|minimal|medium|high|low)$/i);
      const publicId = isTiered ? runtimeId.slice(0, -"-tiered".length) : (suffix?.[1] ?? runtimeId);

      const g = groups.get(publicId) ?? { publicId, variants: {}, displayNames: [] };
      groups.set(publicId, g);
      if (displayName) g.displayNames.push(displayName);
      g.contextWindow = minPos(g.contextWindow, info?.maxTokens);
      g.maxOutputTokens = minPos(g.maxOutputTokens, info?.maxOutputTokens);
      if (info?.supportsThinking === true) g.supportsThinking = true;
      else if (info?.supportsThinking === false && g.supportsThinking !== true) g.supportsThinking = false;
      if (info?.supportsImages === false) g.supportsImages = false;
      else if (info?.supportsImages === true && g.supportsImages === undefined) g.supportsImages = true;

      if (isTiered) {
        g.tieredRuntime ??= runtimeId;
        g.unsuffixed ??= runtimeId;
      } else {
        const suffixLevel = suffix ? LEVEL_MAP[suffix[2]!.toLowerCase()] : undefined;
        const level = parseTrailingTier(displayName).level ?? suffixLevel;
        if (level) g.variants[level] = runtimeId;
        else {
          g.unsuffixed = runtimeId;
          if (info?.supportsThinking === false) g.nonThinkingRuntime = runtimeId;
        }
      }
    }
    return groups;
  };

  let groups = collect(recommendedIds?.size ? recommendedIds : undefined);
  if (groups.size === 0 && recommendedIds?.size) groups = collect();
  if (groups.size === 0) throw new Error("Antigravity catalog contains no supported chat models");

  const models: Model<AntigravityApi>[] = [...groups.values()].map((g) => {
    const seed = FALLBACK_MODELS.find((m) => m.id === g.publicId);
    if (g.tieredRuntime && Object.keys(g.variants).length === 0) {
      g.variants = { low: g.tieredRuntime, medium: g.tieredRuntime, high: g.tieredRuntime };
    }
    const hasVariants = Object.keys(g.variants).length > 0;
    const reasoning =
      g.supportsThinking === false ? false : hasVariants || g.supportsThinking === true || Boolean(seed?.reasoning);
    const supportsImages = g.supportsImages ?? seed?.input.includes("image") ?? true;
    const rawName = parseTrailingTier(g.displayNames[0]).stripped;
    const name = rawName
      ? /\(antigravity\)$/i.test(rawName)
        ? rawName
        : `${rawName} (Antigravity)`
      : (seed?.name ?? `${g.publicId} (Antigravity)`);
    const offRuntime = hasVariants ? g.nonThinkingRuntime : undefined;
    const needsRouteMap =
      reasoning || hasVariants || Boolean(offRuntime) || (g.unsuffixed !== undefined && g.unsuffixed !== g.publicId);

    return {
      id: g.publicId,
      name,
      api: ANTIGRAVITY_API,
      provider: PROVIDER_ID,
      baseUrl,
      reasoning,
      ...(needsRouteMap
        ? { thinkingLevelMap: buildThinkingLevelMap(g.variants, g.unsuffixed ?? g.publicId, offRuntime) }
        : {}),
      input: supportsImages ? ["text", "image"] : ["text"],
      cost: modelCostFor(g.publicId),
      contextWindow: g.contextWindow ?? seed?.contextWindow ?? 128_000,
      maxTokens: g.maxOutputTokens ?? seed?.maxTokens ?? 8_192,
    };
  });

  models.sort((a, b) => a.id.localeCompare(b.id));
  return { models, enums };
}

export function resolveRuntimeModel(model: Model<Api>, reasoning?: ModelThinkingLevel): string {
  const map = model.thinkingLevelMap;
  if (!map) return model.id;
  if ((!model.reasoning || !reasoning || reasoning === "off") && typeof map.off === "string" && map.off) {
    return map.off;
  }
  if (model.reasoning) {
    const mapped = map[clampThinkingLevel(model, reasoning ?? "off")];
    if (typeof mapped === "string" && mapped) return mapped;
  }
  for (const level of ["low", "minimal", "medium", "high", "xhigh"] as const) {
    const candidate = map[level];
    if (typeof candidate === "string" && candidate) return candidate;
  }
  return model.id;
}

const SIBLING_ORDER: readonly ModelThinkingLevel[] = ["high", "medium", "xhigh", "low", "minimal", "off"];

export function siblingRuntimeModels(model: Model<Api>, currentRuntime: string): string[] {
  const map = model.thinkingLevelMap;
  if (!map) return [];
  const out: string[] = [];
  for (const level of SIBLING_ORDER) {
    const runtime = map[level];
    if (typeof runtime === "string" && runtime && runtime !== currentRuntime && !out.includes(runtime)) {
      out.push(runtime);
    }
  }
  return out;
}

export function getThinkingConfig(
  model: Model<Api>,
  runtimeModel: string,
  reasoning?: ModelThinkingLevel,
): { includeThoughts: boolean; thinkingBudget: number } | undefined {
  const isGemini2 = /^gemini-2(?:[.-]|$)/.test(runtimeModel);
  if (!model.reasoning || model.thinkingLevelMap?.off === runtimeModel) {
    return isGemini2 ? { includeThoughts: false, thinkingBudget: 0 } : undefined;
  }
  const clamped = reasoning && reasoning !== "off" ? clampThinkingLevel(model, reasoning) : undefined;
  const directMatch = clamped && model.thinkingLevelMap?.[clamped] === runtimeModel ? clamped : undefined;
  const mappedLevel =
    directMatch ??
    (["xhigh", "high", "medium", "low", "minimal"] as const).find((l) => model.thinkingLevelMap?.[l] === runtimeModel);

  if (!reasoning || reasoning === "off") {
    const requestedRuntime = resolveRuntimeModel(model, "off");
    if (!mappedLevel || runtimeModel === requestedRuntime) {
      if (isGemini2) return { includeThoughts: false, thinkingBudget: 0 };
      if (runtimeModel.startsWith("gemini-4")) return undefined;
    }
  }
  if (runtimeModel.startsWith("claude-")) return { includeThoughts: true, thinkingBudget: 1024 };
  if (runtimeModel.startsWith("gpt-oss-")) return { includeThoughts: true, thinkingBudget: 8192 };
  if (runtimeModel.startsWith("gemini-")) {
    const suffixLevel = runtimeModel.match(/-(extra-low|extra-high|x-high|xhigh|high|medium|low|minimal)$/i)?.[1]?.toLowerCase();
    const normSuffix = suffixLevel ? LEVEL_MAP[suffixLevel] : undefined;
    const effective = directMatch ?? normSuffix ?? mappedLevel ?? clamped ?? "low";
    const budget = effective === "high" || effective === "xhigh" || effective === "max" ? -1 : effective === "medium" ? 4000 : 1000;
    return { includeThoughts: true, thinkingBudget: budget };
  }
  return undefined;
}

type StoredAntigravityEntry = ModelsStoreEntry & {
  enums?: Record<string, string>;
  projectId?: string;
  baseUrl?: string;
};

export function createCatalogManager(fetchFn?: FetchFunction) {
  let models: readonly Model<AntigravityApi>[] = FALLBACK_MODELS;
  let enums: Record<string, string> = {};

  return {
    getModels: (): readonly Model<AntigravityApi>[] => models,
    getModelEnum: (runtimeModel: string): string | undefined => enums[runtimeModel] ?? FALLBACK_ENUMS[runtimeModel],

    async refreshModels(context: RefreshModelsContext): Promise<void> {
      const baseUrl = (context.baseUrl || DEFAULT_ENDPOINT).trim().replace(/\/+$/, "");
      const stored = context.stored as StoredAntigravityEntry | undefined;
      const hasEnums = isRecord(stored?.enums);

      if (hasEnums && stored && Array.isArray(stored.models) && stored.models.length > 0) {
        const hydrated: Model<AntigravityApi>[] = stored.models
          .filter((m): m is Model<Api> => m.type === undefined || m.type === "chat")
          .map(({ compat: _, ...m }) => ({ ...m, api: ANTIGRAVITY_API, provider: PROVIDER_ID, baseUrl }));
        if (hydrated.length > 0) {
          const nextEnums = stored.enums as Record<string, string>;
          await context.publish({
            update: () => {
              models = hydrated;
              enums = nextEnums;
            },
          });
        }
      }

      if (!context.allowNetwork || context.signal.aborted || !context.credential) return;
      const apiKey = context.credential.type === "oauth" ? credentialApiKey(context.credential) : context.credential.key;
      if (!apiKey) return;

      const creds = parseApiKey(apiKey);
      const checkedAt = typeof stored?.checkedAt === "number" ? stored.checkedAt : 0;
      const storedBase = (stored?.baseUrl || DEFAULT_ENDPOINT).trim().replace(/\/+$/, "");
      const now = Date.now();
      if (
        !context.force &&
        hasEnums &&
        stored?.projectId === creds.projectId &&
        storedBase === baseUrl &&
        checkedAt > 0 &&
        now >= checkedAt &&
        now - checkedAt < REFRESH_TTL_MS
      ) {
        return;
      }

      const raw = await fetchAvailableModelsCatalog(creds.token, creds.projectId, {
        signal: context.signal,
        fetch: fetchFn,
        baseUrl,
      });
      if (context.signal.aborted) return;
      const built = buildCatalog(raw.models, raw.recommendedIds, baseUrl);
      await context.publish({
        persist: {
          models: built.models,
          checkedAt: Date.now(),
          enums: built.enums,
          projectId: creds.projectId,
          baseUrl,
        } as StoredAntigravityEntry,
        update: () => {
          models = built.models;
          enums = built.enums;
        },
      });
    },
  };
}
