import { createHash, randomBytes } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { FetchFunction, OAuthAuth, OAuthCredential, ProviderAuthInteraction } from "@earendil-works/pi-ai";
import { PROVIDER_NAME, defaultProjectId, encodeApiKey, isRecord, loadCodeAssist, redactSecrets, withTimeout } from "./client.ts";

export const REDIRECT_URI = "http://localhost:51121/oauth-callback";
const AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL = "https://oauth2.googleapis.com/token";
const USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json";
const CALLBACK_PORT = 51121;
const CALLBACK_TIMEOUT_MS = 5 * 60 * 1000;
const REFRESH_DISCOVERY_BUDGET_MS = 3_000;
const SCOPES = ["aicode", "cloud-platform", "userinfo.email", "userinfo.profile", "cclog", "experimentsandconfigs"]
  .map((s) => `https://www.googleapis.com/auth/${s}`)
  .join(" ");

// Public Google OAuth client used by the official Antigravity desktop CLI.
const CLIENT_ID = atob(
  "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlc" +
    "C5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==",
);
const CLIENT_SECRET = atob("R09DU1BYLUs1OEZXUjQ" + "4NkxkTEoxbUxCOHNYQzR6NnFEQWY=");

export function parseCallbackInput(raw: string, expectedState: string): string {
  const text = raw.trim();
  if (!text) throw new Error("Paste the full redirect URL from your browser's address bar.");
  const url = (() => {
    try {
      return new URL(text);
    } catch {
      return new URL(`${REDIRECT_URI}?${text.replace(/^\?/, "")}`);
    }
  })();
  const state = url.searchParams.get("state");
  if (!state) throw new Error("Callback URL is missing state.");
  if (state !== expectedState) throw new Error("OAuth state mismatch. Start /login antigravity again.");
  const error = url.searchParams.get("error");
  if (error) {
    throw new Error(`OAuth error: ${redactSecrets(url.searchParams.get("error_description") ?? error).slice(0, 200)}`);
  }
  const code = url.searchParams.get("code");
  if (!code) throw new Error("Callback URL is missing code.");
  return code;
}

type CallbackHandle = { wait: Promise<string>; close: () => void };

function isPortConflict(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException | undefined)?.code;
  return code === "EADDRINUSE" || code === "EACCES";
}

function isIpv6Unavailable(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException | undefined)?.code;
  return code === "EAFNOSUPPORT" || code === "EADDRNOTAVAIL";
}

export async function startCallbackServer(expectedState: string, signal: AbortSignal): Promise<CallbackHandle> {
  if (signal.aborted) throw new Error("Login cancelled");
  let resolveCode!: (code: string) => void;
  let rejectCode!: (error: Error) => void;
  const wait = new Promise<string>((res, rej) => {
    resolveCode = res;
    rejectCode = rej;
  });
  wait.catch(() => undefined);

  let settled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const servers: Server[] = [];
  const onAbort = () => settle(() => rejectCode(new Error("Login cancelled")));
  const settle = (fn: () => void) => {
    if (settled) return;
    settled = true;
    if (timer) clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
    fn();
  };

  const sendHtml = (res: ServerResponse, status: number, title: string, detail = "") => {
    const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    res.writeHead(status, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
    res.end(`<!doctype html><meta charset="utf-8"><title>${esc(title)}</title><body style="font-family:system-ui,sans-serif;padding:2rem"><h2>${esc(title)}</h2>${detail ? `<p>${esc(detail)}</p>` : ""}</body>`);
  };

  const handleReq = (req: IncomingMessage, res: ServerResponse) => {
    let url: URL;
    try {
      url = new URL(req.url ?? "/", REDIRECT_URI);
    } catch {
      sendHtml(res, 400, "Invalid callback request.");
      return;
    }
    if (req.method !== "GET" || url.pathname !== "/oauth-callback") {
      sendHtml(res, 404, "Callback route not found.");
      return;
    }
    const state = url.searchParams.get("state");
    if (state !== expectedState) {
      sendHtml(res, 400, `${PROVIDER_NAME} sign-in failed.`, "OAuth state mismatch.");
      return;
    }
    try {
      const code = parseCallbackInput(url.toString(), expectedState);
      sendHtml(res, 200, `Signed in to ${PROVIDER_NAME}.`, "You may close this tab.");
      settle(() => resolveCode(code));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      sendHtml(res, 400, `${PROVIDER_NAME} sign-in failed.`, msg);
      settle(() => rejectCode(new Error(msg)));
    }
  };

  const closeAll = () => {
    for (const s of servers) {
      s.closeAllConnections?.();
      s.close();
    }
  };

  const bindHost = async (host: string, optional = false) => {
    const srv = createServer(handleReq);
    try {
      await new Promise<void>((resolve, reject) => {
        srv.once("error", reject);
        srv.listen({ port: CALLBACK_PORT, host, ipv6Only: host === "::1" }, () => {
          srv.off("error", reject);
          resolve();
        });
      });
      srv.on("error", (err) => settle(() => rejectCode(err)));
      servers.push(srv);
    } catch (err) {
      srv.close();
      if (optional && isIpv6Unavailable(err)) return;
      throw err;
    }
  };

  const explicitHost = process.env.PI_OAUTH_CALLBACK_HOST?.trim();
  try {
    if (explicitHost) {
      await bindHost(explicitHost);
    } else {
      await bindHost("127.0.0.1");
      await bindHost("::1", true);
    }
    if (signal.aborted) throw new Error("Login cancelled");
  } catch (err) {
    closeAll();
    throw err;
  }

  signal.addEventListener("abort", onAbort, { once: true });
  timer = setTimeout(() => settle(() => rejectCode(new Error("Timed out waiting for Google sign-in."))), CALLBACK_TIMEOUT_MS);

  return {
    wait,
    close: () => {
      settled = true;
      if (timer) clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      closeAll();
    },
  };
}

async function acquireAuthCode(
  interaction: ProviderAuthInteraction,
  callback: CallbackHandle | undefined,
  expectedState: string,
): Promise<string> {
  const manualAbort = new AbortController();
  const promptSignal = AbortSignal.any([interaction.signal, manualAbort.signal]);

  const manualPrompt = (async (): Promise<string> => {
    let message = "Complete sign-in in your browser, or paste the redirect URL here:";
    while (true) {
      let input: string;
      try {
        input = await interaction.prompt({
          type: "manual_code",
          message,
          placeholder: REDIRECT_URI,
          signal: promptSignal,
        });
      } catch (err) {
        if (manualAbort.signal.aborted && !interaction.signal.aborted) return new Promise<string>(() => {});
        throw err instanceof Error ? err : new Error(String(err));
      }
      try {
        return parseCallbackInput(input, expectedState);
      } catch (err) {
        if (promptSignal.aborted) throw new Error("Login cancelled", { cause: err });
        message = `Invalid callback (${err instanceof Error ? err.message : String(err)}). Paste the full redirect URL:`;
      }
    }
  })();
  manualPrompt.catch(() => undefined);

  try {
    return callback ? await Promise.race([callback.wait, manualPrompt]) : await manualPrompt;
  } finally {
    manualAbort.abort();
  }
}

export function credentialApiKey(credential: OAuthCredential): string {
  const email = typeof credential.email === "string" ? credential.email : undefined;
  const projectId =
    typeof credential.projectId === "string" && credential.projectId ? credential.projectId : defaultProjectId(email);
  return encodeApiKey(credential.access, projectId);
}

export function createAntigravityOAuth(fetchFn: FetchFunction = fetch): OAuthAuth {
  const postToken = async (params: Record<string, string>, label: string, signal: AbortSignal) => {
    const res = await fetchFn(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ client_id: CLIENT_ID, client_secret: CLIENT_SECRET, ...params }),
      signal,
    });
    if (!res.ok) {
      const raw = await res.text().catch(() => "");
      let detail = "";
      try {
        const p: unknown = JSON.parse(raw);
        if (isRecord(p)) {
          const parts = [p.error, p.error_description].filter((v): v is string => typeof v === "string" && Boolean(v));
          if (parts.length) detail = redactSecrets(parts.join(": ")).slice(0, 200);
        }
      } catch {
        // non-JSON body: omit raw text to prevent leaking echoed parameters
      }
      throw new Error(`Antigravity ${label} failed (${res.status})${detail ? `: ${detail}` : ""}`);
    }
    return (await res.json()) as { access_token?: string; refresh_token?: string; expires_in?: number };
  };

  return {
    name: PROVIDER_NAME,
    isSubscription: true,
    async login(interaction): Promise<OAuthCredential> {
      if (interaction.signal.aborted) throw new Error("Login cancelled");
      const verifier = randomBytes(32).toString("base64url");
      const challenge = createHash("sha256").update(verifier).digest("base64url");
      const state = randomBytes(16).toString("hex");
      const params = new URLSearchParams({
        client_id: CLIENT_ID,
        response_type: "code",
        redirect_uri: REDIRECT_URI,
        scope: SCOPES,
        code_challenge: challenge,
        code_challenge_method: "S256",
        state,
        access_type: "offline",
        prompt: "consent",
      });

      const callback = await startCallbackServer(state, interaction.signal).catch((err) => {
        if (isPortConflict(err)) return undefined;
        throw err;
      });
      if (interaction.signal.aborted) {
        callback?.close();
        throw new Error("Login cancelled");
      }

      interaction.notify({
        type: "auth_url",
        url: `${AUTH_URL}?${params.toString()}`,
        instructions: callback
          ? "Complete sign-in in your browser, or paste the redirect URL if running headless."
          : "Port 51121 is busy; complete sign-in in your browser and paste the redirect URL.",
      });

      try {
        const code = await acquireAuthCode(interaction, callback, state);
        interaction.notify({ type: "progress", message: "Exchanging authorization code…" });
        const tokens = await postToken(
          { code, grant_type: "authorization_code", redirect_uri: REDIRECT_URI, code_verifier: verifier },
          "login",
          interaction.signal,
        );
        if (!tokens.access_token || !tokens.refresh_token || typeof tokens.expires_in !== "number") {
          throw new Error("Google OAuth token response was missing access_token or refresh_token.");
        }

        interaction.notify({ type: "progress", message: "Discovering Antigravity project…" });
        const [email, discovered] = await Promise.all([
          fetchFn(USERINFO_URL, {
            headers: { Authorization: `Bearer ${tokens.access_token}` },
            signal: withTimeout(interaction.signal, 8_000),
          })
            .then(async (r) => (r.ok ? ((await r.json()) as { email?: unknown }).email : undefined))
            .then((e) => (typeof e === "string" && e ? e : undefined))
            .catch(() => undefined),
          loadCodeAssist(tokens.access_token, { signal: interaction.signal, fetch: fetchFn }),
        ]);

        return {
          type: "oauth",
          access: tokens.access_token,
          refresh: tokens.refresh_token,
          expires: Date.now() + tokens.expires_in * 1000 - 5 * 60 * 1000,
          projectId: discovered || defaultProjectId(email),
          ...(email ? { email } : {}),
        };
      } finally {
        callback?.close();
      }
    },

    async refresh(credential, signal): Promise<OAuthCredential> {
      const data = await postToken({ refresh_token: credential.refresh, grant_type: "refresh_token" }, "token refresh", signal);
      if (!data.access_token || typeof data.expires_in !== "number") {
        throw new Error("Antigravity token refresh returned an incomplete response.");
      }

      const email = typeof credential.email === "string" ? credential.email : undefined;
      const storedProj = typeof credential.projectId === "string" && credential.projectId ? credential.projectId : undefined;
      const isRealProj = storedProj && storedProj !== defaultProjectId(email) && storedProj !== defaultProjectId();
      const discovered = isRealProj
        ? storedProj
        : await loadCodeAssist(data.access_token, {
            signal,
            fetch: fetchFn,
            timeoutMs: REFRESH_DISCOVERY_BUDGET_MS,
          }).catch(() => undefined);

      return {
        ...credential,
        type: "oauth",
        access: data.access_token,
        refresh: data.refresh_token || credential.refresh,
        expires: Date.now() + data.expires_in * 1000 - 5 * 60 * 1000,
        projectId: discovered || defaultProjectId(email),
      };
    },

    toAuth: async (credential) => ({ apiKey: credentialApiKey(credential) }),
  };
}
