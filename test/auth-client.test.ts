import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { createAntigravityOAuth, parseCallbackInput, REDIRECT_URI, startCallbackServer } from "../src/auth.ts";
import {
  buildHeaders,
  defaultProjectId,
  encodeApiKey,
  fetchAvailableModelsCatalog,
  loadCodeAssist,
  parseApiKey,
  redactSecrets,
  USER_AGENT,
} from "../src/client.ts";

describe("client & auth", () => {
  it("encodes/parses apiKey and rejects malformed keys", () => {
    const encoded = encodeApiKey("tok_123", "proj-abc");
    assert.deepEqual(parseApiKey(encoded), { token: "tok_123", projectId: "proj-abc" });
    assert.throws(() => parseApiKey(undefined), /No Antigravity credentials/);
    assert.throws(() => parseApiKey("not-json"), /Invalid Antigravity credentials/);
    assert.throws(() => parseApiKey(JSON.stringify({ token: "has space", projectId: "ok" })), /Invalid/);
  });

  it("protects Authorization header from caller overrides and redacts secrets in O(N)", () => {
    const h = buildHeaders("real-token", { authorization: "Bearer attacker", "X-Custom": "1" }, { "x-custom": null });
    assert.equal(h.Authorization, "Bearer real-token");
    assert.equal(h["User-Agent"], USER_AGENT);
    assert.equal(h["X-Custom"], undefined);

    const leaked = redactSecrets(
      'Authorization: Bearer ya29.secret_tok {"refresh_token":"1//0g_abc","code_verifier":"unterminated' + "\\".repeat(500),
    );
    assert.doesNotMatch(leaked, /ya29|0g_abc|unterminated/);
  });

  it("parses OAuth callback URLs and validates state before error", () => {
    const code = parseCallbackInput(`${REDIRECT_URI}?code=4/0A_code&state=s123`, "s123");
    assert.equal(code, "4/0A_code");
    assert.throws(() => parseCallbackInput(`${REDIRECT_URI}?error=access_denied&state=wrong`, "s123"), /state mismatch/);
    assert.throws(() => parseCallbackInput(`${REDIRECT_URI}?error=access_denied&state=s123`, "s123"), /access_denied/);
  });

  it("ignores wrong-state error hits on callback server", async () => {
    const ac = new AbortController();
    const prevHost = process.env.PI_OAUTH_CALLBACK_HOST;
    process.env.PI_OAUTH_CALLBACK_HOST = "127.0.0.1";
    let cb: Awaited<ReturnType<typeof startCallbackServer>> | undefined;
    try {
      cb = await startCallbackServer("good_state", ac.signal);
      const stray = await fetch("http://127.0.0.1:51121/oauth-callback?error=access_denied&state=wrong_state");
      assert.equal(stray.status, 400);
      const ok = await fetch("http://127.0.0.1:51121/oauth-callback?code=my_code&state=good_state");
      assert.equal(ok.status, 200);
      assert.equal(await cb.wait, "my_code");
    } finally {
      cb?.close();
      if (prevHost === undefined) delete process.env.PI_OAUTH_CALLBACK_HOST;
      else process.env.PI_OAUTH_CALLBACK_HOST = prevHost;
    }
  });

  it("discovers project via loadCodeAssist fallback and preserves daily model metadata", async () => {
    let calls = 0;
    const id = await loadCodeAssist("tok", {
      fetch: async (url) => {
        calls++;
        const u = String(url);
        if (u.endsWith(":loadCodeAssist")) return new Response("{}", { status: 200 });
        return new Response(JSON.stringify({ projects: [{ id: "discovered-proj" }] }), { status: 200 });
      },
    });
    assert.equal(id, "discovered-proj");
    assert.equal(calls, 2);

    const cat = await fetchAvailableModelsCatalog("tok", "proj", {
      fetch: async (url) => {
        const isDaily = String(url).includes("daily-cloudcode-pa.googleapis.com");
        return new Response(
          JSON.stringify({
            models: {
              "gemini-4-hydrogen-low": {
                displayName: "Gemini 4 Hydrogen (Low)",
                disabled: !isDaily,
                maxOutputTokens: isDaily ? 262144 : 8192,
              },
            },
          }),
          { status: 200 },
        );
      },
    });
    assert.equal(cat.models["gemini-4-hydrogen-low"]?.disabled, false);
    assert.equal(cat.models["gemini-4-hydrogen-low"]?.maxOutputTokens, 262144);
  });

  it("refreshes OAuth token, preserves HTTP status on error, and bounds slow discovery", async () => {
    const errOauth = createAntigravityOAuth(async () => new Response('{"error":"temporarily_unavailable"}', { status: 503 }));
    await assert.rejects(
      () =>
        errOauth.refresh(
          { type: "oauth", access: "old", refresh: "ref_1", expires: 0, projectId: defaultProjectId() },
          new AbortController().signal,
        ),
      /503/,
    );

    const oauth = createAntigravityOAuth(async (url, init) => {
      if (String(url).includes("oauth2.googleapis.com/token")) {
        return new Response(JSON.stringify({ access_token: "new_acc", refresh_token: "rot_ref", expires_in: 3600 }), { status: 200 });
      }
      return new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new Error("timeout")), { once: true });
      });
    });
    const refreshed = await oauth.refresh(
      { type: "oauth", access: "old", refresh: "ref_1", expires: 0, email: "u@example.com", projectId: defaultProjectId("u@example.com") },
      new AbortController().signal,
    );
    assert.equal(refreshed.access, "new_acc");
    assert.equal(refreshed.refresh, "rot_ref");
    assert.equal(refreshed.projectId, defaultProjectId("u@example.com"));
  });
});
