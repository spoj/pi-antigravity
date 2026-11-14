import assert from "node:assert/strict";
import test from "node:test";
import { createAntigravityOAuth, parseCallbackInput, REDIRECT_URI } from "../src/auth.ts";
import {
  USER_AGENT,
  buildHeaders,
  defaultProjectId,
  encodeApiKey,
  loadCodeAssist,
  parseApiKey,
  redactSecrets,
} from "../src/client.ts";

test("parseCallbackInput accepts full URL and bare query, rejects state mismatch and errors", () => {
  assert.equal(parseCallbackInput(`${REDIRECT_URI}?code=abc123&state=s1`, "s1"), "abc123");
  assert.equal(parseCallbackInput("?state=s1&code=xyz", "s1"), "xyz");
  assert.throws(() => parseCallbackInput(`${REDIRECT_URI}?code=abc&state=wrong`, "s1"), /state mismatch/i);
  assert.throws(() => parseCallbackInput(`${REDIRECT_URI}?error=access_denied&state=s1`, "s1"), /access_denied/);
});

test("redactSecrets scrubs bearer/basic auth, Google tokens, JWTs, client secrets, and URL fragments", () => {
  const dirty =
    'Authorization: Bearer ya29.sec~1+2/3= {"refresh_token":"1//0g_ref","client_secret":"GOCSPX-priv"} #access_token=frag99 code_verifier=ver123';
  const clean = redactSecrets(dirty);
  for (const secret of ["ya29.sec", "1//0g_ref", "GOCSPX-priv", "frag99", "ver123"]) {
    assert(!clean.includes(secret), `leaked ${secret} in ${clean}`);
  }
});

test("parseApiKey validates printable token and projectId without leaking raw input", () => {
  assert.deepEqual(parseApiKey(encodeApiKey("tok_1", "proj-1")), { token: "tok_1", projectId: "proj-1" });
  assert.throws(() => parseApiKey(""), /No Antigravity credentials/);
  assert.throws(() => parseApiKey("not-json-ya29.secret"), (err) => !String(err).includes("ya29.secret"));
});

test("buildHeaders merges overrides case-insensitively, drops nulls, protects Authorization", () => {
  const h = buildHeaders("tok", { "user-agent": "custom-ua", "X-Extra": "1" }, { "x-extra": null, Authorization: "Bearer bad" });
  assert.equal(h.Authorization, "Bearer tok");
  assert.equal(h["user-agent"], "custom-ua");
  assert.equal(h["User-Agent"], undefined);
  assert.equal(h["X-Extra"], undefined);
  assert.equal(buildHeaders("tok")["User-Agent"], USER_AGENT);
});

test("loadCodeAssist falls back across endpoints and listCloudAICompanionProjects", async () => {
  const calls: string[] = [];
  const fakeFetch: typeof fetch = async (input) => {
    const url = String(input);
    calls.push(url);
    if (url.includes("daily-cloudcode-pa.googleapis.com")) return new Response("down", { status: 503 });
    if (url.endsWith(":loadCodeAssist")) return Response.json({});
    return Response.json({ projects: [{ id: "discovered-proj" }] });
  };
  assert.equal(await loadCodeAssist("tok-fallback", { fetch: fakeFetch }), "discovered-proj");
  assert(calls.length >= 3);
});

test("OAuth login, manual retry, refresh project reuse, and synthetic defaultProjectId recovery", async () => {
  let loadCalls = 0;
  let failDiscovery = false;
  const fakeFetch: typeof fetch = async (input) => {
    const url = String(input);
    if (url.includes("oauth2.googleapis.com/token")) {
      return Response.json({ access_token: "acc_new", refresh_token: "ref_new", expires_in: 3600 });
    }
    if (url.includes("userinfo")) return Response.json({ email: "user@example.com" });
    if (url.includes("loadCodeAssist")) {
      loadCalls++;
      return failDiscovery ? new Response("err", { status: 503 }) : Response.json({ cloudaicompanionProject: "proj-abc" });
    }
    return new Response("unexpected", { status: 404 });
  };

  const oauth = createAntigravityOAuth(fakeFetch);
  let authUrl = "";
  let promptCalls = 0;
  const signal = new AbortController().signal;
  const cred = await oauth.login({
    signal,
    notify(ev) {
      if (ev.type === "auth_url") authUrl = ev.url;
    },
    async prompt() {
      promptCalls++;
      if (promptCalls === 1) return "bad-paste-without-code";
      const state = new URL(authUrl).searchParams.get("state")!;
      return `${REDIRECT_URI}?code=test_code&state=${state}`;
    },
  });

  assert.equal(promptCalls, 2);
  assert.equal(cred.projectId, "proj-abc");
  assert.equal(loadCalls, 1);

  // Real projectId is reused on refresh without calling loadCodeAssist again.
  const refreshed = await oauth.refresh(cred, signal);
  assert.equal(refreshed.projectId, "proj-abc");
  assert.equal(loadCalls, 1);

  // Synthetic fallback projectId retries loadCodeAssist on refresh and heals.
  const fallbackCred = { ...cred, projectId: defaultProjectId("user@example.com") };
  const healed = await oauth.refresh(fallbackCred, signal);
  assert.equal(healed.projectId, "proj-abc");
  assert.equal(loadCalls, 2);

  failDiscovery = true;
  const stillFallback = await oauth.refresh(fallbackCred, signal);
  assert.equal(stillFallback.projectId, defaultProjectId("user@example.com"));
});
