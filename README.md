# pi-antigravity

Google Antigravity / Cloud Code Assist OAuth provider for [Pi](https://github.com/earendil-works/pi).

Zero runtime dependencies, zero custom settings, zero extra tools or slash commands — everything integrates natively with Pi's provider, OAuth, and model catalog APIs.

## Install

```bash
pi install git:github.com/spoj/pi-antigravity
```

Requires Node >= 22.19.0 and Pi >= 1.5.2.

## Usage

1. Sign in with Google OAuth (PKCE):
   ```text
   /login antigravity
   ```
   On a remote/SSH machine where `localhost:51121` isn't reachable from your browser, paste the final redirect URL into Pi's prompt.
2. Pick a model with `/model` (for example `antigravity/gemini-4-hydrogen`, `antigravity/gemini-4-carbon`, `antigravity/gemini-4-argon`, `antigravity/gemini-3.8-flash`, `antigravity/claude-opus-5-5`, `antigravity/claude-sonnet-5-5`, or `antigravity/gpt-oss-120b`) and thinking level with `/thinking`.
3. Interactive sessions refresh the model catalog automatically on startup (cached in `~/.pi/agent/models-store.json`). Sign out anytime with `/logout antigravity`.

## Development

```bash
npm ci
npm run check
```
