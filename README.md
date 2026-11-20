# pi-antigravity

Google Antigravity / Cloud Code Assist OAuth provider for [Pi](https://github.com/earendil-works/pi).

Zero runtime dependencies, zero custom settings, zero slash commands — everything integrates natively with Pi's provider, OAuth, and model catalog APIs.

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
4. In interactive/RPC sessions (not `print`/`json`), selecting any `*/gemini-4-neon*` model (regardless of provider) activates a `model-only` terminating `yield` tool so async waits settle cleanly without emitting synthetic follow-up turns.

## Troubleshooting

Antigravity can return `RESOURCE_EXHAUSTED` for specific system-prompt wording even when quota remains. The provider rewords known Pi boilerplate while preserving instructions, file paths, and tools. Other capacity failures can still occur.

## Development

```bash
npm ci
npm run check
```
