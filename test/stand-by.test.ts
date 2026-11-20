import assert from "node:assert/strict";
import { describe, it } from "node:test";
import antigravityExtension from "../src/index.ts";

type Handler = (event: unknown, ctx: unknown) => void;

function createMockPi(initialTools = ["read", "bash"]) {
  let activeTools = [...initialTools];
  const handlers = new Map<string, Handler[]>();
  const tools: Array<Record<string, unknown>> = [];
  const pi = {
    registerProvider() {},
    registerTool(tool: Record<string, unknown>) {
      tools.push(tool);
    },
    getActiveTools: () => [...activeTools],
    setActiveTools(next: string[]) {
      activeTools = [...next];
    },
    on(event: string, handler: Handler) {
      handlers.set(event, [...(handlers.get(event) ?? []), handler]);
    },
  };
  antigravityExtension(pi as never);
  const emit = (event: string, payload: unknown, ctx: unknown) => {
    for (const h of handlers.get(event) ?? []) h(payload, ctx);
  };
  return { tools, emit, getActiveTools: () => [...activeTools] };
}

describe("gemini-4-neon stand_by tool", () => {
  it("activates stand_by for any provider's gemini-4-neon model and deactivates for other models", async () => {
    const mock = createMockPi();
    assert.equal(mock.tools.length, 1);
    const tool = mock.tools[0]!;
    assert.equal(tool.name, "stand_by");
    assert.equal(tool.exposure, "model-only");
    assert.equal(tool.defaultActive, false);

    const execResult = await (tool.execute as () => Promise<{ terminate?: boolean }>)();
    assert.equal(execResult.terminate, true);

    mock.emit("session_start", {}, { model: { provider: "google", id: "gemini-4-neon" }, mode: "tui" });
    assert.deepEqual(mock.getActiveTools(), ["read", "bash", "stand_by"]);

    mock.emit("model_select", { model: { provider: "openrouter", id: "google/gemini-4-neon-preview" } }, { mode: "rpc" });
    assert.deepEqual(mock.getActiveTools(), ["read", "bash", "stand_by"]);

    mock.emit("model_select", { model: { provider: "antigravity", id: "gemini-4-carbon" } }, { mode: "tui" });
    assert.deepEqual(mock.getActiveTools(), ["read", "bash"]);

    mock.emit("model_select", { model: { provider: "antigravity", id: "gemini-4-neon" } }, { mode: "tui" });
    assert.deepEqual(mock.getActiveTools(), ["read", "bash", "stand_by"]);
  });

  it("does not activate stand_by in print or json modes", () => {
    const mock = createMockPi();
    mock.emit("session_start", {}, { model: { provider: "antigravity", id: "gemini-4-neon" }, mode: "print" });
    assert.deepEqual(mock.getActiveTools(), ["read", "bash"]);
    mock.emit("model_select", { model: { provider: "google", id: "gemini-4-neon" } }, { mode: "json" });
    assert.deepEqual(mock.getActiveTools(), ["read", "bash"]);
  });
});
