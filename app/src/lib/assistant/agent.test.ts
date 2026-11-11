import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Message, MessageParam } from "./api";

const responses: Message[] = [];
const requests: MessageParam[][] = [];
vi.mock("./api", () => ({
  sendRequest: vi.fn(async ({ messages }: { messages: MessageParam[] }) => {
    requests.push(structuredClone(messages));
    const next = responses.shift();
    if (!next) throw new Error("no response queued");
    return next;
  }),
}));
vi.mock("./prompt", () => ({ SYSTEM_PROMPT: "system", appState: () => "<app_state>state</app_state>" }));

const { runTurn, runToolCall } = await import("./agent");
const { ToolError } = await import("./tool-types");
import type { AssistantTool } from "./tool-types";

function message(content: Message["content"], stop_reason: Message["stop_reason"]): Message {
  return { id: "m", type: "message", role: "assistant", model: "claude-opus-5-5", content, stop_reason, stop_sequence: null, usage: {} } as unknown as Message;
}

const echo: AssistantTool = {
  kind: "read",
  definition: { name: "echo", description: "", input_schema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } },
  describe: (i) => `Echo ${i.text}`,
  run: async (i) => ({ said: i.text }),
};
const failing: AssistantTool = {
  kind: "read",
  definition: { name: "fail", description: "", input_schema: { type: "object", properties: {} } },
  describe: () => "Failing",
  run: async () => { throw new ToolError("nope"); },
};

const events = () => ({ onText: vi.fn(), onTool: vi.fn(), cancelled: () => false });

beforeEach(() => { responses.length = 0; requests.length = 0; });

describe("runTurn", () => {
  it("runs the tools Claude asks for and sends all results back in one message", async () => {
    responses.push(
      message([
        { type: "text", text: "Let me look.", citations: null },
        { type: "tool_use", id: "t1", name: "echo", input: { text: "a" } },
        { type: "tool_use", id: "t2", name: "fail", input: {} },
      ] as Message["content"], "tool_use"),
      message([{ type: "text", text: "Done.", citations: null }] as Message["content"], "end_turn"),
    );
    const transcript: MessageParam[] = [];
    const ev = events();
    const outcome = await runTurn("hi", { model: "claude-opus-5-5", tools: [echo, failing], transcript }, ev);

    expect(outcome).toEqual({ kind: "answered" });
    expect(transcript.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
    const first = transcript[0].content as { type: string; text: string }[];
    expect(first[0].text).toBe("<app_state>state</app_state>\n\nhi");
    const results = transcript[2].content as { tool_use_id: string; content: string; is_error?: boolean }[];
    expect(results).toEqual([
      { type: "tool_result", tool_use_id: "t1", content: JSON.stringify({ said: "a" }) },
      { type: "tool_result", tool_use_id: "t2", content: "nope", is_error: true },
    ]);
    expect(ev.onText.mock.calls.map((c) => c[0])).toEqual(["Let me look.", "Done."]);
    expect(ev.onTool).toHaveBeenCalledWith(expect.objectContaining({ id: "t1", label: "Echo a", status: "done" }));
    expect(ev.onTool).toHaveBeenCalledWith(expect.objectContaining({ id: "t2", status: "error", detail: "nope" }));
  });

  it("passes thinking and fallback blocks back unchanged", async () => {
    const content = [
      { type: "thinking", thinking: "", signature: "sig" },
      { type: "tool_use", id: "t1", name: "echo", input: { text: "a" } },
    ] as unknown as Message["content"];
    responses.push(message(content, "tool_use"), message([{ type: "text", text: "ok", citations: null }] as Message["content"], "end_turn"));
    const transcript: MessageParam[] = [];
    await runTurn("hi", { model: "m", tools: [echo], transcript }, events());
    expect(requests[1][1].content).toEqual(content);
  });

  it("reports a refusal", async () => {
    responses.push({ ...message([], "refusal"), stop_details: { type: "refusal", category: null, explanation: "Because." } } as unknown as Message);
    expect(await runTurn("hi", { model: "m", tools: [], transcript: [] }, events())).toEqual({ kind: "refused", explanation: "Because." });
  });

  it("answers tool calls it won't run when stopped, so the conversation stays valid", async () => {
    responses.push(message([{ type: "tool_use", id: "t1", name: "echo", input: { text: "a" } }] as Message["content"], "tool_use"));
    let stop = false;
    const transcript: MessageParam[] = [];
    const ev = { onText: vi.fn(), onTool: vi.fn(), cancelled: () => { const s = stop; stop = true; return s; } };
    expect(await runTurn("hi", { model: "m", tools: [echo], transcript }, ev)).toEqual({ kind: "cancelled" });
    expect(transcript[2].content).toEqual([expect.objectContaining({ tool_use_id: "t1", is_error: true })]);
  });
});

describe("runToolCall", () => {
  it("rejects input that doesn't match the tool's schema", async () => {
    const r = await runToolCall([echo], { id: "t", name: "echo", input: { text: 3 } });
    expect(r.ok).toBe(false);
    expect(r.result.content).toBe("Invalid input: input.text must be a string.");
  });
  it("rejects a tool that doesn't exist", async () => {
    const r = await runToolCall([echo], { id: "t", name: "nope", input: {} });
    expect(r.result.content).toBe("There is no tool called nope.");
  });
});
