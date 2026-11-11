/**
 * One assistant turn: send the user's message with a snapshot of the app, run the tools Claude asks
 * for, send their results back, and repeat until Claude answers.
 */

import type Anthropic from "@anthropic-ai/sdk";
import { sendRequest, type Message, type MessageParam } from "./api";
import { SYSTEM_PROMPT, appState } from "./prompt";
import { ToolError, validateInput, type AssistantTool, type ApprovalRequest } from "./tool-types";

/** Stops a turn that keeps calling tools without answering */
export const MAX_STEPS = 16;

export interface TurnEvents {
  /** Text Claude wrote, as each response arrives */
  onText: (text: string) => void;
  /** A tool call started (status "running") or finished */
  onTool: (call: { id: string; name: string; label: string; status: "running" | "done" | "error"; detail?: string }) => void;
  /** Called before each request; return true to stop the turn */
  cancelled: () => boolean;
  /** Asks the user to approve a tool that runs code; resolves to their answer */
  approve?: (request: ApprovalRequest) => Promise<boolean>;
}

export interface TurnOptions {
  model: string;
  tools: AssistantTool[];
  /** The conversation so far, in API form. The turn appends to it. */
  transcript: MessageParam[];
  /** The system prompt; defaults to the assistant's */
  system?: string;
}

export type TurnOutcome =
  | { kind: "answered" }
  | { kind: "cancelled" }
  | { kind: "refused"; explanation: string | null }
  | { kind: "truncated" }
  | { kind: "step_limit" };

type ToolResult = Anthropic.Beta.BetaToolResultBlockParam;

/** Runs one tool call, turning every failure into an error result the model can read */
export async function runToolCall(
  tools: AssistantTool[],
  block: { id: string; name: string; input: unknown },
  approve?: (request: ApprovalRequest) => Promise<boolean>,
): Promise<{ result: ToolResult; ok: boolean; detail?: string }> {
  const tool = tools.find((t) => t.definition.name === block.name);
  const error = (message: string) => ({ result: { type: "tool_result" as const, tool_use_id: block.id, content: message, is_error: true }, ok: false, detail: message });
  if (!tool) return error(`There is no tool called ${block.name}.`);
  const invalid = validateInput(tool.definition.input_schema as Parameters<typeof validateInput>[0], block.input);
  if (invalid) return error(`Invalid input: ${invalid}.`);
  try {
    if (tool.confirm) {
      // Never run code the user hasn't seen and approved
      const request = tool.confirm(block.input as Record<string, unknown>);
      const approved = approve ? await approve(request) : false;
      if (!approved) {
        return { result: { type: "tool_result", tool_use_id: block.id, content: "The user didn't approve this, so it didn't run. Ask what they'd like instead; don't retry it unchanged." }, ok: false, detail: "Not approved" };
      }
    }
    const value = await tool.run(block.input as Record<string, unknown>);
    const content = typeof value === "string" ? value : JSON.stringify(value);
    return { result: { type: "tool_result", tool_use_id: block.id, content }, ok: true };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    // Unexpected failures are reported too, so the model can tell the user rather than retry blindly
    return error(e instanceof ToolError ? message : `The tool failed: ${message}`);
  }
}

function userMessage(text: string): MessageParam {
  return { role: "user", content: [{ type: "text", text: `${appState()}\n\n${text}` }] };
}

/** Results for tool calls that will never run, so the transcript stays valid after a stop */
function abandonedResults(message: Message): MessageParam | null {
  const calls = message.content.filter((b): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use");
  if (!calls.length) return null;
  return { role: "user", content: calls.map((c) => ({ type: "tool_result" as const, tool_use_id: c.id, content: "Stopped by the user before this ran.", is_error: true })) };
}

export async function runTurn(text: string, options: TurnOptions, events: TurnEvents): Promise<TurnOutcome> {
  const { transcript, tools, model } = options;
  transcript.push(userMessage(text));
  const definitions = tools.map((t) => t.definition);

  for (let step = 0; step < MAX_STEPS; step++) {
    if (events.cancelled()) return { kind: "cancelled" };
    const response = await sendRequest({ model, system: options.system ?? SYSTEM_PROMPT, tools: definitions, messages: transcript });
    // The whole content goes back unchanged, thinking and fallback blocks included
    transcript.push({ role: "assistant", content: response.content as Anthropic.Beta.BetaContentBlockParam[] });

    const said = response.content.filter((b): b is Anthropic.Beta.BetaTextBlock => b.type === "text").map((b) => b.text).join("\n\n").trim();
    if (said) events.onText(said);

    if (response.stop_reason === "refusal") {
      return { kind: "refused", explanation: response.stop_details?.explanation ?? null };
    }
    if (response.stop_reason === "max_tokens") {
      const abandoned = abandonedResults(response);
      if (abandoned) transcript.push(abandoned);
      return { kind: "truncated" };
    }
    if (response.stop_reason !== "tool_use") return { kind: "answered" };

    if (events.cancelled()) {
      transcript.push(abandonedResults(response)!);
      return { kind: "cancelled" };
    }
    const calls = response.content.filter((b): b is Anthropic.Beta.BetaToolUseBlock => b.type === "tool_use");
    // All results go back in one message, so Claude keeps making parallel calls
    const results = await Promise.all(calls.map(async (c) => {
      const tool = tools.find((t) => t.definition.name === c.name);
      let label = c.name;
      try { label = tool ? tool.describe(c.input as Record<string, unknown>) : c.name; } catch { /* keep the name */ }
      events.onTool({ id: c.id, name: c.name, label, status: "running" });
      const r = await runToolCall(tools, c, events.approve);
      events.onTool({ id: c.id, name: c.name, label, status: r.ok ? "done" : "error", detail: r.detail });
      return r.result;
    }));
    transcript.push({ role: "user", content: results });
  }
  return { kind: "step_limit" };
}
