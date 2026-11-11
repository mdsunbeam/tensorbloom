/**
 * Talks to Claude through the desktop app's Rust side, which holds the API key in the OS keychain and
 * adds it to each request. The key never enters the webview.
 */

import type Anthropic from "@anthropic-ai/sdk";

export type MessageParam = Anthropic.Beta.BetaMessageParam;
export type Message = Anthropic.Beta.BetaMessage;
export type ContentBlock = Anthropic.Beta.BetaContentBlock;

export interface ModelChoice {
  id: string;
  label: string;
  note: string;
}

export const MODELS: ModelChoice[] = [
  { id: "claude-opus-5-5", label: "Claude Opus 5.5", note: "Most capable. The default." },
  { id: "claude-sonnet-5-5", label: "Claude Sonnet 5.5", note: "Faster and about half the price." },
  { id: "claude-haiku-5-5", label: "Claude Haiku 5.5", note: "Fastest and cheapest, for quick questions." },
];
export const DEFAULT_MODEL = MODELS[0].id;

export interface KeyStatus {
  configured: boolean;
  source: "keychain" | "environment" | null;
}

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!("__TAURI_INTERNALS__" in window)) throw new Error("The assistant needs the TensorBloom desktop app.");
  const { invoke } = await import("@tauri-apps/api/core");
  try {
    return await invoke<T>(cmd, args);
  } catch (e) {
    throw new Error(typeof e === "string" ? e : e instanceof Error ? e.message : String(e));
  }
}

export const keyStatus = () => invoke<KeyStatus>("assistant_key_status").catch((): KeyStatus => ({ configured: false, source: null }));
export const saveKey = (key: string) => invoke<void>("assistant_save_key", { key });
export const deleteKey = () => invoke<void>("assistant_delete_key");

export interface RequestOptions {
  model: string;
  system: string;
  tools: Anthropic.Beta.BetaTool[];
  messages: MessageParam[];
}

/** Haiku has no server-side fallback; the others fall back to another model if they decline */
function supportsFallback(model: string): boolean {
  return !model.startsWith("claude-haiku");
}

/** The request body exactly as the Rust side forwards it; `betas` becomes the anthropic-beta header */
export function buildRequest({ model, system, tools, messages }: RequestOptions) {
  return {
    model,
    max_tokens: 16000,
    // Medium effort keeps chat replies quick; the default for Opus 5.5 is also medium, but say so
    output_config: { effort: "medium" },
    // Cache the system prompt, the tools and the conversation so far
    cache_control: { type: "ephemeral" },
    system,
    tools,
    messages,
    ...(supportsFallback(model) ? { fallbacks: "default", betas: ["server-side-fallback-2026-07-01"] } : {}),
  };
}

export function sendRequest(options: RequestOptions): Promise<Message> {
  return invoke<Message>("assistant_request", { body: buildRequest(options) });
}
