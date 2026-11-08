/**
 * Running a pretrained LLM in the sidecar: installing the packages it needs, downloading and loading
 * the weights, and chatting with it. State lives in the LLM store.
 */

import { sidecarCall, onSidecarNotification } from "./sidecar";
import { useLLMStore, type LoadedModel } from "../store/llm-store";
import type { Precision } from "./llm";

const set = useLLMStore.setState;
const get = useLLMStore.getState;

interface Status { loaded: LoadedModel | null; busy: string | null; missing: string[] }

/** Asks the sidecar what's installed and loaded (after a restart, nothing is) */
export async function refreshLLMStatus() {
  try {
    const s = await sidecarCall<Status>("llm.status");
    set({ missing: s.missing, loaded: s.loaded, generating: s.busy === "generating", ...(s.busy === "loading" ? {} : { loading: null }) });
  } catch { /* sidecar not running: leave as is */ }
}

let stopInstallListener: (() => void) | null = null;

export async function installLLMPackages() {
  set({ install: { status: "running", lines: [] } });
  stopInstallListener?.();
  stopInstallListener = onSidecarNotification("llm.install", (params) => {
    const p = params as { status: string; line?: string; error?: string };
    if (p.status === "running" && p.line) {
      set((s) => ({ install: { ...s.install, lines: [...s.install.lines.slice(-200), p.line!] } }));
    } else if (p.status === "done") {
      set((s) => ({ install: { ...s.install, status: "done" } }));
      stopInstallListener?.();
      void refreshLLMStatus();
    } else if (p.status === "error") {
      set((s) => ({ install: { ...s.install, status: "error", error: p.error } }));
      stopInstallListener?.();
    }
  });
  try {
    await sidecarCall("llm.install");
  } catch (e) {
    stopInstallListener?.();
    set((s) => ({ install: { ...s.install, status: "error", error: e instanceof Error ? e.message : String(e) } }));
  }
}

let stopLoadListener: (() => void) | null = null;

/** Downloads (if needed) and loads the imported model. Resolves when loading has started. */
export async function loadLLM(precision: Precision) {
  const { spec, hfToken } = get();
  if (!spec?.source) return;
  set({ loading: { status: spec.source.kind === "hf" ? "downloading" : "loading" }, loadError: null, loaded: null });
  stopLoadListener?.();
  stopLoadListener = onSidecarNotification("llm.progress", (params) => {
    const p = params as { status: string; bytes?: number; total?: number | null; info?: LoadedModel; error?: string };
    if (p.status === "downloading") set({ loading: { status: "downloading", bytes: p.bytes, total: p.total } });
    else if (p.status === "loading") set({ loading: { status: "loading" } });
    else if (p.status === "loaded") { set({ loading: null, loaded: p.info ?? null }); stopLoadListener?.(); }
    else if (p.status === "error") { set({ loading: null, loadError: p.error ?? "Loading failed" }); stopLoadListener?.(); }
  });
  try {
    await sidecarCall("llm.load", {
      source: spec.source,
      local_path: spec.local_path,
      precision,
      token: hfToken || undefined,
      expected_bytes: spec.stored_bytes,
    });
  } catch (e) {
    stopLoadListener?.();
    set({ loading: null, loadError: e instanceof Error ? e.message : String(e) });
  }
}

export async function cancelLoad() {
  await sidecarCall("llm.stop").catch(() => undefined);
}

export async function unloadLLM() {
  try {
    await sidecarCall("llm.unload");
  } finally {
    set({ loaded: null });
  }
}

let requestId = 0;

/** Sends a message and streams the reply into the chat */
export async function sendChatMessage(text: string) {
  const { chat, systemPrompt, maxNewTokens, temperature, spec, loaded } = get();
  const content = text.trim();
  if (!content || get().generating || !loaded) return;
  const history = [...chat.filter((m) => !m.error), { role: "user" as const, content }];
  const id = ++requestId;
  set({ chat: [...chat, { role: "user", content }, { role: "assistant", content: "", adapter: loaded.adapter }], generating: true });

  const updateReply = (fn: (m: (typeof chat)[number]) => (typeof chat)[number]) =>
    set((s) => ({ chat: s.chat.map((m, i) => (i === s.chat.length - 1 ? fn(m) : m)) }));
  const offToken = onSidecarNotification("llm.token", (params) => {
    const p = params as { id: number; text: string };
    if (p.id === id) updateReply((m) => ({ ...m, content: m.content + p.text }));
  });
  const offDone = onSidecarNotification("llm.done", (params) => {
    const p = params as { id: number; text?: string; tokens?: number; seconds?: number; tokens_per_second?: number | null; stopped?: boolean; error?: string };
    if (p.id !== id) return;
    offToken();
    offDone();
    updateReply((m) => p.error
      ? { ...m, error: p.error }
      : { ...m, content: p.text ?? m.content, stats: { tokens: p.tokens ?? 0, seconds: p.seconds ?? 0, tokensPerSecond: p.tokens_per_second ?? null, stopped: p.stopped } });
    set({ generating: false });
  });

  const messages = [
    ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
    ...history.map(({ role, content }) => ({ role, content })),
  ];
  try {
    await sidecarCall("llm.generate", { id, messages, max_new_tokens: maxNewTokens, temperature, max_context: spec?.max_context });
  } catch (e) {
    offToken();
    offDone();
    updateReply((m) => ({ ...m, error: e instanceof Error ? e.message : String(e) }));
    set({ generating: false });
  }
}

export async function stopGenerating() {
  await sidecarCall("llm.stop").catch(() => undefined);
}

export function clearChat() {
  if (!get().generating) set({ chat: [] });
}
