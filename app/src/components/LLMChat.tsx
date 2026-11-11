import { useEffect, useRef, useState } from "react";
import Button from "./ui/Button";
import Segmented from "./ui/Segmented";
import { useLLMStore } from "../store/llm-store";
import {
  refreshLLMStatus, installLLMPackages, loadLLM, cancelLoad, unloadLLM, sendChatMessage, stopGenerating, clearChat,
} from "../lib/llm-runtime";
import { estimateMemory, fitFor, formatBytes, PRECISION_LABELS, type Precision } from "../lib/llm";

const card = "rounded-lg border border-line p-3";

/** Installs transformers and friends, showing pip's output as it goes */
export function LLMInstallCard({ why }: { why: string }) {
  const missing = useLLMStore((s) => s.missing);
  const install = useLLMStore((s) => s.install);
  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight }); }, [install.lines.length]);
  return (
    <div className={card}>
      <div className="text-xs text-fg">{why} It needs {missing?.join(", ")}, installed into TensorBloom's Python (a few hundred MB).</div>
      <div className="mt-2 flex items-center gap-2">
        <Button size="sm" variant="primary" onClick={() => void installLLMPackages()} disabled={install.status === "running"}>
          {install.status === "running" ? "Installing…" : install.status === "error" ? "Try again" : "Install"}
        </Button>
        {install.status === "done" && <span className="text-[11px] text-success">Installed</span>}
      </div>
      {install.status === "error" && (
        <div role="alert" className="mt-2 px-3 py-2 rounded-lg text-xs bg-danger-subtle text-danger whitespace-pre-wrap">{install.error}</div>
      )}
      {install.lines.length > 0 && install.status === "running" && (
        <pre ref={logRef} className="mt-2 max-h-24 overflow-auto text-[10px] leading-snug text-fg-muted font-mono">{install.lines.join("\n")}</pre>
      )}
    </div>
  );
}

/** Picks a precision and loads the model, with download progress */
export function LLMLoadCard() {
  const spec = useLLMStore((s) => s.spec)!;
  const device = useLLMStore((s) => s.device);
  const loading = useLLMStore((s) => s.loading);
  const loadError = useLLMStore((s) => s.loadError);
  const quantOk = device?.kind === "cuda";
  const [precision, setPrecision] = useState<Precision>(() => {
    // The largest precision that fits, preferring 16-bit
    const order: Precision[] = quantOk ? ["bf16", "int8", "nf4"] : ["bf16"];
    return order.find((p) => fitFor(estimateMemory(spec, "run", p, 4096)!.total, device?.memory) === "fits") ?? order[order.length - 1];
  });

  if (spec.loadable === false || !spec.source) {
    return (
      <div className={`${card} text-xs text-fg-secondary`}>
        {spec.quantization?.method === "gguf"
          ? "GGUF files run in llama.cpp, not PyTorch. To chat with or fine-tune this model here, import the original model from Hugging Face (the GGUF's model card usually links to it)."
          : "This model can't be loaded here: it needs its config.json next to the weights."}
      </div>
    );
  }

  const est = estimateMemory(spec, "run", precision, 4096)!;
  const fit = fitFor(est.total, device?.memory);
  const pct = loading?.bytes && loading.total ? Math.min(100, Math.round((loading.bytes / loading.total) * 100)) : null;

  return (
    <div className={card}>
      <div className="flex flex-wrap items-center gap-3">
        <Segmented
          label="Precision"
          size="sm"
          value={precision}
          onChange={setPrecision}
          options={(["bf16", "int8", "nf4"] as Precision[]).map((p) => ({
            value: p,
            label: PRECISION_LABELS[p],
            disabled: p !== "bf16" && !quantOk,
            title: p !== "bf16" && !quantOk ? "8-bit and 4-bit loading needs an NVIDIA GPU" : undefined,
          }))}
        />
        <span className={`text-[11px] ${fit === "too-big" ? "text-danger" : fit === "tight" ? "text-warning" : "text-fg-muted"}`}>
          About {formatBytes(est.total)}{fit === "too-big" ? ", more than this computer has" : fit === "tight" ? ", a tight fit" : ""}
        </span>
      </div>
      <div className="mt-3 flex items-center gap-2">
        {loading ? (
          <>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] text-fg-secondary mb-1">
                {loading.status === "downloading"
                  ? `Downloading${loading.bytes ? ` ${formatBytes(loading.bytes)}` : ""}${loading.total ? ` of ${formatBytes(loading.total)}` : ""}…`
                  : "Loading weights…"}
              </div>
              <div className="h-1.5 rounded-full bg-surface-raised overflow-hidden" role="progressbar" aria-valuenow={pct ?? undefined} aria-valuemin={0} aria-valuemax={100}>
                <div className={`h-full bg-accent ${pct == null ? "animate-pulse w-1/3" : ""}`} style={pct != null ? { width: `${pct}%` } : undefined} />
              </div>
            </div>
            {loading.status === "downloading" && <Button size="sm" onClick={() => void cancelLoad()}>Cancel</Button>}
          </>
        ) : (
          <Button size="sm" variant="primary" onClick={() => void loadLLM(precision)}>
            {spec.source.kind === "hf" ? "Download and load" : "Load"}
          </Button>
        )}
      </div>
      {spec.source.kind === "hf" && !loading && spec.stored_bytes ? (
        <div className="mt-1.5 text-[11px] text-fg-muted">Downloads {formatBytes(spec.stored_bytes)} the first time, into the Hugging Face cache.</div>
      ) : null}
      {loadError && <div role="alert" className="mt-2 px-3 py-2 rounded-lg text-xs bg-danger-subtle text-danger">{loadError}</div>}
    </div>
  );
}

/** Chat with the loaded model */
export default function LLMChat() {
  const missing = useLLMStore((s) => s.missing);
  const loaded = useLLMStore((s) => s.loaded);
  const chat = useLLMStore((s) => s.chat);
  const generating = useLLMStore((s) => s.generating);
  const systemPrompt = useLLMStore((s) => s.systemPrompt);
  const maxNewTokens = useLLMStore((s) => s.maxNewTokens);
  const temperature = useLLMStore((s) => s.temperature);
  const [draft, setDraft] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => { void refreshLLMStatus(); }, []);
  useEffect(() => { listRef.current?.scrollTo({ top: listRef.current.scrollHeight }); }, [chat]);

  if (missing === null) return <div className="text-xs text-fg-muted">Checking Python packages…</div>;
  if (missing.some((m) => m !== "bitsandbytes")) return <LLMInstallCard why="Chatting with the model runs it with Hugging Face transformers." />;
  if (!loaded) return <LLMLoadCard />;

  const send = () => { void sendChatMessage(draft); setDraft(""); };
  const field = "px-2 h-7 text-xs rounded-md bg-surface-raised border border-line text-fg outline-none focus:border-accent";

  return (
    <div className="flex flex-col h-full min-h-0 gap-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-fg-muted">
        <span className="text-fg-secondary">
          {PRECISION_LABELS[loaded.precision]} on {loaded.device_name} · {formatBytes(loaded.memory_used)}
          {loaded.adapter && <span className="text-accent-text"> · with fine-tuned adapter</span>}
        </span>
        <button type="button" className="hover:text-fg underline-offset-2 hover:underline" onClick={() => setShowSettings((v) => !v)}>Settings</button>
        <button type="button" className="hover:text-fg underline-offset-2 hover:underline disabled:opacity-50" onClick={clearChat} disabled={generating || chat.length === 0}>Clear chat</button>
        <button type="button" className="hover:text-fg underline-offset-2 hover:underline disabled:opacity-50" onClick={() => void unloadLLM()} disabled={generating}>Unload</button>
      </div>
      {showSettings && (
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-fg-secondary">
          <label className="flex items-center gap-1.5 flex-1 min-w-[220px]">System prompt
            <input className={`${field} flex-1`} value={systemPrompt} placeholder="You are a helpful assistant."
              onChange={(e) => useLLMStore.setState({ systemPrompt: e.target.value })} />
          </label>
          <label className="flex items-center gap-1.5">Max tokens
            <input className={`${field} w-16`} type="number" min={1} max={8192} value={maxNewTokens}
              onChange={(e) => useLLMStore.setState({ maxNewTokens: Math.max(1, Number(e.target.value) || 1) })} />
          </label>
          <label className="flex items-center gap-1.5" title="0 always picks the likeliest token; higher is more varied">Temperature
            <input className={`${field} w-14`} type="number" min={0} max={2} step={0.1} value={temperature}
              onChange={(e) => useLLMStore.setState({ temperature: Math.max(0, Number(e.target.value) || 0) })} />
          </label>
        </div>
      )}
      <div ref={listRef} className="flex-1 min-h-0 overflow-auto space-y-2 pr-1" aria-live="polite">
        {chat.length === 0 && <div className="text-xs text-fg-muted">Say something to the model.</div>}
        {chat.map((m, i) => (
          <div key={i} className={m.role === "user" ? "flex justify-end" : ""}>
            <div className={`max-w-[85%] px-3 py-2 rounded-lg text-xs whitespace-pre-wrap break-words ${
              m.role === "user" ? "bg-accent-subtle text-fg" : "bg-surface-raised text-fg"}`}>
              {m.content || (m.error ? "" : <span className="text-fg-muted animate-pulse">…</span>)}
              {m.error && <div role="alert" className="text-danger">{m.error}</div>}
              {m.stats && (
                <div className="mt-1 text-[10px] text-fg-muted tabular-nums">
                  {m.stats.tokens} tokens · {m.stats.tokensPerSecond != null ? `${m.stats.tokensPerSecond} tokens/s` : `${m.stats.seconds}s`}
                  {m.stats.stopped && " · stopped"}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="flex gap-2 items-end">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !generating) { e.preventDefault(); send(); } }}
          rows={2}
          placeholder="Message (Enter to send, Shift+Enter for a new line)"
          aria-label="Message"
          className="flex-1 min-w-0 p-2 text-xs rounded-lg resize-none outline-none bg-surface-raised border border-line text-fg focus:border-accent"
        />
        {generating
          ? <Button size="sm" onClick={() => void stopGenerating()}>Stop</Button>
          : <Button size="sm" variant="primary" onClick={send} disabled={!draft.trim()}>Send</Button>}
      </div>
    </div>
  );
}
