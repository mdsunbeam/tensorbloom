import { useEffect, useRef, useState } from "react";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import Segmented from "./ui/Segmented";
import LLMMemoryTable from "./LLMMemoryTable";
import { sidecarCall } from "../lib/sidecar";
import { activeParams, formatBytes, type LLMDevice, type LLMSpec } from "../lib/llm";
import { formatParamCount } from "../lib/model-summary";
import { useLLMStore } from "../store/llm-store";

type Source = "hf" | "file";

const EXAMPLES = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mixtral-8x7B-v0.1"];

const fieldInput = "flex-1 min-w-0 px-2.5 h-8 text-xs rounded-lg bg-surface-raised border border-line text-fg outline-none focus:border-accent";
const fieldLabel = "block text-[11px] font-medium mb-1.5 text-fg-secondary";

/** Loads the device's memory once per session, for the memory table */
export function useLLMDevice(): LLMDevice | null {
  const device = useLLMStore((s) => s.device);
  useEffect(() => {
    if (device) return;
    sidecarCall<LLMDevice>("llm.device").then((d) => useLLMStore.getState().setDevice(d)).catch(() => { /* shown as unknown */ });
  }, [device]);
  return device;
}

function Fact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-fg-muted">{label}</div>
      <div className="text-xs text-fg tabular-nums truncate">{value}</div>
    </div>
  );
}

/** The architecture at a glance */
export function LLMFacts({ spec }: { spec: LLMSpec }) {
  const attention = spec.n_kv_heads === spec.n_heads
    ? `${spec.n_heads} heads`
    : spec.n_kv_heads === 1 ? `${spec.n_heads} heads, multi-query` : `${spec.n_heads} heads, ${spec.n_kv_heads} KV (grouped)`;
  return (
    <div className="grid grid-cols-3 gap-x-4 gap-y-2.5">
      <Fact label="Parameters" value={`${formatParamCount(spec.params)}${spec.params_exact ? "" : " (computed)"}`} />
      <Fact label="Layers" value={spec.n_layers} />
      <Fact label="Hidden size" value={spec.d_model.toLocaleString()} />
      <Fact label="Attention" value={attention} />
      <Fact label="MLP" value={spec.n_experts
        ? `${spec.n_experts} experts, ${spec.experts_per_token} per token (${formatParamCount(activeParams(spec))} active)`
        : `${spec.mlp.toUpperCase()}, ${spec.ffn_dim.toLocaleString()} wide`} />
      <Fact label="Vocabulary" value={spec.vocab_size.toLocaleString()} />
      <Fact label="Context" value={`${spec.max_context.toLocaleString()} tokens`} />
      <Fact label="Positions" value={spec.rope_theta ? `Rotary (base ${spec.rope_theta.toLocaleString()})` : spec.learned_positions ? "Learned" : "None"} />
      <Fact label="Stored as" value={spec.stored_dtype
        ? `${spec.stored_dtype.toUpperCase()}${spec.stored_bytes ? `, ${formatBytes(spec.stored_bytes)}` : ""}`
        : spec.torch_dtype ?? "–"} />
    </div>
  );
}

export default function ImportLLMDialog({ open, onClose, onImport }: {
  open: boolean;
  onClose: () => void;
  /** Resolves to false when the import was called off (e.g. at the unsaved-changes prompt) */
  onImport: (spec: LLMSpec) => Promise<boolean>;
}) {
  const [source, setSource] = useState<Source>("hf");
  const [modelId, setModelId] = useState("");
  const [path, setPath] = useState("");
  const token = useLLMStore((s) => s.hfToken);
  const [showToken, setShowToken] = useState(false);
  const [spec, setSpec] = useState<LLMSpec | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const request = useRef(0);
  const device = useLLMDevice();

  const inspect = async () => {
    const id = ++request.current;
    setError("");
    setSpec(null);
    setLoading(true);
    try {
      const params = source === "hf" ? { model_id: modelId, token: token || undefined } : { path };
      const result = await sidecarCall<LLMSpec>("llm.inspect", params);
      if (id === request.current) setSpec(result);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (id === request.current) {
        setError(msg);
        if (/token/i.test(msg)) setShowToken(true);
      }
    } finally {
      if (id === request.current) setLoading(false);
    }
  };

  const browse = async () => {
    try {
      const { open: openDialog } = await import("@tauri-apps/plugin-dialog");
      const picked = await openDialog({
        multiple: false, directory: false, title: "Choose a model file",
        filters: [{ name: "Model files", extensions: ["gguf", "safetensors", "json"] }],
      });
      if (typeof picked === "string") { setPath(picked); setSpec(null); }
    } catch { /* dialog unavailable outside Tauri */ }
  };

  const add = async () => {
    if (!spec) return;
    setLoading(true);
    try {
      if (await onImport(spec)) onClose();
    } finally {
      setLoading(false);
    }
  };

  const ready = source === "hf" ? modelId.trim() !== "" : path.trim() !== "";

  return (
    <Dialog
      open={open}
      onClose={onClose}
      width={640}
      title="Import a language model"
      description="Reads the architecture from a Hugging Face model or a model file, without downloading the weights."
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          {spec
            ? <Button variant="primary" onClick={add} disabled={loading}>Open as graph</Button>
            : <Button variant="primary" onClick={inspect} disabled={loading || !ready}>{loading ? "Reading…" : "Read architecture"}</Button>}
        </>
      }
    >
      <div className="p-5 space-y-4">
        <Segmented
          label="Where the model is"
          value={source}
          onChange={(s) => { setSource(s); setSpec(null); setError(""); }}
          options={[{ value: "hf", label: "Hugging Face" }, { value: "file", label: "File on this computer" }]}
        />

        {source === "hf" ? (
          <div>
            <label htmlFor="llm-model-id" className={fieldLabel}>Model id</label>
            <input
              id="llm-model-id" type="text" value={modelId} placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct" spellCheck={false}
              onChange={(e) => { setModelId(e.target.value); setSpec(null); }}
              onKeyDown={(e) => { if (e.key === "Enter" && ready && !loading) void inspect(); }}
              className={`${fieldInput} w-full`}
            />
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {EXAMPLES.map((ex) => (
                <button key={ex} type="button" onClick={() => { setModelId(ex); setSpec(null); }}
                  className="px-2 h-6 rounded-md text-[11px] bg-surface-raised text-fg-secondary hover:text-fg border border-line">
                  {ex.split("/")[1]}
                </button>
              ))}
            </div>
            {showToken ? (
              <div className="mt-3">
                <label htmlFor="llm-hf-token" className={fieldLabel}>Hugging Face token (for gated or private models)</label>
                <input id="llm-hf-token" type="password" value={token} placeholder="hf_…" autoComplete="off"
                  onChange={(e) => useLLMStore.getState().setHfToken(e.target.value)} className={`${fieldInput} w-full`} />
                <div className="mt-1 text-[11px] text-fg-muted">Kept until you close TensorBloom, never saved. A token from <code>huggingface-cli login</code> is used automatically.</div>
              </div>
            ) : (
              <button type="button" onClick={() => setShowToken(true)} className="mt-2 text-[11px] text-accent-text hover:underline">Use a Hugging Face token…</button>
            )}
          </div>
        ) : (
          <div>
            <label htmlFor="llm-path" className={fieldLabel}>.gguf or .safetensors file, or a model folder</label>
            <div className="flex gap-2">
              <input id="llm-path" type="text" value={path} spellCheck={false} placeholder="/path/to/model.gguf"
                onChange={(e) => { setPath(e.target.value); setSpec(null); }}
                onKeyDown={(e) => { if (e.key === "Enter" && ready && !loading) void inspect(); }}
                className={fieldInput} />
              <Button onClick={browse}>Browse</Button>
            </div>
          </div>
        )}

        {error && <div role="alert" className="px-3 py-2 rounded-lg text-xs bg-danger-subtle text-danger">{error}</div>}

        {spec && (
          <div className="space-y-4">
            <div className="rounded-lg border border-line p-3">
              <div className="flex items-baseline justify-between gap-2 mb-3">
                <div className="text-sm font-medium text-fg truncate">{spec.name}</div>
                <div className="text-[11px] text-fg-muted shrink-0">{spec.family}</div>
              </div>
              <LLMFacts spec={spec} />
            </div>
            {spec.warnings.length > 0 && (
              <ul className="px-3 py-2 rounded-lg text-xs bg-warning-subtle text-fg-secondary space-y-1 list-none">
                {spec.warnings.map((w) => <li key={w}>{w}</li>)}
              </ul>
            )}
            <div>
              <div className={fieldLabel}>Memory needed</div>
              <LLMMemoryTable spec={spec} device={device} />
            </div>
          </div>
        )}
      </div>
    </Dialog>
  );
}
