import { useEffect } from "react";
import Button from "./ui/Button";
import Segmented from "./ui/Segmented";
import { LLMInstallCard } from "./LLMChat";
import { useLLMStore } from "../store/llm-store";
import { useTrainingStore } from "../store/training-store";
import { startTraining, stopTraining } from "../lib/training";
import { refreshLLMStatus } from "../lib/llm-runtime";
import {
  estimateMemory, finetuneProblem, fitFor, formatBytes, loraParams,
  type FinetuneDataset, type FinetuneSettings,
} from "../lib/llm";
import { formatParamCount } from "../lib/model-summary";

const field = "min-w-0 px-2.5 h-8 text-xs rounded-lg bg-surface-raised border border-line text-fg outline-none focus:border-accent";
const label = "block text-[11px] font-medium mb-1 text-fg-secondary";

const DATASET_EXAMPLES = [
  { name: "mlabonne/guanaco-llama2-1k", note: "1k chats" },
  { name: "yahma/alpaca-cleaned", note: "instructions" },
  { name: "databricks/databricks-dolly-15k", note: "instructions, 15k" },
];

function update(patch: Partial<FinetuneSettings>) {
  useLLMStore.setState((s) => ({ finetune: { ...s.finetune, ...patch } }));
}

function updateDataset(patch: Partial<FinetuneDataset>) {
  useLLMStore.setState((s) => ({ finetune: { ...s.finetune, dataset: { ...s.finetune.dataset, ...patch } as FinetuneDataset } }));
}

/** LoRA / QLoRA fine-tuning settings and the button that starts it */
export default function LLMFinetune({ onShowCharts }: { onShowCharts: () => void }) {
  const spec = useLLMStore((s) => s.spec)!;
  const ft = useLLMStore((s) => s.finetune);
  const device = useLLMStore((s) => s.device);
  const missing = useLLMStore((s) => s.missing);
  const loaded = useLLMStore((s) => s.loaded);
  const config = useTrainingStore((s) => s.config);
  const status = useTrainingStore((s) => s.status);
  const running = status === "running" || status === "starting";

  useEffect(() => { void refreshLLMStatus(); }, []);

  if (missing === null) return <div className="text-xs text-fg-muted">Checking Python packages…</div>;
  if (missing.some((m) => m !== "bitsandbytes")) return <LLMInstallCard why="Fine-tuning uses Hugging Face transformers, peft and datasets." />;

  const est = estimateMemory(spec, "lora", ft.precision, ft.maxLength, config.batchSize, ft.rank)!;
  const fit = fitFor(est.total, device?.memory);
  const quantOk = device?.kind === "cuda" && !missing.includes("bitsandbytes");
  const problem = finetuneProblem(spec, ft);
  const ds = ft.dataset;

  return (
    <div className="grid gap-5 lg:grid-cols-2 max-w-6xl">
      <section aria-label="Dataset" className="space-y-3">
        <Segmented label="Dataset source" size="sm" value={ds.kind}
          onChange={(kind) => update({ dataset: kind === "hf" ? { kind, name: "", split: "train" } : { kind, path: "" } })}
          options={[{ value: "hf", label: "Hugging Face dataset" }, { value: "file", label: "File" }]} />
        {ds.kind === "hf" ? (
          <>
            <div className="flex gap-2">
              <div className="flex-1 min-w-0">
                <label htmlFor="ft-ds-name" className={label}>Dataset</label>
                <input id="ft-ds-name" className={`${field} w-full`} value={ds.name} spellCheck={false} placeholder="e.g. mlabonne/guanaco-llama2-1k"
                  onChange={(e) => updateDataset({ name: e.target.value })} />
              </div>
              <div className="w-24">
                <label htmlFor="ft-ds-split" className={label}>Split</label>
                <input id="ft-ds-split" className={`${field} w-full`} value={ds.split} onChange={(e) => updateDataset({ split: e.target.value })} />
              </div>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {DATASET_EXAMPLES.map((d) => (
                <button key={d.name} type="button" onClick={() => updateDataset({ name: d.name })}
                  className="px-2 h-6 rounded-md text-[11px] bg-surface-raised text-fg-secondary hover:text-fg border border-line">
                  {d.name.split("/")[1]} <span className="text-fg-muted">· {d.note}</span>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div>
            <label htmlFor="ft-ds-path" className={label}>.jsonl, .json or .txt file</label>
            <div className="flex gap-2">
              <input id="ft-ds-path" className={`${field} flex-1`} value={ds.path} spellCheck={false} placeholder="/path/to/data.jsonl"
                onChange={(e) => updateDataset({ path: e.target.value })} />
              <Button size="sm" onClick={async () => {
                try {
                  const { open } = await import("@tauri-apps/plugin-dialog");
                  const picked = await open({ multiple: false, directory: false, filters: [{ name: "Text data", extensions: ["jsonl", "json", "txt", "md"] }] });
                  if (typeof picked === "string") updateDataset({ path: picked });
                } catch { /* dialog unavailable outside Tauri */ }
              }}>Browse</Button>
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <div className="flex-1">
            <label htmlFor="ft-ds-col" className={label}>Column (optional)</label>
            <input id="ft-ds-col" className={`${field} w-full`} value={ds.column ?? ""} placeholder="found automatically"
              onChange={(e) => updateDataset({ column: e.target.value || undefined })} />
          </div>
          <div className="w-28">
            <label htmlFor="ft-max-samples" className={label}>Max examples</label>
            <input id="ft-max-samples" type="number" min={1} className={`${field} w-full`} value={ft.maxSamples}
              onChange={(e) => update({ maxSamples: Math.max(1, Number(e.target.value) || 1) })} />
          </div>
        </div>
        <p className="text-[11px] text-fg-muted leading-relaxed">
          Rows with <code>messages</code> (chats), <code>instruction</code>/<code>output</code>, <code>prompt</code>/<code>completion</code> or <code>text</code> are used as they are; chats are written in the model's own chat format.
        </p>
      </section>

      <section aria-label="Method" className="space-y-3">
        <div>
          <div className={label}>Method</div>
          <Segmented label="Method" size="sm" value={ft.precision} onChange={(precision) => update({ precision })}
            options={[
              { value: "bf16", label: "LoRA (16-bit weights)" },
              { value: "nf4", label: "QLoRA (4-bit weights)", disabled: !quantOk, title: quantOk ? undefined : "QLoRA needs an NVIDIA GPU and bitsandbytes" },
            ]} />
        </div>
        <div className="grid grid-cols-4 gap-2">
          {([["rank", "Rank", 1, 1], ["alpha", "Alpha", 1, 1], ["maxLength", "Tokens / example", 16, 16], ["gradAccum", "Grad. accum.", 1, 1]] as const).map(([key, text, min, step]) => (
            <div key={key}>
              <label htmlFor={`ft-${key}`} className={label}>{text}</label>
              <input id={`ft-${key}`} type="number" min={min} step={step} className={`${field} w-full`} value={ft[key]}
                onChange={(e) => update({ [key]: Math.max(min, Number(e.target.value) || min) } as Partial<FinetuneSettings>)} />
            </div>
          ))}
        </div>
        <div className="text-[11px] text-fg-secondary">
          Trains {formatParamCount(loraParams(spec, ft.rank))} adapter parameters
          ({(100 * loraParams(spec, ft.rank) / spec.params).toFixed(2)}% of the model).
          Learning rate {config.optimizer.lr}, {config.epochs} epoch{config.epochs === 1 ? "" : "s"}, batch size {config.batchSize} come from the Training panel.
        </div>
        <div className={`text-[11px] ${fit === "too-big" ? "text-danger" : fit === "tight" ? "text-warning" : "text-fg-muted"}`} title={est.parts.map((p) => `${p.label}: ${formatBytes(p.bytes)}`).join("\n")}>
          Needs about {formatBytes(est.total)}{device?.memory ? ` of ${formatBytes(device.memory)}` : ""}
          {fit === "too-big" ? ". Try QLoRA, fewer tokens per example or a smaller batch." : fit === "tight" ? ", a tight fit." : "."}
        </div>
        <div className="flex items-center gap-2">
          {running
            ? <Button size="sm" onClick={() => void stopTraining()}>Stop</Button>
            : <Button size="sm" variant="primary" disabled={!!problem} title={problem ?? undefined} onClick={() => { onShowCharts(); void startTraining(); }}>Fine-tune</Button>}
          {running && <button type="button" className="text-[11px] text-accent-text hover:underline" onClick={onShowCharts}>Show progress</button>}
        </div>
        {loaded?.adapter && !running && (
          <div className="px-3 py-2 rounded-lg text-xs bg-success-subtle text-fg-secondary">
            The loaded model now uses the fine-tuned adapter, saved in <code className="break-all">{loaded.adapter}</code>.{" "}
            <button type="button" className="text-accent-text hover:underline" onClick={() => useLLMStore.setState({ view: "chat" })}>Chat with it</button>
          </div>
        )}
      </section>
    </div>
  );
}
