import { useMemo, useState } from "react";
import Segmented from "./ui/Segmented";
import {
  estimateMemory, fitFor, formatBytes, precisionsFor, PRECISION_LABELS, TASK_LABELS,
  type Fit, type LLMDevice, type LLMSpec, type Precision, type Task,
} from "../lib/llm";

const FIT_STYLE: Record<Fit, { dot: string; label: string }> = {
  fits: { dot: "bg-success", label: "Fits" },
  tight: { dot: "bg-warning", label: "Tight" },
  "too-big": { dot: "bg-danger", label: "Too big" },
  unknown: { dot: "bg-fg-muted", label: "" },
};

const TASKS: Task[] = ["run", "lora", "full"];

function contextOptions(max: number): number[] {
  const steps = [2048, 8192, 32768, 131072].filter((c) => c < max);
  return [...steps, max].slice(-4);
}

function formatContext(n: number): string {
  return n >= 1024 ? `${Math.round(n / 1024)}k` : String(n);
}

/** Device memory in words, e.g. "RTX 4090 · 24 GB" */
export function deviceSummary(device: LLMDevice | null): string {
  if (!device) return "Checking this computer's memory…";
  if (!device.memory) return `${device.name} (memory unknown)`;
  const what = device.kind === "mps" ? "usable by the GPU" : device.kind === "cpu" ? "RAM, no GPU" : "GPU memory";
  return `${device.name.replace(/^NVIDIA (GeForce )?/, "")} · ${formatBytes(device.memory)} ${what}`;
}

/**
 * Memory an LLM needs to run, LoRA fine-tune, or fully fine-tune at each precision, colored by
 * whether it fits this computer. Estimates are rough: real use varies with the library and settings.
 */
export default function LLMMemoryTable({ spec, device }: { spec: LLMSpec; device: LLMDevice | null }) {
  const contexts = useMemo(() => contextOptions(spec.max_context || 2048), [spec.max_context]);
  const [context, setContext] = useState(() => contexts.find((c) => c >= 4096) ?? contexts[contexts.length - 1]);
  const precisions = precisionsFor(spec);
  const available = device?.memory ?? null;

  return (
    <div>
      <div className="flex items-center justify-between gap-3 mb-2">
        <div className="text-[11px] font-medium text-fg-secondary">Context length</div>
        <Segmented
          label="Context length"
          size="sm"
          value={String(context)}
          onChange={(v) => setContext(Number(v))}
          options={contexts.map((c) => ({ value: String(c), label: formatContext(c) }))}
        />
      </div>
      <table className="w-full text-xs border-separate border-spacing-0">
        <thead>
          <tr className="text-[11px] text-fg-muted">
            <th className="text-left font-medium py-1.5 pr-2">Weights</th>
            {TASKS.map((t) => <th key={t} className="text-left font-medium py-1.5 px-2">{TASK_LABELS[t]}</th>)}
          </tr>
        </thead>
        <tbody>
          {precisions.map((p: Precision) => (
            <tr key={p} className="border-t border-line">
              <td className="py-1.5 pr-2 text-fg-secondary whitespace-nowrap border-t border-line">
                {p === "stored" ? `As stored (${spec.stored_dtype?.toUpperCase() ?? "file"})` : PRECISION_LABELS[p]}
                {p === "nf4" && <span className="text-fg-muted"> (QLoRA)</span>}
              </td>
              {TASKS.map((t) => {
                const est = estimateMemory(spec, t, p, context);
                if (!est || (p === "stored" && t !== "run")) {
                  return <td key={t} className="py-1.5 px-2 text-fg-muted border-t border-line">–</td>;
                }
                const fit = fitFor(est.total, available);
                const style = FIT_STYLE[fit];
                const breakdown = est.parts.map((part) => `${part.label}: ${formatBytes(part.bytes)}`).join("\n");
                return (
                  <td key={t} className="py-1.5 px-2 border-t border-line" title={breakdown}>
                    <span className="inline-flex items-center gap-1.5 tabular-nums text-fg">
                      <span aria-hidden="true" className={`w-1.5 h-1.5 rounded-full shrink-0 ${style.dot}`} />
                      {formatBytes(est.total)}
                      {style.label && <span className="sr-only">, {style.label}</span>}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-fg-muted">
        <span>{deviceSummary(device)}</span>
        {available ? (
          <span className="inline-flex items-center gap-2">
            {(["fits", "tight", "too-big"] as Fit[]).map((f) => (
              <span key={f} className="inline-flex items-center gap-1">
                <span aria-hidden="true" className={`w-1.5 h-1.5 rounded-full ${FIT_STYLE[f].dot}`} />{FIT_STYLE[f].label}
              </span>
            ))}
          </span>
        ) : null}
      </div>
      <div className="mt-1 text-[11px] text-fg-muted">
        Rough estimates with batch size 1. Hover a cell for the breakdown.
        {device && device.kind !== "cuda" && " 8-bit and 4-bit loading needs an NVIDIA GPU."}
      </div>
    </div>
  );
}
