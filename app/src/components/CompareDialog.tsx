import { useMemo } from "react";
import { computeModelSummary, formatParamCount, formatFlops } from "../lib/model-summary";
import type { Node } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";

interface CompareDialogProps {
  open: boolean;
  onClose: () => void;
  modelA: { name: string; nodes: Node<NodeData>[] };
  modelB: { name: string; nodes: Node<NodeData>[] };
}

export default function CompareDialog({ open, onClose, modelA, modelB }: CompareDialogProps) {
  const summaryA = useMemo(() => computeModelSummary(modelA.nodes), [modelA.nodes]);
  const summaryB = useMemo(() => computeModelSummary(modelB.nodes), [modelB.nodes]);

  if (!open) return null;

  const rows: Array<{ label: string; a: string; b: string; better?: "a" | "b" | "equal" }> = [
    {
      label: "Parameters",
      a: formatParamCount(summaryA.totalParams),
      b: formatParamCount(summaryB.totalParams),
      better: summaryA.totalParams < summaryB.totalParams ? "a" : summaryA.totalParams > summaryB.totalParams ? "b" : "equal",
    },
    {
      label: "Model Size",
      a: `${summaryA.estimatedSizeMB.toFixed(1)} MB`,
      b: `${summaryB.estimatedSizeMB.toFixed(1)} MB`,
      better: summaryA.estimatedSizeMB < summaryB.estimatedSizeMB ? "a" : summaryA.estimatedSizeMB > summaryB.estimatedSizeMB ? "b" : "equal",
    },
    {
      label: "FLOPs",
      a: formatFlops(summaryA.estimatedFlops),
      b: formatFlops(summaryB.estimatedFlops),
      better: summaryA.estimatedFlops < summaryB.estimatedFlops ? "a" : summaryA.estimatedFlops > summaryB.estimatedFlops ? "b" : "equal",
    },
    {
      label: "Est. VRAM",
      a: `${summaryA.estimatedVramMB.toFixed(0)} MB`,
      b: `${summaryB.estimatedVramMB.toFixed(0)} MB`,
      better: summaryA.estimatedVramMB < summaryB.estimatedVramMB ? "a" : summaryA.estimatedVramMB > summaryB.estimatedVramMB ? "b" : "equal",
    },
    {
      label: "Layers",
      a: String(modelA.nodes.filter((n) => !["Input", "Output"].includes(n.data.moduleType)).length),
      b: String(modelB.nodes.filter((n) => !["Input", "Output"].includes(n.data.moduleType)).length),
    },
  ];

  // Layer type diff
  const typesA = new Map<string, number>();
  const typesB = new Map<string, number>();
  for (const n of modelA.nodes) {
    if (n.data.moduleType !== "Input" && n.data.moduleType !== "Output")
      typesA.set(n.data.moduleType, (typesA.get(n.data.moduleType) ?? 0) + 1);
  }
  for (const n of modelB.nodes) {
    if (n.data.moduleType !== "Input" && n.data.moduleType !== "Output")
      typesB.set(n.data.moduleType, (typesB.get(n.data.moduleType) ?? 0) + 1);
  }
  const allTypes = new Set([...typesA.keys(), ...typesB.keys()]);

  return (
    <>
      <div className="fixed inset-0 z-50" style={{ background: "rgba(0,0,0,0.5)" }} onClick={onClose} />
      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[540px] max-h-[75vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Model Comparison</div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-md"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {/* Summary comparison */}
          <table className="w-full text-xs">
            <thead>
              <tr style={{ color: "var(--text-muted)" }}>
                <th className="text-left font-semibold pb-2 w-1/3">Metric</th>
                <th className="text-right font-semibold pb-2 w-1/3" style={{ color: "var(--accent-text)" }}>{modelA.name}</th>
                <th className="text-right font-semibold pb-2 w-1/3" style={{ color: "#f97316" }}>{modelB.name}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label} style={{ borderTop: "1px solid var(--border)" }}>
                  <td className="py-1.5" style={{ color: "var(--text-secondary)" }}>{r.label}</td>
                  <td className="text-right py-1.5 tabular-nums font-medium" style={{
                    color: r.better === "a" ? "var(--success)" : "var(--text-primary)",
                  }}>{r.better === "a" && <SmallerMark />}{r.a}</td>
                  <td className="text-right py-1.5 tabular-nums font-medium" style={{
                    color: r.better === "b" ? "var(--success)" : "var(--text-primary)",
                  }}>{r.better === "b" && <SmallerMark />}{r.b}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Layer diff */}
          <div className="mt-4 pt-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>
              Layer Types
            </div>
            <div className="space-y-0.5">
              {[...allTypes].sort().map((type) => {
                const countA = typesA.get(type) ?? 0;
                const countB = typesB.get(type) ?? 0;
                return (
                  <div key={type} className="flex items-center text-[11px] py-0.5">
                    <span className="w-1/3 truncate" style={{ color: "var(--text-primary)" }}>{type}</span>
                    <span className="w-1/3 text-right tabular-nums" style={{ color: countA > 0 ? "var(--text-primary)" : "var(--text-muted)" }}>{countA || "-"}</span>
                    <span className="w-1/3 text-right tabular-nums" style={{ color: countB > 0 ? "var(--text-primary)" : "var(--text-muted)" }}>{countB || "-"}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/** Marks the smaller (cheaper) value in a row with a shape as well as a color */
function SmallerMark() {
  return (
    <>
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" className="inline-block mr-1 -mt-0.5" aria-hidden="true">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      <span className="sr-only">(smaller) </span>
    </>
  );
}
