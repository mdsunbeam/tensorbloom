import { useState } from "react";
import { useTrainingStore } from "../store/training-store";

interface SweepDialogProps {
  open: boolean;
  onClose: () => void;
  onLaunch: (configs: SweepRun[]) => void;
}

export interface SweepRun {
  id: number;
  lr: number;
  batchSize: number;
  optimizer: string;
  label: string;
}

export default function SweepDialog({ open, onClose, onLaunch }: SweepDialogProps) {
  const baseConfig = useTrainingStore((s) => s.config);
  const [lrValues, setLrValues] = useState("0.001, 0.0005, 0.0001");
  const [bsValues, setBsValues] = useState("32");
  const [optValues, setOptValues] = useState(baseConfig.optimizer.type);

  if (!open) return null;

  const parseList = (s: string): string[] => s.split(",").map((v) => v.trim()).filter(Boolean);

  const generateRuns = (): SweepRun[] => {
    const lrs = parseList(lrValues).map(Number).filter((v) => !isNaN(v));
    const bss = parseList(bsValues).map(Number).filter((v) => !isNaN(v));
    const opts = parseList(optValues);

    const runs: SweepRun[] = [];
    let id = 0;
    for (const lr of lrs) {
      for (const bs of bss) {
        for (const opt of opts) {
          runs.push({
            id: ++id,
            lr,
            batchSize: bs,
            optimizer: opt,
            label: `lr=${lr}, bs=${bs}, ${opt}`,
          });
        }
      }
    }
    return runs;
  };

  const runs = generateRuns();

  return (
    <>
      <div className="fixed inset-0 z-50" style={{ background: "rgba(0,0,0,0.5)" }} onClick={onClose} />
      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] max-h-[75vh] rounded-xl shadow-2xl flex flex-col overflow-hidden"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div>
            <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Hyperparameter Sweep</div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
              Define ranges — each combination becomes a separate run
            </div>
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-md"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        <div className="p-5 space-y-4">
          <ParamField label="Learning Rates (comma-separated)" value={lrValues} onChange={setLrValues}
            hint="e.g., 0.01, 0.001, 0.0001" />
          <ParamField label="Batch Sizes (comma-separated)" value={bsValues} onChange={setBsValues}
            hint="e.g., 16, 32, 64" />
          <ParamField label="Optimizers (comma-separated)" value={optValues} onChange={setOptValues}
            hint="e.g., Adam, AdamW, SGD" />

          {/* Preview */}
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>
              {runs.length} run{runs.length !== 1 ? "s" : ""} will be generated
            </div>
            <div className="max-h-[120px] overflow-y-auto space-y-0.5">
              {runs.slice(0, 20).map((r) => (
                <div key={r.id} className="text-[11px] px-2 py-0.5 rounded" style={{ color: "var(--text-secondary)", background: "var(--bg-tertiary)" }}>
                  {r.label}
                </div>
              ))}
              {runs.length > 20 && (
                <div className="text-[11px] px-2 py-0.5" style={{ color: "var(--text-muted)" }}>
                  ...and {runs.length - 20} more
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <button onClick={onClose} className="px-4 py-1.5 text-xs font-medium rounded-lg"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}>
            Cancel
          </button>
          <button
            onClick={() => { onLaunch(runs); onClose(); }}
            disabled={runs.length === 0}
            className="px-4 py-1.5 text-xs font-medium rounded-lg text-white"
            style={{ background: runs.length > 0 ? "var(--accent)" : "var(--text-muted)" }}>
            Launch {runs.length} Run{runs.length !== 1 ? "s" : ""}
          </button>
        </div>
      </div>
    </>
  );
}

function ParamField({ label, value, onChange, hint }: { label: string; value: string; onChange: (v: string) => void; hint: string }) {
  return (
    <div>
      <label className="block text-[11px] font-medium mb-1" style={{ color: "var(--text-secondary)" }}>{label}</label>
      <input type="text" value={value} onChange={(e) => onChange(e.target.value)}
        className="w-full px-2.5 py-[6px] text-xs rounded-lg"
        style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
      <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>{hint}</div>
    </div>
  );
}
