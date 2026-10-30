import { useState } from "react";
import { sidecarCall } from "../lib/sidecar";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore } from "../store/training-store";
import { buildSidecarGraph, graphInputShapes } from "../lib/sidecar-graph";
import { useEscapeKey } from "../lib/use-escape-key";

interface ExportDialogProps {
  open: boolean;
  onClose: () => void;
}

type ExportFormat = "onnx" | "torchscript";
type TSMethod = "trace" | "script";
type WeightSource = "trained" | "checkpoint" | "random";

const WEIGHT_OPTIONS: { value: WeightSource; label: string }[] = [
  { value: "trained", label: "Last training run" },
  { value: "checkpoint", label: "Checkpoint file" },
  { value: "random", label: "Untrained" },
];

export default function ExportDialog({ open, onClose }: ExportDialogProps) {
  const lastCheckpoint = useTrainingStore((s) => s.lastCheckpointPath);
  // The sidecar keeps weights in memory only after a run completes, so "trained" isn't available before that
  const hasTrainedRun = useTrainingStore((s) => s.status === "finished");
  const [format, setFormat] = useState<ExportFormat>("onnx");
  const [tsMethod, setTsMethod] = useState<TSMethod>("trace");
  const [outputPath, setOutputPath] = useState("model.onnx");
  const [weights, setWeights] = useState<WeightSource>(hasTrainedRun ? "trained" : "random");
  const [checkpointPath, setCheckpointPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEscapeKey(onClose, { enabled: open });

  if (!open) return null;

  const browseOutput = async () => {
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const path = await save({
        defaultPath: outputPath,
        filters: format === "onnx"
          ? [{ name: "ONNX", extensions: ["onnx"] }]
          : [{ name: "TorchScript", extensions: ["pt"] }],
        title: "Export Model",
      });
      if (path) setOutputPath(path);
    } catch { /* dialog unavailable outside Tauri */ }
  };

  const browseCheckpoint = async () => {
    try {
      const { open: openDialog } = await import("@tauri-apps/plugin-dialog");
      const path = await openDialog({
        multiple: false,
        directory: false,
        defaultPath: checkpointPath || lastCheckpoint || undefined,
        filters: [{ name: "PyTorch checkpoint", extensions: ["pt", "pth"] }],
        title: "Select Checkpoint",
      });
      if (path && typeof path === "string") setCheckpointPath(path);
    } catch { /* dialog unavailable outside Tauri */ }
  };

  const handleExport = async () => {
    setError(null);
    setResult(null);

    const ckpt = checkpointPath || lastCheckpoint || "";
    if (weights === "checkpoint" && !ckpt) {
      setError("Choose a checkpoint file to export.");
      return;
    }

    setLoading(true);

    const { nodes, edges } = useGraphStore.getState();
    const graphJson = buildSidecarGraph(nodes, edges);

    const path = format === "onnx"
      ? (outputPath.endsWith(".onnx") ? outputPath : outputPath + ".onnx")
      : (outputPath.endsWith(".pt") ? outputPath : outputPath + ".pt");

    try {
      const method = format === "onnx" ? "export.onnx" : "export.torchscript";
      const res = await sidecarCall<{ path: string; weights: WeightSource }>(method, {
        graph: graphJson,
        input_shapes: graphInputShapes(graphJson),
        output_path: path,
        method: tsMethod,
        weights,
        checkpoint_path: weights === "checkpoint" ? ckpt : null,
      });
      const note = res.weights === "random" ? " (untrained weights)" : "";
      setResult(`Exported to ${res.path}${note}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="fixed inset-0 z-50" style={{ background: "rgba(0,0,0,0.5)" }} onClick={onClose} />
      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[440px] rounded-xl shadow-2xl flex flex-col overflow-hidden"
        style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between px-5 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
          <div>
            <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Export Model</div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>Export to ONNX or TorchScript</div>
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-md" style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-tertiary)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Format */}
          <div>
            <label className="block text-[11px] font-medium mb-2" style={{ color: "var(--text-secondary)" }}>Format</label>
            <div className="flex gap-2">
              {(["onnx", "torchscript"] as const).map((f) => (
                <button key={f} onClick={() => { setFormat(f); setOutputPath(outputPath.replace(/\.(onnx|pt)$/, f === "onnx" ? ".onnx" : ".pt")); }}
                  className="flex-1 py-2 text-xs font-medium rounded-lg transition-colors"
                  style={{
                    background: format === f ? "var(--accent)" : "var(--bg-tertiary)",
                    color: format === f ? "#fff" : "var(--text-secondary)",
                    border: `1px solid ${format === f ? "var(--accent)" : "var(--border)"}`,
                  }}>
                  {f === "onnx" ? "ONNX (.onnx)" : "TorchScript (.pt)"}
                </button>
              ))}
            </div>
          </div>

          {/* TorchScript method */}
          {format === "torchscript" && (
            <div>
              <label className="block text-[11px] font-medium mb-2" style={{ color: "var(--text-secondary)" }}>Method</label>
              <div className="flex gap-2">
                {(["trace", "script"] as const).map((m) => (
                  <button key={m} onClick={() => setTsMethod(m)}
                    className="flex-1 py-1.5 text-xs font-medium rounded-lg transition-colors"
                    style={{
                      background: tsMethod === m ? "var(--accent-subtle)" : "var(--bg-tertiary)",
                      color: tsMethod === m ? "var(--accent)" : "var(--text-secondary)",
                      border: `1px solid ${tsMethod === m ? "var(--accent)" : "var(--border)"}`,
                    }}>
                    {m === "trace" ? "Trace" : "Script"}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Weights */}
          <div>
            <label className="block text-[11px] font-medium mb-2" style={{ color: "var(--text-secondary)" }}>Weights</label>
            <div className="flex gap-2">
              {WEIGHT_OPTIONS.map((w) => {
                const unavailable = w.value === "trained" && !hasTrainedRun;
                return (
                  <button key={w.value} onClick={() => setWeights(w.value)} disabled={unavailable}
                    title={unavailable ? "No trained weights yet. Finish a training run first." : undefined}
                    className="flex-1 py-1.5 text-xs font-medium rounded-lg transition-colors"
                    style={{
                      background: weights === w.value ? "var(--accent-subtle)" : "var(--bg-tertiary)",
                      color: weights === w.value ? "var(--accent)" : "var(--text-secondary)",
                      border: `1px solid ${weights === w.value ? "var(--accent)" : "var(--border)"}`,
                      opacity: unavailable ? 0.5 : 1,
                      cursor: unavailable ? "not-allowed" : undefined,
                    }}>
                    {w.label}
                  </button>
                );
              })}
            </div>
            {weights === "checkpoint" && (
              <div className="flex gap-2 mt-2">
                <input type="text" value={checkpointPath || lastCheckpoint || ""} placeholder="Path to checkpoint .pt"
                  onChange={(e) => setCheckpointPath(e.target.value)}
                  className="flex-1 min-w-0 px-2.5 py-[6px] text-xs rounded-lg"
                  style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
                <button onClick={browseCheckpoint} className="px-3 text-xs rounded-lg"
                  style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}>Browse</button>
              </div>
            )}
            {weights === "random" && (
              <div className="text-[11px] mt-1.5" style={{ color: "var(--text-muted)" }}>
                Exports the architecture with freshly initialized weights.
                {!hasTrainedRun && " To export trained weights, finish a training run first or pick a checkpoint file."}
              </div>
            )}
          </div>

          {/* Output path */}
          <div>
            <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Output Path</label>
            <div className="flex gap-2">
              <input type="text" value={outputPath} onChange={(e) => setOutputPath(e.target.value)}
                className="flex-1 min-w-0 px-2.5 py-[6px] text-xs rounded-lg"
                style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
              <button onClick={browseOutput} className="px-3 text-xs rounded-lg"
                style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}>Browse</button>
            </div>
          </div>

          {error && <div className="px-3 py-2 rounded-lg text-xs" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>{error}</div>}
          {result && <div className="px-3 py-2 rounded-lg text-xs" style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}>{result}</div>}
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3" style={{ borderTop: "1px solid var(--border)" }}>
          <button onClick={onClose} className="px-4 py-1.5 text-xs font-medium rounded-lg"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}>Cancel</button>
          <button onClick={handleExport} disabled={loading}
            className="px-4 py-1.5 text-xs font-medium rounded-lg text-white"
            style={{ background: loading ? "var(--text-muted)" : "var(--accent)", opacity: loading ? 0.7 : 1 }}>
            {loading ? "Exporting..." : "Export"}
          </button>
        </div>
      </div>
    </>
  );
}
