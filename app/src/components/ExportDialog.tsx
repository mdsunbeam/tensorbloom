import { useState } from "react";
import { sidecarCall } from "../lib/sidecar";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore } from "../store/training-store";
import { buildSidecarGraph, graphInputShapes } from "../lib/sidecar-graph";
import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import Segmented from "./ui/Segmented";

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

  const fieldInput = "flex-1 min-w-0 px-2.5 h-8 text-xs rounded-lg bg-surface-raised border border-line text-fg outline-none focus:border-accent";

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Export model"
      description="Save the model as ONNX or TorchScript"
      footer={
        <>
          <Button onClick={onClose}>{result ? "Done" : "Cancel"}</Button>
          <Button variant="primary" onClick={handleExport} disabled={loading}>{loading ? "Exporting…" : "Export"}</Button>
        </>
      }
    >
      <div className="p-5 space-y-4">
        <div>
          <div className="block text-[11px] font-medium mb-1.5 text-fg-secondary">Format</div>
          <Segmented
            label="Format"
            value={format}
            onChange={(f) => { setFormat(f); setOutputPath(outputPath.replace(/\.(onnx|pt)$/, f === "onnx" ? ".onnx" : ".pt")); }}
            options={[{ value: "onnx", label: "ONNX (.onnx)" }, { value: "torchscript", label: "TorchScript (.pt)" }]}
          />
        </div>

        {format === "torchscript" && (
          <div>
            <div className="block text-[11px] font-medium mb-1.5 text-fg-secondary">Method</div>
            <Segmented label="Method" size="sm" value={tsMethod} onChange={setTsMethod}
              options={[{ value: "trace", label: "Trace" }, { value: "script", label: "Script" }]} />
          </div>
        )}

        <div>
          <div className="block text-[11px] font-medium mb-1.5 text-fg-secondary">Weights</div>
          <Segmented
            label="Weights"
            size="sm"
            value={weights}
            onChange={setWeights}
            options={WEIGHT_OPTIONS.map((w) => {
              const unavailable = w.value === "trained" && !hasTrainedRun;
              return { ...w, disabled: unavailable, title: unavailable ? "No trained weights yet. Finish a training run first." : undefined };
            })}
          />
          {weights === "checkpoint" && (
            <div className="flex gap-2 mt-2">
              <input type="text" value={checkpointPath || lastCheckpoint || ""} placeholder="Path to checkpoint .pt"
                aria-label="Checkpoint file"
                onChange={(e) => setCheckpointPath(e.target.value)}
                className={fieldInput} />
              <Button onClick={browseCheckpoint}>Browse</Button>
            </div>
          )}
          {weights === "random" && (
            <div className="text-[11px] mt-1.5 text-fg-muted">
              Exports the architecture with freshly initialized weights.
              {!hasTrainedRun && " To export trained weights, finish a training run first or pick a checkpoint file."}
            </div>
          )}
        </div>

        <div>
          <label htmlFor="export-output-path" className="block text-[11px] font-medium mb-1.5 text-fg-secondary">Output path</label>
          <div className="flex gap-2">
            <input id="export-output-path" type="text" value={outputPath} onChange={(e) => setOutputPath(e.target.value)} className={fieldInput} />
            <Button onClick={browseOutput}>Browse</Button>
          </div>
        </div>

        {error && <div role="alert" className="px-3 py-2 rounded-lg text-xs bg-danger-subtle text-danger">{error}</div>}
        {result && <div role="status" className="px-3 py-2 rounded-lg text-xs bg-success-subtle text-success break-all">{result}</div>}
      </div>
    </Dialog>
  );
}
