import { useState } from "react";
import { sidecarCall } from "../lib/sidecar";
import { useGraphStore } from "../store/graph-store";

interface ExportDialogProps {
  open: boolean;
  onClose: () => void;
}

type ExportFormat = "onnx" | "torchscript";
type TSMethod = "trace" | "script";

export default function ExportDialog({ open, onClose }: ExportDialogProps) {
  const [format, setFormat] = useState<ExportFormat>("onnx");
  const [tsMethod, setTsMethod] = useState<TSMethod>("trace");
  const [outputPath, setOutputPath] = useState("./model.onnx");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleExport = async () => {
    setError(null);
    setResult(null);
    setLoading(true);

    const { nodes, edges } = useGraphStore.getState();
    const graphJson = {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.moduleType === "Input" ? "placeholder" : n.data.moduleType === "Output" ? "output" : "call_module",
        moduleType: n.data.moduleType,
        label: n.data.label,
        params: n.data.params,
      })),
      edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
    };

    // Find input shapes
    const inputNode = nodes.find((n) => n.data.moduleType === "Input");
    let inputShape = [1, 3, 224, 224];
    if (inputNode) {
      try {
        const s = String(inputNode.data.params.shape ?? "[1, 3, 224, 224]");
        inputShape = JSON.parse(s.replace(/\(/g, "[").replace(/\)/g, "]"));
      } catch { /* use default */ }
    }

    const path = format === "onnx"
      ? (outputPath.endsWith(".onnx") ? outputPath : outputPath + ".onnx")
      : (outputPath.endsWith(".pt") ? outputPath : outputPath + ".pt");

    try {
      const method = format === "onnx" ? "export.onnx" : "export.torchscript";
      await sidecarCall(method, {
        graph: graphJson,
        input_shapes: { x: inputShape },
        output_path: path,
        method: tsMethod,
      });
      setResult(`Exported to ${path}`);
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
                <button key={f} onClick={() => { setFormat(f); setOutputPath(f === "onnx" ? "./model.onnx" : "./model.pt"); }}
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

          {/* Output path */}
          <div>
            <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Output Path</label>
            <input type="text" value={outputPath} onChange={(e) => setOutputPath(e.target.value)}
              className="w-full px-2.5 py-[6px] text-xs rounded-lg"
              style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
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
