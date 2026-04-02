import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { useReactFlow } from "@xyflow/react";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore } from "../store/training-store";
import { MODULE_REGISTRY, CATEGORY_COLORS } from "../lib/registry";
import { computeModelSummary, formatParamCount, formatFlops } from "../lib/model-summary";
import { formatCtorArgs, generateNodeSnippet } from "../lib/decompiler";
import TrainingPanel from "./TrainingPanel";
import { startTraining, stopTraining } from "../lib/training";
import type { ParamDef, DataOutputDef, ScanResult } from "../lib/types";
import { safePythonId, DEFAULT_DATA_OUTPUTS } from "../lib/types";
import type { ShapeError } from "../lib/shape-inference";
import { isDatasetNode, DATASET_METADATA, extractDatasetConfig, DATA_PRESETS, type DataDomain, type IntrospectionResult, type ProbeResult } from "../lib/dataset-metadata";
import { sidecarCall } from "../lib/sidecar";

type Tab = "properties" | "training";

export default function PropertyPanel() {
  const nodes = useGraphStore((s) => s.nodes);
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const [tab, setTab] = useState<Tab>("properties");
  const trainingStatus = useTrainingStore((s) => s.status);
  const prevSelectedRef = useRef<string | null>(null);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  // Auto-switch to properties when a NEW node is selected
  useEffect(() => {
    if (selectedNodeId && selectedNodeId !== prevSelectedRef.current) {
      setTab("properties");
    }
    prevSelectedRef.current = selectedNodeId;
  }, [selectedNodeId]);

  const activeTab = selectedNode ? tab : tab;

  return (
    <div
      className="flex flex-col h-full overflow-hidden"
      style={{
        background: "var(--bg-secondary)",
      }}
    >
      {/* Tab bar */}
      <div className="flex items-center h-9 flex-shrink-0" style={{ borderBottom: "1px solid var(--border)" }}>
        <TabButton label="Properties" active={activeTab === "properties"} onClick={() => setTab("properties")} />
        <TabButton label="Training" active={activeTab === "training"} onClick={() => setTab("training")} />
      </div>

      {/* Tab content */}
      {activeTab === "properties" ? (
        selectedNode ? (
          <NodeEditor node={selectedNode} />
        ) : (
          <EmptyState onGoToTraining={() => setTab("training")} />
        )
      ) : (
        <div className="flex flex-col flex-1 min-h-0">
          <TrainingPanel />
          {/* Train/Stop button */}
          <div className="px-4 py-3 flex-shrink-0" style={{ borderTop: "1px solid var(--border)" }}>
            <TrainingProgress />
            {trainingStatus === "running" ? (
              <button
                onClick={stopTraining}
                className="w-full py-3 rounded-lg text-sm font-semibold text-white"
                style={{ background: "var(--danger)" }}
              >
                Stop Training
              </button>
            ) : trainingStatus === "starting" ? (
              <button
                disabled
                className="w-full py-3 rounded-lg text-sm font-semibold text-white cursor-not-allowed"
                style={{ background: "var(--accent)", animation: "pulse-subtle 1.5s ease-in-out infinite" }}
              >
                Starting...
              </button>
            ) : (
              <button
                onClick={startTraining}
                className="w-full py-3 rounded-lg text-sm font-semibold text-white"
                style={{ background: "var(--accent)" }}
                onMouseEnter={(e) => { e.currentTarget.style.background = "var(--accent-hover)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "var(--accent)"; }}
              >
                Start Training
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex-1 h-full text-[11px] font-semibold uppercase tracking-wider relative"
      style={{
        color: active ? "var(--text-primary)" : "var(--text-muted)",
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.color = "var(--text-secondary)";
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.color = "var(--text-muted)";
      }}
    >
      {label}
      {active && (
        <span
          className="absolute bottom-0 left-1/2 -translate-x-1/2 h-[2px] rounded-full"
          style={{ width: "40%", background: "var(--accent)" }}
        />
      )}
    </button>
  );
}

function EmptyState({ onGoToTraining }: { onGoToTraining: () => void }) {
  const nodes = useGraphStore((s) => s.nodes);
  const summary = useMemo(() => computeModelSummary(nodes), [nodes]);

  if (nodes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full px-6 text-center">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-3" style={{ background: "var(--bg-tertiary)" }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
          </svg>
        </div>
        <div className="text-xs font-medium mb-1" style={{ color: "var(--text-secondary)" }}>No node selected</div>
        <div className="text-[11px] leading-relaxed" style={{ color: "var(--text-muted)" }}>Click a node to edit its parameters</div>
      </div>
    );
  }

  // Show model summary when nodes exist but none selected
  const layerCounts: Record<string, number> = {};
  for (const n of nodes) {
    const mt = n.data.moduleType;
    if (mt !== "Input" && mt !== "Output") layerCounts[mt] = (layerCounts[mt] ?? 0) + 1;
  }
  const sortedLayers = Object.entries(layerCounts).sort((a, b) => b[1] - a[1]);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Summary header */}
      <div className="px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="text-[10px] font-semibold uppercase tracking-wider mb-2.5" style={{ color: "var(--text-muted)" }}>
          Model Summary
        </div>
        <div className="space-y-2">
          <SummaryRow label="Parameters" value={formatParamCount(summary.totalParams)} />
          <SummaryRow label="Model Size (FP32)" value={`${summary.estimatedSizeMB.toFixed(1)} MB`} />
          {summary.estimatedFlops > 0 && <SummaryRow label="FLOPs" value={formatFlops(summary.estimatedFlops)} />}
          <SummaryRow label="Est. VRAM" value={`~${summary.estimatedVramMB.toFixed(0)} MB`} />
          {/* VRAM bar */}
          <div className="w-full h-[2px] rounded-full" style={{ background: "var(--bg-tertiary)" }}>
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(100, summary.estimatedVramMB / 8192 * 100)}%`,
                background: summary.estimatedVramMB > 6144 ? "var(--danger)" : summary.estimatedVramMB > 3072 ? "var(--warning)" : "var(--text-muted)",
                opacity: 0.5,
              }}
            />
          </div>
          <SummaryRow label="Layers" value={String(nodes.length - nodes.filter((n) => n.data.moduleType === "Input" || n.data.moduleType === "Output").length)} />
        </div>
      </div>

      {/* Layer breakdown */}
      <div className="px-4 py-3 flex-1">
        <div className="text-[10px] font-semibold uppercase tracking-wider mb-2.5" style={{ color: "var(--text-muted)" }}>
          Layer Breakdown
        </div>
        <div className="space-y-1.5">
          {sortedLayers.map(([type, count]) => {
            const def = MODULE_REGISTRY[type];
            const catColor = def ? CATEGORY_COLORS[def.category] : "var(--text-muted)";
            return (
              <div key={type} className="flex items-center justify-between text-[11px]">
                <span className="flex items-center gap-1.5" style={{ color: "var(--text-primary)" }}>
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0 opacity-70" style={{ background: catColor }} />
                  {type}
                </span>
                <span className="tabular-nums" style={{ color: "var(--text-muted)" }}>{count}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Action */}
      <div className="px-4 pb-4">
        <button
          onClick={onGoToTraining}
          className="w-full py-2 text-xs font-medium rounded-lg transition-colors"
          style={{ color: "var(--accent)", background: "var(--accent-subtle)" }}
        >
          Configure Training
        </button>
      </div>
    </div>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-[11px]">
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{value}</span>
    </div>
  );
}

function TrainingProgress() {
  const status = useTrainingStore((s) => s.status);
  const currentEpoch = useTrainingStore((s) => s.currentEpoch);
  const epochs = useTrainingStore((s) => s.config.epochs);
  const metrics = useTrainingStore((s) => s.metrics);

  if (status !== "running") return null;

  const lastLoss = metrics.length > 0 ? metrics[metrics.length - 1].trainLoss : null;
  const progress = epochs > 0 ? (currentEpoch / epochs) * 100 : 0;

  return (
    <div className="mb-3 space-y-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span style={{ color: "var(--text-secondary)" }}>Epoch {currentEpoch} / {epochs}</span>
        {lastLoss !== null && (
          <span className="font-mono tabular-nums text-[11px]" style={{ color: "var(--text-primary)" }}>
            loss: {lastLoss.toFixed(4)}
          </span>
        )}
      </div>
      <div className="w-full h-1 rounded-full" style={{ background: "var(--bg-tertiary)" }}>
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(100, progress)}%`, background: "var(--accent)" }}
        />
      </div>
    </div>
  );
}

// --- Error Banner ---
function ErrorBanner({ error, nodeId }: { error: ShapeError; nodeId: string }) {
  const nodes = useGraphStore((s) => s.nodes);
  const shapeMap = useGraphStore((s) => s.shapeMap);
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam);
  const setSelectedNodeId = useGraphStore((s) => s.setSelectedNodeId);
  const reactFlow = useReactFlow();

  const isRoot = error.type === "root";
  const bgColor = isRoot ? "var(--danger-subtle)" : "var(--warning-subtle)";
  const borderColor = isRoot ? "var(--danger)" : "var(--warning)";
  const textColor = isRoot ? "var(--danger)" : "var(--warning)";

  const sourceLabel = !isRoot
    ? nodes.find((n) => n.id === error.sourceNodeId)?.data.label ?? error.sourceNodeId
    : "";

  const handleAutoFix = useCallback(() => {
    if (!error.paramHint || !error.receivedShape) return;
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    // Determine the correct value based on paramHint
    let correctValue: number | undefined;
    const incoming = useGraphStore.getState().edges.filter((e) => e.target === nodeId);
    const sourceId = incoming[0]?.source;
    const upstreamShape = sourceId ? shapeMap[sourceId]?.shape : undefined;

    if (!upstreamShape) return;

    switch (error.paramHint) {
      case "in_features":
        correctValue = upstreamShape[upstreamShape.length - 1];
        break;
      case "in_channels":
        correctValue = upstreamShape.length >= 2 ? upstreamShape[1] : undefined;
        break;
      case "num_features":
        correctValue = upstreamShape.length >= 2 ? upstreamShape[1] : undefined;
        break;
      case "input_size":
        correctValue = upstreamShape[upstreamShape.length - 1];
        break;
      case "embed_dim":
        correctValue = upstreamShape[upstreamShape.length - 1];
        break;
      case "d_model":
        correctValue = upstreamShape[upstreamShape.length - 1];
        break;
      case "num_channels":
        correctValue = upstreamShape.length >= 2 ? upstreamShape[1] : undefined;
        break;
    }

    if (correctValue !== undefined) {
      updateNodeParam(nodeId, error.paramHint, correctValue);
    }
  }, [error, nodeId, nodes, shapeMap, updateNodeParam]);

  const handleGoToSource = useCallback(() => {
    setSelectedNodeId(error.sourceNodeId);
    const sourceNode = nodes.find((n) => n.id === error.sourceNodeId);
    if (sourceNode) {
      reactFlow.fitView({ nodes: [sourceNode], duration: 300, padding: 0.5 });
    }
  }, [error.sourceNodeId, setSelectedNodeId, nodes, reactFlow]);

  return (
    <div
      className="mx-4 mt-3 px-3 py-2 rounded-lg text-[11px] leading-relaxed"
      style={{ background: bgColor, border: `1px solid ${borderColor}`, color: textColor }}
    >
      {isRoot ? (
        <>
          <div className="font-semibold mb-1">{error.message}</div>
          {error.receivedShape && error.expectedShape && (
            <div className="font-mono text-[10px] mb-1.5" style={{ opacity: 0.8 }}>
              Got {JSON.stringify(error.receivedShape)} {"\u2192"} Expected dim: {error.paramHint}={error.expectedShape[error.expectedShape.length - 1] ?? "?"}
            </div>
          )}
          {error.paramHint && (
            <button
              onClick={handleAutoFix}
              className="mt-1 px-2.5 py-1 rounded text-[10px] font-semibold text-white transition-colors"
              style={{ background: "var(--danger)" }}
            >
              Auto-fix {error.paramHint}
            </button>
          )}
        </>
      ) : (
        <>
          <div className="font-semibold mb-1">Error from: {sourceLabel}</div>
          <div className="text-[10px] mb-1.5" style={{ opacity: 0.8 }}>Fix the source node first to resolve this error</div>
          <button
            onClick={handleGoToSource}
            className="mt-1 px-2.5 py-1 rounded text-[10px] font-semibold text-white transition-colors"
            style={{ background: "var(--warning)" }}
          >
            Go to source
          </button>
        </>
      )}
    </div>
  );
}

// --- Per-Node Code Editor ---
function CodeEditor({ node }: { node: ReturnType<typeof useGraphStore.getState>["nodes"][number] }) {
  const edges = useGraphStore((s) => s.edges);
  const nodes = useGraphStore((s) => s.nodes);
  const shapeMap = useGraphStore((s) => s.shapeMap);
  const [expanded, setExpanded] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const [copied, setCopied] = useState(false);
  const setCustomCode = useGraphStore((s) => s.setCustomCode);
  const mt = node.data.moduleType;
  const hidden = mt === "Input" || mt === "Output";
  const isDataset = isDatasetNode(mt);

  // Generate snippet for non-dataset nodes
  const snippet = useMemo(() => {
    if (hidden || isDataset) return null;
    return generateNodeSnippet(node, nodes, edges, shapeMap);
  }, [hidden, isDataset, node, nodes, edges, shapeMap]);

  // Dataset default code (existing behavior)
  const datasetCode = useMemo(() => {
    if (!isDataset) return "";
    const meta = DATASET_METADATA[mt];
    const root = node.data.params.root ?? "./data";
    if (meta) {
      return `# ${meta.description}\n# Output shape: ${JSON.stringify(meta.inputShape)}, classes: ${meta.numClasses}\ntorchvision.datasets.${mt}(root='${root}', download=True)`;
    }
    return `# Custom dataset\nload_dataset('${root}')`;
  }, [isDataset, mt, node.data.params]);

  const hasInit = !!snippet?.initCode;

  // Compose display code
  const displayCode = useMemo(() => {
    if (isDataset) return node.data.customCode ?? datasetCode;
    if (!snippet) return "";
    const { initCode, forwardCode } = snippet;
    if (initCode) {
      return `# __init__\n${initCode}\n\n${forwardCode}`;
    }
    return forwardCode;
  }, [isDataset, datasetCode, snippet, node.data.customCode]);

  const isCustom = !!node.data.customCode;

  // Default init RHS for comparison when saving
  const defaultInitRhs = useMemo(() => {
    if (isDataset || !hasInit) return "";
    return `nn.${mt}(${formatCtorArgs(mt, node.data.params)})`;
  }, [isDataset, hasInit, mt, node.data.params]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(displayCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [displayCode]);

  const handleStartEdit = useCallback(() => {
    if (isDataset) {
      setEditValue(node.data.customCode ?? datasetCode);
    } else if (snippet) {
      setEditValue(snippet.initCode);
    }
    setEditing(true);
  }, [isDataset, datasetCode, snippet, node.data.customCode]);

  const handleSave = useCallback(() => {
    const trimmed = editValue.trim();
    if (isDataset) {
      if (trimmed && trimmed !== datasetCode) {
        setCustomCode(node.id, trimmed);
      } else {
        setCustomCode(node.id, undefined);
      }
    } else {
      // Extract RHS from "self.X = ..." and store as customCode
      const rhsMatch = trimmed.match(/^self\.\w+\s*=\s*([\s\S]*)$/);
      const rhs = rhsMatch ? rhsMatch[1].trim() : trimmed;
      if (rhs && rhs !== defaultInitRhs) {
        setCustomCode(node.id, rhs);
      } else {
        setCustomCode(node.id, undefined);
      }
    }
    setEditing(false);
  }, [editValue, isDataset, datasetCode, defaultInitRhs, node.id, setCustomCode]);

  const handleReset = useCallback(() => {
    setCustomCode(node.id, undefined);
    setEditing(false);
  }, [node.id, setCustomCode]);

  if (hidden) return null;

  // Can edit init for non-dataset nodes with init code, or any dataset node
  const canEdit = isDataset || hasInit;

  return (
    <div style={{ borderTop: "1px solid var(--border)" }}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2 text-[10px] font-semibold uppercase tracking-wider"
        style={{ color: "var(--text-muted)" }}
      >
        <span className="flex items-center gap-1.5">
          Code
          {isCustom && (
            <span className="text-[8px] px-1 py-0.5 rounded normal-case tracking-normal font-medium"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}>edited</span>
          )}
        </span>
        <svg
          width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.15s" }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>
      {expanded && (
        <div className="px-4 pb-3">
          {editing ? (
            <div className="space-y-2">
              {!isDataset && (
                <div className="text-[9px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>__init__</div>
              )}
              <textarea
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                rows={isDataset ? 3 : 1}
                className="w-full rounded-lg px-3 py-2 text-[10px] font-mono leading-relaxed resize-y"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--accent)" }}
                spellCheck={false}
              />
              {/* Show forward line as grayed-out read-only when editing a non-dataset node */}
              {!isDataset && snippet && (
                <div>
                  <div className="text-[9px] font-medium mt-1 mb-1" style={{ color: "var(--text-muted)" }}>forward (from graph)</div>
                  <div
                    className="rounded-lg px-3 py-2 text-[10px] font-mono leading-relaxed"
                    style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)", border: "1px solid var(--border)", opacity: 0.6 }}
                  >
                    <code style={{ whiteSpace: "pre-wrap" }}>{snippet.forwardCode}</code>
                  </div>
                </div>
              )}
              <div className="flex gap-1.5">
                <button
                  onClick={handleSave}
                  className="px-2.5 py-1 rounded text-[10px] font-medium text-white"
                  style={{ background: "var(--accent)" }}
                >Save</button>
                <button
                  onClick={() => setEditing(false)}
                  className="px-2.5 py-1 rounded text-[10px] font-medium"
                  style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
                >Cancel</button>
                {isCustom && (
                  <button
                    onClick={handleReset}
                    className="px-2.5 py-1 rounded text-[10px] font-medium ml-auto"
                    style={{ color: "var(--danger)" }}
                  >Reset to Default</button>
                )}
              </div>
            </div>
          ) : (
            <div
              className="relative rounded-lg px-3 py-2 text-[10px] font-mono leading-relaxed overflow-x-auto"
              style={{
                background: "var(--bg-tertiary)",
                color: "var(--text-primary)",
                border: isCustom ? "1px solid var(--accent)" : "1px solid var(--border)",
              }}
            >
              <code style={{ whiteSpace: "pre-wrap" }}>{displayCode}</code>
              <div className="absolute top-1 right-1 flex gap-1">
                {canEdit && (
                  <button
                    onClick={handleStartEdit}
                    className="px-1.5 py-0.5 rounded text-[9px] transition-colors"
                    style={{ background: "var(--bg-elevated)", color: "var(--text-muted)", border: "1px solid var(--border)" }}
                    title="Edit init code"
                  >Edit</button>
                )}
                <button
                  onClick={handleCopy}
                  className="px-1.5 py-0.5 rounded text-[9px] transition-colors"
                  style={{
                    background: copied ? "var(--accent)" : "var(--bg-elevated)",
                    color: copied ? "#fff" : "var(--text-muted)",
                    border: "1px solid var(--border)",
                  }}
                  title="Copy to clipboard"
                >{copied ? "Copied" : "Copy"}</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DataPreview({ node }: { node: ReturnType<typeof useGraphStore.getState>["nodes"][number] }) {
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<{
    train_size: number;
    val_size: number;
    sample_shape: number[];
    sample_dtype: string;
    num_classes: number;
    feature_stats: { mean: number; std: number; min: number; max: number };
    error?: string;
  } | null>(null);

  const handlePreview = useCallback(async () => {
    setLoading(true);
    setPreview(null);
    try {
      const { dataset, valSplit, datasetExtra } = extractDatasetConfig(
        node.data.moduleType,
        node.data.params,
      );
      const config = { ...dataset, val_split: valSplit, ...datasetExtra };
      const result = await sidecarCall("data.preview", { config, batch_size: 32 });
      setPreview(result as typeof preview);
    } catch (e) {
      setPreview({ train_size: 0, val_size: 0, sample_shape: [], sample_dtype: "", num_classes: 0, feature_stats: { mean: 0, std: 0, min: 0, max: 0 }, error: e instanceof Error ? e.message : "Preview failed" });
    }
    setLoading(false);
  }, [node.data.moduleType, node.data.params]);

  if (!isDatasetNode(node.data.moduleType)) return null;
  // Don't show preview for Data nodes with no source selected
  if (node.data.moduleType === "Data" && !node.data.params.source) return null;

  return (
    <div style={{ borderTop: "1px solid var(--border)" }}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2 text-[10px] font-semibold uppercase tracking-wider"
        style={{ color: "var(--text-muted)" }}
      >
        <span>Data Preview</span>
        <svg
          width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.15s" }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>
      {expanded && (
        <div className="px-4 pb-3">
          <button
            onClick={handlePreview}
            disabled={loading}
            className="w-full py-1.5 rounded-lg text-[11px] font-medium transition-colors mb-2"
            style={{
              background: "var(--accent-subtle)",
              color: "var(--accent)",
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Loading..." : "Preview Data"}
          </button>
          {preview && (
            preview.error ? (
              <div className="text-[11px] px-2 py-1.5 rounded-lg" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
                {preview.error}
              </div>
            ) : (
              <div className="space-y-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>
                <div className="flex justify-between">
                  <span>Train samples</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{preview.train_size.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Val samples</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{preview.val_size.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Sample shape</span>
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-primary)" }}>{JSON.stringify(preview.sample_shape)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Classes</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{preview.num_classes}</span>
                </div>
                <div className="flex justify-between">
                  <span>Dtype</span>
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-primary)" }}>{preview.sample_dtype}</span>
                </div>
                <div className="mt-1.5 pt-1.5" style={{ borderTop: "1px solid var(--border)" }}>
                  <div className="text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Stats</div>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                    <div className="flex justify-between"><span>Mean</span><span className="tabular-nums">{preview.feature_stats.mean.toFixed(4)}</span></div>
                    <div className="flex justify-between"><span>Std</span><span className="tabular-nums">{preview.feature_stats.std.toFixed(4)}</span></div>
                    <div className="flex justify-between"><span>Min</span><span className="tabular-nums">{preview.feature_stats.min.toFixed(4)}</span></div>
                    <div className="flex justify-between"><span>Max</span><span className="tabular-nums">{preview.feature_stats.max.toFixed(4)}</span></div>
                  </div>
                </div>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// DataNodeEditor — custom editor for the universal Data node
// ---------------------------------------------------------------------------

const PRESET_SOURCES = ["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "TinyShakespeare", "WikiText2", "IMDB", "AGNews", "SpeechCommands"];
const CUSTOM_SOURCES = ["TensorFile"];
const CUSTOM_SOURCE_LABELS: Record<string, string> = { TensorFile: "Custom Tensors" };

function DataNodeEditor({ node }: { node: ReturnType<typeof useGraphStore.getState>["nodes"][number] }) {
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam);
  const deleteSelected = useGraphStore((s) => s.deleteSelected);
  const setDataOutputs = useGraphStore((s) => s.setDataOutputs);
  const updateDataOutput = useGraphStore((s) => s.updateDataOutput);
  const shapeError: ShapeError | undefined = useGraphStore((s) => s.shapeErrors[node.id]);
  const color = CATEGORY_COLORS[node.data.category];
  const p = node.data.params;

  const source = String(p.source ?? "");
  const preset = source ? DATA_PRESETS[source] : undefined;
  const isPreset = !!preset;
  const domain = isPreset ? preset.domain : String(p.domain ?? "auto") as DataDomain;
  const introspected = p._introspected as IntrospectionResult | undefined;
  const probed = p._probed as ProbeResult | undefined;

  const [introspecting, setIntrospecting] = useState(false);
  const [introspectError, setIntrospectError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const scanResult = p._scan as ScanResult | undefined;

  // Apply preset defaults when source changes
  const handleSourceChange = useCallback((newSource: string) => {
    updateNodeParam(node.id, "source", newSource);
    const newPreset = newSource ? DATA_PRESETS[newSource] : undefined;
    if (newPreset) {
      updateNodeParam(node.id, "domain", newPreset.domain);
      // Apply preset defaults
      for (const [k, v] of Object.entries(newPreset.defaults)) {
        updateNodeParam(node.id, k, v);
      }
    } else {
      updateNodeParam(node.id, "domain", "auto");
    }
    // Clear all cached data + reset outputs on source change
    updateNodeParam(node.id, "_introspected", undefined);
    updateNodeParam(node.id, "_probed", undefined);
    updateNodeParam(node.id, "_scan", undefined);
    setDataOutputs(node.id, [...DEFAULT_DATA_OUTPUTS]);
  }, [node.id, updateNodeParam]);

  const handleIntrospect = useCallback(async () => {
    setIntrospecting(true);
    setIntrospectError(null);
    try {
      const result = await sidecarCall<IntrospectionResult>("data.introspect", {
        source,
        dataset_name: p.dataset_name ?? "",
        subset: p.subset ?? "",
        path: p.path ?? "",
      });
      updateNodeParam(node.id, "_introspected", result);
      if (result.detected_domain && result.detected_domain !== "auto") {
        updateNodeParam(node.id, "domain", result.detected_domain);
      }
    } catch (e) {
      setIntrospectError(e instanceof Error ? e.message : "Introspection failed");
    }
    setIntrospecting(false);
  }, [source, p.dataset_name, p.subset, p.path, node.id, updateNodeParam]);

  const handleProbe = useCallback(async () => {
    setProbing(true);
    setProbeError(null);
    try {
      const { dataset, valSplit, datasetExtra } = extractDatasetConfig(
        node.data.moduleType,
        node.data.params,
      );
      const config = { ...dataset, val_split: valSplit, ...datasetExtra };
      const result = await sidecarCall<ProbeResult>("data.probe", { config });
      if ((result as unknown as { error?: string }).error) {
        setProbeError((result as unknown as { error: string }).error);
      } else {
        updateNodeParam(node.id, "_probed", result);
        // Auto-sync connected Input nodes
        const { edges, nodes } = useGraphStore.getState();
        const connectedInputIds = edges
          .filter((e) => e.source === node.id)
          .map((e) => e.target)
          .filter((targetId) => {
            const targetNode = nodes.find((n) => n.id === targetId);
            return targetNode?.data.moduleType === "Input";
          });
        for (const inputId of connectedInputIds) {
          updateNodeParam(inputId, "shape", JSON.stringify([1, ...result.sample_shape]));
        }
      }
    } catch (e) {
      setProbeError(e instanceof Error ? e.message : "Probe failed");
    }
    setProbing(false);
  }, [node.data.moduleType, node.data.params, node.id, updateNodeParam]);

  const handleScan = useCallback(async (pathOverride?: string) => {
    const path = pathOverride ?? String(p.path ?? "");
    if (!path) return;
    setScanning(true);
    setScanError(null);
    try {
      const result = await sidecarCall<ScanResult>("data.scan", { path });
      updateNodeParam(node.id, "_scan", result);
      // Auto-map data outputs to discovered tensors
      if (result.tensors && result.tensors.length > 0) {
        const currentOutputs: DataOutputDef[] = node.data.dataOutputs ?? [...DEFAULT_DATA_OUTPUTS];
        const tensorList = result.tensors;
        // Map each existing handle to a discovered tensor by position (x→first, y→second, ...)
        const updated = currentOutputs.map((out, i) => {
          const mapped = tensorList[i];
          if (!mapped) return out;
          return {
            ...out,
            column: mapped.name,
            dtype: mapped.dtype.replace("float64", "float32").replace("int64", "long"),
            shape: mapped.shape.slice(1),
          };
        });
        setDataOutputs(node.id, updated);
      }
    } catch (e) {
      setScanError(e instanceof Error ? e.message : "Scan failed");
    }
    setScanning(false);
  }, [p.path, node.id, updateNodeParam, setDataOutputs]);

  const columns = introspected?.columns ?? [];
  const inputCls = "w-full px-2.5 py-2 text-xs rounded-md";
  const inputStyle = { background: "transparent", border: "1px solid var(--border)", color: "var(--text-primary)" };

  return (
    <>
      {/* Header */}
      <div className="px-4 py-3 flex items-start gap-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-[11px] font-bold flex-shrink-0 mt-0.5" style={{ background: color }}>
          DA
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>{node.data.label}</div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
            {!source ? "Select a data source" : isPreset ? preset.description : `Custom: ${source}`}
          </div>
        </div>
      </div>

      {shapeError && <ErrorBanner error={shapeError} nodeId={node.id} />}

      <div className="flex-1 overflow-y-auto">
        {/* Source selector */}
        <div className="px-4 pt-3 pb-3">
          <select
            value={source}
            onChange={(e) => handleSourceChange(e.target.value)}
            className={inputCls}
            style={inputStyle}
          >
            <option value="" disabled={false}>Select a source...</option>
            <optgroup label="Presets">
              {PRESET_SOURCES.map((s) => (
                <option key={s} value={s}>{DATA_PRESETS[s]?.label ?? s}</option>
              ))}
            </optgroup>
            <optgroup label="Custom">
              {CUSTOM_SOURCES.map((s) => (
                <option key={s} value={s}>
                  {CUSTOM_SOURCE_LABELS[s] ?? s}
                </option>
              ))}
            </optgroup>
          </select>
          {isPreset && (
            <div className="mt-1.5 text-[10px]" style={{ color: "var(--text-muted)" }}>
              {preset.sizeHint} {preset.requiresPip && `· requires pip install ${preset.requiresPip}`}
            </div>
          )}
        </div>

        {/* Nothing else to show when no source selected */}
        {!source && (
          <div className="px-4 pb-4 text-[11px] italic" style={{ color: "var(--text-muted)" }}>
            Choose a preset or custom source to configure the data pipeline.
          </div>
        )}

        {/* Source-specific fields */}
        {source === "TensorFile" && (
          <TensorFilePathInput
            path={String(p.path ?? "")}
            scanning={scanning}
            scanResult={scanResult}
            scanError={scanError}
            onPathChange={(newPath) => {
              updateNodeParam(node.id, "path", newPath);
              updateNodeParam(node.id, "_probed", undefined);
            }}
            onScan={(path) => handleScan(path)}
            inputStyle={inputStyle}
          />
        )}

        {/* Introspect button for custom sources (non-TensorFile) */}
        {source && !isPreset && source !== "TensorFile" && (
          <div className="px-4 pb-3">
            <button
              onClick={handleIntrospect}
              disabled={introspecting}
              className="w-full py-1.5 rounded-lg text-[11px] font-medium transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", opacity: introspecting ? 0.6 : 1 }}
            >
              {introspecting ? "Inspecting..." : "Inspect Data"}
            </button>
            {introspectError && (
              <div className="mt-1.5 text-[10px] px-2 py-1 rounded" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
                {introspectError}
              </div>
            )}
          </div>
        )}

        {/* Introspection results */}
        {introspected && (
          <div className="px-4 pb-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[10px] font-semibold uppercase tracking-wider mt-2 mb-2" style={{ color: "var(--text-muted)" }}>
              Introspection
            </div>
            <div className="space-y-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>
              {introspected.columns.length > 0 && (
                <div className="flex justify-between">
                  <span>Columns</span>
                  <span className="font-mono text-[10px] truncate ml-2" style={{ color: "var(--text-primary)", maxWidth: "140px" }}>
                    {introspected.columns.join(", ")}
                  </span>
                </div>
              )}
              {introspected.num_classes > 0 && (
                <div className="flex justify-between">
                  <span>Classes</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{introspected.num_classes}</span>
                </div>
              )}
              {introspected.label_values.length > 0 && introspected.label_values.length <= 10 && (
                <div className="flex justify-between">
                  <span>Labels</span>
                  <span className="font-mono text-[10px] truncate ml-2" style={{ color: "var(--text-primary)", maxWidth: "140px" }}>
                    {introspected.label_values.join(", ")}
                  </span>
                </div>
              )}
              {introspected.sample_shape.length > 0 && (
                <div className="flex justify-between">
                  <span>Sample shape</span>
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-primary)" }}>{JSON.stringify(introspected.sample_shape)}</span>
                </div>
              )}
              {introspected.total_samples > 0 && (
                <div className="flex justify-between">
                  <span>Total samples</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{introspected.total_samples.toLocaleString()}</span>
                </div>
              )}
              {introspected.splits.length > 0 && (
                <div className="flex justify-between">
                  <span>Splits</span>
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-primary)" }}>{introspected.splits.join(", ")}</span>
                </div>
              )}
            </div>
            {/* Preprocessing warning for complex datasets */}
            {(() => {
              const ir = introspected as unknown as Record<string, unknown>;
              if (!ir.needs_preprocessing) return null;
              const complexCols = (ir.complex_columns as string[]) || [];
              const example = ir.example_preprocess ? String(ir.example_preprocess) : "";
              return (
                <div className="mt-2 px-2.5 py-2 rounded-lg text-[10px] leading-relaxed"
                  style={{ background: "var(--warning-subtle)", border: "1px solid var(--warning)", color: "var(--warning)" }}>
                  <div className="font-semibold mb-1">Preprocessing required</div>
                  <div className="mb-1.5" style={{ opacity: 0.85 }}>
                    This dataset has nested/complex columns ({complexCols.join(", ")}).
                    Add a preprocessing script below to extract features and labels.
                  </div>
                  {example && (
                    <pre className="font-mono text-[9px] p-1.5 rounded overflow-x-auto"
                      style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                      {example}
                    </pre>
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {/* Domain override — not shown for TensorFile (raw tensors have no domain) */}
        {source && !isPreset && source !== "TensorFile" && (
          <div className="px-4 pb-3">
            <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Domain</label>
            <select
              value={String(p.domain ?? "auto")}
              onChange={(e) => updateNodeParam(node.id, "domain", e.target.value)}
              className={inputCls}
              style={inputStyle}
            >
              <option value="auto">Auto-detect</option>
              <option value="vision">Vision</option>
              <option value="text">Text</option>
              <option value="tabular">Tabular</option>
              <option value="audio">Audio</option>
            </select>
          </div>
        )}

        {/* Domain-specific params */}
        {source && (domain === "vision" || (isPreset && preset.domain === "vision")) && (
          <div className="px-4 pb-3 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[10px] font-semibold uppercase tracking-wider mt-2" style={{ color: "var(--text-muted)" }}>Vision</div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Image Size</label>
              <input type="number" value={p.image_size == null ? "" : String(p.image_size)} min={8} max={1024}
                onChange={(e) => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) updateNodeParam(node.id, "image_size", v); }}
                className={inputCls} style={inputStyle} />
            </div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Channels</label>
              <select value={String(p.channels ?? "rgb")} onChange={(e) => updateNodeParam(node.id, "channels", e.target.value)}
                className={inputCls} style={inputStyle}>
                <option value="rgb">RGB</option>
                <option value="grayscale">Grayscale</option>
              </select>
            </div>
          </div>
        )}

        {source && (domain === "text" || (isPreset && preset.domain === "text")) && (
          <div className="px-4 pb-3 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[10px] font-semibold uppercase tracking-wider mt-2" style={{ color: "var(--text-muted)" }}>Text</div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Max Length</label>
              <input type="number" value={p.max_length == null ? "" : String(p.max_length)} min={8} max={2048}
                onChange={(e) => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) updateNodeParam(node.id, "max_length", v); }}
                className={inputCls} style={inputStyle} />
            </div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Vocab Size</label>
              <input type="number" value={p.vocab_size == null ? "" : String(p.vocab_size)} min={100} max={100000}
                onChange={(e) => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) updateNodeParam(node.id, "vocab_size", v); }}
                className={inputCls} style={inputStyle} />
            </div>
            {/* Column dropdowns from introspection */}
            {!isPreset && (
              <>
                <div>
                  <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Text Column</label>
                  {columns.length > 0 ? (
                    <select value={String(p.text_column ?? "")} onChange={(e) => updateNodeParam(node.id, "text_column", e.target.value)}
                      className={inputCls} style={inputStyle}>
                      <option value="">Auto-detect</option>
                      {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  ) : (
                    <input type="text" value={String(p.text_column ?? "")} onChange={(e) => updateNodeParam(node.id, "text_column", e.target.value)}
                      placeholder="auto-detect" className={inputCls} style={inputStyle} />
                  )}
                </div>
                <div>
                  <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Label Column</label>
                  {columns.length > 0 ? (
                    <select value={String(p.label_column ?? "")} onChange={(e) => updateNodeParam(node.id, "label_column", e.target.value)}
                      className={inputCls} style={inputStyle}>
                      <option value="">Auto-detect</option>
                      {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  ) : (
                    <input type="text" value={String(p.label_column ?? "")} onChange={(e) => updateNodeParam(node.id, "label_column", e.target.value)}
                      placeholder="auto-detect" className={inputCls} style={inputStyle} />
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {source && (domain === "tabular") && !isPreset && (
          <div className="px-4 pb-3 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[10px] font-semibold uppercase tracking-wider mt-2" style={{ color: "var(--text-muted)" }}>Tabular</div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Target Column</label>
              {columns.length > 0 ? (
                <select value={String(p.target_column ?? "")} onChange={(e) => updateNodeParam(node.id, "target_column", e.target.value)}
                  className={inputCls} style={inputStyle}>
                  <option value="">Auto (last column)</option>
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              ) : (
                <input type="text" value={String(p.target_column ?? "")} onChange={(e) => updateNodeParam(node.id, "target_column", e.target.value)}
                  placeholder="auto (last column)" className={inputCls} style={inputStyle} />
              )}
            </div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Feature Columns</label>
              <input type="text" value={String(p.feature_columns ?? "")} onChange={(e) => updateNodeParam(node.id, "feature_columns", e.target.value)}
                placeholder="comma-separated or empty=all" className={inputCls} style={inputStyle} />
            </div>
            {source === "CSV" && (
              <>
                <div>
                  <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Delimiter</label>
                  <select value={String(p.delimiter ?? ",")} onChange={(e) => updateNodeParam(node.id, "delimiter", e.target.value)}
                    className={inputCls} style={inputStyle}>
                    <option value=",">Comma (,)</option>
                    <option value=";">Semicolon (;)</option>
                    <option value="|">Pipe (|)</option>
                    <option value="\t">Tab</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Has Header</label>
                  <label className="flex items-center gap-2.5 cursor-pointer rounded-lg px-2.5 py-[6px]" style={{ background: "var(--bg-tertiary)" }}>
                    <input type="checkbox" checked={p.has_header !== false} onChange={(e) => updateNodeParam(node.id, "has_header", e.target.checked)} className="accent-[var(--accent)]" />
                    <span className="text-xs" style={{ color: "var(--text-primary)" }}>{p.has_header !== false ? "Yes" : "No"}</span>
                  </label>
                </div>
              </>
            )}
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Normalization</label>
              <select value={String(p.normalization ?? "none")} onChange={(e) => updateNodeParam(node.id, "normalization", e.target.value)}
                className={inputCls} style={inputStyle}>
                <option value="none">None</option>
                <option value="standard">Standard (z-score)</option>
                <option value="minmax">Min-Max</option>
              </select>
            </div>
          </div>
        )}

        {/* HuggingFace vision: image column */}
        {source === "HuggingFace" && domain === "vision" && !isPreset && (
          <div className="px-4 pb-3">
            <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Image Column</label>
            {columns.length > 0 ? (
              <select value={String(p.image_column ?? "")} onChange={(e) => updateNodeParam(node.id, "image_column", e.target.value)}
                className={inputCls} style={inputStyle}>
                <option value="">Auto-detect</option>
                {columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : (
              <input type="text" value={String(p.image_column ?? "")} onChange={(e) => updateNodeParam(node.id, "image_column", e.target.value)}
                placeholder="auto-detect" className={inputCls} style={inputStyle} />
            )}
          </div>
        )}


        {/* Augmentation — vision only */}
        {source && (domain === "vision" || (isPreset && preset.domain === "vision")) && (
          <div className="px-4 pb-3 space-y-2" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[10px] font-semibold uppercase tracking-wider mt-2 mb-1" style={{ color: "var(--text-muted)" }}>Augmentation</div>
            <label className="flex items-center gap-2.5 cursor-pointer text-xs" style={{ color: "var(--text-primary)" }}>
              <input type="checkbox" checked={!!p.horizontal_flip} onChange={(e) => updateNodeParam(node.id, "horizontal_flip", e.target.checked)} className="accent-[var(--accent)]" />
              Random H-Flip
            </label>
            <label className="flex items-center gap-2.5 cursor-pointer text-xs" style={{ color: "var(--text-primary)" }}>
              <input type="checkbox" checked={!!p.random_crop} onChange={(e) => updateNodeParam(node.id, "random_crop", e.target.checked)} className="accent-[var(--accent)]" />
              Random Crop
            </label>
            <label className="flex items-center gap-2.5 cursor-pointer text-xs" style={{ color: "var(--text-primary)" }}>
              <input type="checkbox" checked={!!p.color_jitter} onChange={(e) => updateNodeParam(node.id, "color_jitter", e.target.checked)} className="accent-[var(--accent)]" />
              Color Jitter
            </label>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Rotation (deg)</label>
              <input type="number" value={p.rotation == null ? "0" : String(p.rotation)} min={0} max={180}
                onChange={(e) => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) updateNodeParam(node.id, "rotation", v); }}
                className={inputCls} style={inputStyle} />
            </div>
          </div>
        )}

        {/* Val split — shown when source is selected */}
        {source && (
          <div className="px-4 pb-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="mt-2">
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Val Split</label>
              <input type="number" value={p.val_split == null ? "0.2" : String(p.val_split)} step="any" min={0} max={0.5}
                onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateNodeParam(node.id, "val_split", v); }}
                className={inputCls} style={inputStyle} />
            </div>
          </div>
        )}
      </div>

      {/* Data Outputs — configurable output handles */}
      {source && (
        <DataOutputsEditor node={node} source={source} updateDataOutput={updateDataOutput} />
      )}

      {/* Preprocessing script — custom sources (not TensorFile — raw tensors need no preprocessing) */}
      {source && !isPreset && source !== "TensorFile" && <PreprocessEditor nodeId={node.id} value={String(p.preprocess ?? "")} onChange={(v) => { updateNodeParam(node.id, "preprocess", v); updateNodeParam(node.id, "_probed", undefined); }} />}

      {/* Probe output shape — custom sources only, not TensorFile (scan replaces probe) */}
      {source && !isPreset && source !== "TensorFile" && (
        <div style={{ borderTop: "1px solid var(--border)" }}>
          <div className="px-4 py-3">
            <button
              onClick={handleProbe}
              disabled={probing}
              className="w-full py-1.5 rounded-lg text-[11px] font-medium transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", opacity: probing ? 0.6 : 1 }}
            >
              {probing ? "Probing..." : "Probe Output Shape"}
            </button>
            {probeError && (
              <div className="mt-1.5 text-[10px] px-2 py-1 rounded" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
                {probeError}
              </div>
            )}
            {probed && (
              <div className="mt-2 space-y-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>
                <div className="flex justify-between">
                  <span>Output shape</span>
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-primary)" }}>[1, {probed.sample_shape.join(", ")}]</span>
                </div>
                <div className="flex justify-between">
                  <span>Dtype</span>
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-primary)" }}>{probed.sample_dtype}</span>
                </div>
                <div className="flex justify-between">
                  <span>Classes</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{probed.num_classes}</span>
                </div>
                <div className="flex justify-between">
                  <span>Label dtype</span>
                  <span className="font-mono text-[10px]" style={{ color: "var(--text-primary)" }}>{probed.label_dtype}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Data preview */}
      <DataPreview node={node} />

      <div style={{ borderTop: "1px solid var(--border)" }}>
        <div className="px-4 py-2 text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>ID: {node.id}</div>
        <div className="px-4 pb-3">
          <button onClick={deleteSelected} className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
            style={{ background: "var(--danger-subtle)", color: "var(--danger)", border: "1px solid transparent" }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--danger)")}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = "transparent")}
          >Delete Node</button>
        </div>
      </div>
    </>
  );
}

function TensorFilePathInput({ path, scanning, scanResult, scanError, onPathChange, onScan, inputStyle }: {
  path: string;
  scanning: boolean;
  scanResult: ScanResult | undefined;
  scanError: string | null;
  onPathChange: (p: string) => void;
  onScan: (path: string) => void;
  inputStyle: React.CSSProperties;
}) {
  // Browse: immediate scan (path passed directly)
  const handleBrowse = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false, directory: false,
        filters: [{ name: "Tensor files", extensions: ["pt", "npz", "npy", "safetensors"] }, { name: "All files", extensions: ["*"] }],
        title: "Select dataset",
      });
      if (selected) {
        onPathChange(selected);
        onScan(selected);
      }
    } catch {
      const inp = document.createElement("input"); inp.type = "file"; inp.accept = ".pt,.npz,.npy,.safetensors";
      inp.onchange = () => {
        const f = inp.files?.[0];
        if (f) { onPathChange(f.name); onScan(f.name); }
      };
      inp.click();
    }
  }, [onPathChange, onScan]);

  return (
    <div className="px-4 pb-3 space-y-2">
      {/* Path input + browse */}
      <div className="relative">
        <input type="text" value={path}
          onChange={(e) => onPathChange(e.target.value)}
          placeholder=".pt, .npz, or folder path"
          className="w-full pl-2.5 pr-8 py-2 text-xs rounded-md" style={inputStyle} />
        <button onClick={handleBrowse}
          className="absolute right-1.5 top-1/2 -translate-y-1/2 w-5 h-5 flex items-center justify-center rounded"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-secondary)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
          title="Browse"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      </div>
      {/* Scan button — explicit, no auto-scan on type */}
      {path && !scanResult && !scanning && (
        <button onClick={() => onScan(path)}
          className="w-full py-1.5 rounded-md text-[11px] font-medium"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}
        >
          Scan Data
        </button>
      )}
      {scanning && (
        <div className="text-[10px] text-center py-1" style={{ color: "var(--text-muted)" }}>Scanning...</div>
      )}
      {scanResult && (
        <div className="flex items-center justify-between text-[10px]" style={{ color: "var(--text-muted)" }}>
          <span className="tabular-nums">{scanResult.total_samples.toLocaleString()} samples · {scanResult.tensors.length} tensors</span>
          <button onClick={() => onScan(path)} className="text-[10px]" style={{ color: "var(--accent)" }}>rescan</button>
        </div>
      )}
      {scanError && (
        <div className="text-[10px] px-2 py-1.5 rounded-md" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
          {scanError}
        </div>
      )}
    </div>
  );
}

function DataOutputsEditor({ node, source, updateDataOutput }: {
  node: ReturnType<typeof useGraphStore.getState>["nodes"][number];
  source: string;
  updateDataOutput: (nodeId: string, handleId: string, patch: Partial<DataOutputDef>) => void;
}) {
  const setDataOutputs = useGraphStore((s) => s.setDataOutputs);
  const outputs: DataOutputDef[] = node.data.dataOutputs ?? DEFAULT_DATA_OUTPUTS;
  const scanResult = node.data.params._scan as ScanResult | undefined;
  const introspected = node.data.params._introspected as { columns?: string[] } | undefined;
  const columns = introspected?.columns ?? [];
  const ROLES = ["Model Input", "Loss Target", "Not Used"] as const;
  type Role = typeof ROLES[number];

  // Build role map: tensor name → role based on current outputs
  const getRoleForTensor = (tensorName: string): Role => {
    for (const o of outputs) {
      if (o.column === tensorName || (!o.column && o.id === tensorName)) {
        if (o.id === "y") return "Loss Target";
        return "Model Input";
      }
    }
    return "Not Used";
  };

  const handleRoleChange = (tensorName: string, role: Role, tensor: { dtype: string; shape: number[] }) => {
    const dtype = tensor.dtype.includes("int") || tensor.dtype.includes("long") ? "long" : "float32";
    const shape = tensor.shape.slice(1);
    let updated = outputs.map((o) => o.column === tensorName ? { ...o, column: "" } : { ...o });

    if (role === "Model Input") {
      // Find an unused input handle or create one
      const existing = updated.find((o) => o.id !== "y" && !o.column);
      if (existing) {
        existing.column = tensorName;
        existing.dtype = dtype;
        existing.shape = shape;
      } else {
        const id = `in_${updated.filter((o) => o.id !== "y").length}`;
        updated.push({ id, column: tensorName, dtype, shape });
      }
    } else if (role === "Loss Target") {
      // Clear any previous loss target, set new one
      for (const o of updated) {
        if (o.id === "y") o.column = "";
      }
      const yOut = updated.find((o) => o.id === "y");
      if (yOut) {
        yOut.column = tensorName;
        yOut.dtype = dtype;
        yOut.shape = shape;
      }
    }

    // Clean up empty non-default handles
    updated = updated.filter((o) => o.id === "x" || o.id === "y" || o.column);
    // Ensure x and y always exist
    if (!updated.find((o) => o.id === "x")) updated.unshift({ id: "x", column: "", dtype: "float32" });
    if (!updated.find((o) => o.id === "y")) updated.splice(1, 0, { id: "y", column: "", dtype: "long" });

    setDataOutputs(node.id, updated);
  };

  // Sample count mismatch check
  const sampleMismatch = useMemo(() => {
    if (!scanResult) return null;
    const mapped = outputs
      .map((o) => {
        const key = o.column || o.id;
        const t = scanResult.tensors.find((tt) => tt.name === key);
        return t ? { name: key, samples: t.shape[0] } : null;
      }).filter(Boolean) as { name: string; samples: number }[];
    if (mapped.length < 2) return null;
    const m = mapped.find((x) => x.samples !== mapped[0].samples);
    return m ? `"${mapped[0].name}" has ${mapped[0].samples} samples but "${m.name}" has ${m.samples}` : null;
  }, [outputs, scanResult]);

  // ── TensorFile with scan results: show tensor-centric view ──
  if (source === "TensorFile" && scanResult && scanResult.tensors.length > 0) {
    return (
      <div style={{ borderTop: "1px solid var(--border)" }}>
        <div className="px-4 pt-3 pb-1">
          <div className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
            Tensor Mapping
          </div>
        </div>
        <div className="px-4 pb-3 space-y-1">
          {scanResult.tensors.map((t) => {
            const role = getRoleForTensor(t.name);
            const isActive = role !== "Not Used";
            return (
              <div key={t.name} className="rounded-md px-2.5 py-2" style={{
                border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
                opacity: isActive ? 1 : 0.6,
              }}>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono font-medium" style={{ color: "var(--text-primary)" }}>
                    {t.name}
                  </span>
                  <select
                    value={role}
                    onChange={(e) => handleRoleChange(t.name, e.target.value as Role, t)}
                    className="text-[10px] px-1.5 py-0.5 rounded"
                    style={{
                      background: "transparent",
                      border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
                      color: isActive ? "var(--accent)" : "var(--text-muted)",
                    }}
                  >
                    {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                  </select>
                </div>
                <div className="mt-1 text-[10px] flex items-center gap-2" style={{ color: "var(--text-muted)" }}>
                  <span className="font-mono">[{t.shape.join(", ")}] {t.dtype}</span>
                  {t.num_classes != null && (
                    <span>{t.num_classes} classes{t.class_values && t.class_values.length <= 10 ? `: ${t.class_values.join(", ")}` : ""}</span>
                  )}
                  {t.mean != null && (
                    <span className="font-mono">mean {t.mean}</span>
                  )}
                </div>
              </div>
            );
          })}
          {/* Inline validation */}
          {sampleMismatch && (
            <div className="text-[10px] px-2 py-1.5 rounded-md" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
              {sampleMismatch}
            </div>
          )}
          {!outputs.some((o) => o.column && o.id !== "y") && (
            <div className="text-[10px] px-2 py-1.5 rounded-md" style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}>
              No tensor assigned as Model Input
            </div>
          )}
          {!outputs.find((o) => o.id === "y")?.column && (
            <div className="text-[10px] px-2 py-1.5 rounded-md" style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}>
              No tensor assigned as Loss Target
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── TensorFile without scan: prompt to scan ──
  if (source === "TensorFile") {
    return (
      <div style={{ borderTop: "1px solid var(--border)" }}>
        <div className="px-4 py-4 text-center">
          <div className="text-[11px] mb-1" style={{ color: "var(--text-secondary)" }}>
            Set a data path and scan to discover tensors
          </div>
          <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>
            Supports .pt, .npz, .npy, and .safetensors files
          </div>
        </div>
      </div>
    );
  }

  // ── Non-TensorFile custom sources (HuggingFace, CSV, etc.) ──
  return (
    <div style={{ borderTop: "1px solid var(--border)" }}>
      <div className="px-4 pt-3 pb-1">
        <div className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          Data Mapping
        </div>
      </div>
      <div className="px-4 pb-3 space-y-2">
        {outputs.map((o, i) => (
          <div key={o.id}>
            <label className="block text-[11px] font-medium mb-1" style={{ color: "var(--text-secondary)" }}>
              {i === 0 ? "Model Input" : i === 1 ? "Loss Target" : o.id}
            </label>
            <div className="flex items-center gap-1.5">
              {columns.length > 0 ? (
                <select value={o.column} onChange={(e) => updateDataOutput(node.id, o.id, { column: e.target.value })}
                  className="flex-1 px-2 py-1.5 text-[11px] rounded-md"
                  style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-primary)" }}>
                  <option value="">auto</option>
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              ) : (
                <input type="text" value={o.column || ""}
                  onChange={(e) => updateDataOutput(node.id, o.id, { column: e.target.value })}
                  placeholder={i === 0 ? "column for input" : i === 1 ? "column for target" : "column name"}
                  className="flex-1 px-2 py-1.5 text-[11px] rounded-md"
                  style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PreprocessEditor({ value, onChange }: { nodeId: string; value: string; onChange: (v: string) => void }) {
  const [expanded, setExpanded] = useState(!!value);
  const [editValue, setEditValue] = useState(value);
  const [editing, setEditing] = useState(false);

  const handleSave = useCallback(() => {
    onChange(editValue.trim());
    setEditing(false);
  }, [editValue, onChange]);

  const placeholder = `# Transform each row into (tensor, label).
# \`item\` is a dict with the dataset columns.
# Return (features_tensor, label_int).
features = [float(item['col1']), float(item['col2'])]
label = item['label']
return torch.tensor(features), label`;

  return (
    <div style={{ borderTop: "1px solid var(--border)" }}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2 text-[10px] font-semibold uppercase tracking-wider"
        style={{ color: "var(--text-muted)" }}
      >
        <span className="flex items-center gap-1.5">
          Preprocessing
          {value && (
            <span className="text-[8px] px-1 py-0.5 rounded normal-case tracking-normal font-medium"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}>custom</span>
          )}
        </span>
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.15s" }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>
      {expanded && (
        <div className="px-4 pb-3">
          <div className="text-[10px] mb-2 leading-relaxed" style={{ color: "var(--text-muted)" }}>
            Python function body. Receives <code style={{ color: "var(--accent)" }}>item</code> (dict), must return <code style={{ color: "var(--accent)" }}>(tensor, label)</code>.
          </div>
          {editing ? (
            <div className="space-y-2">
              <textarea
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                rows={8}
                className="w-full rounded-lg px-3 py-2 text-[10px] font-mono leading-relaxed resize-y"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--accent)" }}
                spellCheck={false}
                placeholder={placeholder}
              />
              <div className="flex gap-1.5">
                <button onClick={handleSave} className="px-2.5 py-1 rounded text-[10px] font-medium text-white" style={{ background: "var(--accent)" }}>Save</button>
                <button onClick={() => { setEditValue(value); setEditing(false); }} className="px-2.5 py-1 rounded text-[10px] font-medium"
                  style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}>Cancel</button>
                {value && (
                  <button onClick={() => { onChange(""); setEditValue(""); setEditing(false); }} className="px-2.5 py-1 rounded text-[10px] font-medium ml-auto" style={{ color: "var(--danger)" }}>Clear</button>
                )}
              </div>
            </div>
          ) : (
            <div>
              {value ? (
                <div
                  className="relative rounded-lg px-3 py-2 text-[10px] font-mono leading-relaxed overflow-x-auto cursor-pointer"
                  style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--accent)" }}
                  onClick={() => { setEditValue(value); setEditing(true); }}
                >
                  <code style={{ whiteSpace: "pre-wrap" }}>{value}</code>
                </div>
              ) : (
                <button
                  onClick={() => setEditing(true)}
                  className="w-full py-1.5 rounded-lg text-[11px] font-medium transition-colors"
                  style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)", border: "1px dashed var(--border)" }}
                >
                  + Add custom preprocessing
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generic NodeEditor
// ---------------------------------------------------------------------------

function NodeEditor({ node }: { node: ReturnType<typeof useGraphStore.getState>["nodes"][number] }) {
  // Branch: universal Data node gets its own editor
  if (node.data.moduleType === "Data") {
    return <DataNodeEditor node={node} />;
  }

  // Branch: code block mode
  if (node.data.codeBlock) {
    return <CodeBlockEditor node={node} />;
  }

  const updateNodeParam = useGraphStore((s) => s.updateNodeParam);
  const deleteSelected = useGraphStore((s) => s.deleteSelected);
  const convertToCodeBlock = useGraphStore((s) => s.convertToCodeBlock);
  const shapeError: ShapeError | undefined = useGraphStore((s) => s.shapeErrors[node.id]);
  const def = MODULE_REGISTRY[node.data.moduleType];
  const color = CATEGORY_COLORS[node.data.category];

  const canConvert = !["Input", "Output", "Data"].includes(node.data.moduleType) && !isDatasetNode(node.data.moduleType);

  return (
    <>
      <div className="px-4 py-3 flex items-start gap-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-[11px] font-bold flex-shrink-0 mt-0.5" style={{ background: color }}>
          {node.data.label.slice(0, 2).toUpperCase()}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>{node.data.label}</div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
            {isDatasetNode(node.data.moduleType) ? `dataset.${node.data.moduleType}` : `torch.nn.${node.data.moduleType}`}
          </div>
        </div>
      </div>

      {def && (
        <div className="px-4 py-2 text-[11px] leading-relaxed" style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>
          {def.description}
        </div>
      )}

      {/* Error banner */}
      {shapeError && <ErrorBanner error={shapeError} nodeId={node.id} />}

      <div className="flex-1 overflow-y-auto">
        {def?.params.length ? (
          <div className="px-4 py-3 space-y-2.5">
            {def.params.map((param) => (
              <ParamInput
                key={param.name}
                param={param}
                value={node.data.params[param.name]}
                onChange={(v) => updateNodeParam(node.id, param.name, v)}
                highlight={shapeError?.type === "root" && shapeError.paramHint === param.name}
              />
            ))}
          </div>
        ) : (
          <div className="px-4 py-4 text-[11px]" style={{ color: "var(--text-muted)" }}>No configurable parameters</div>
        )}
      </div>

      {/* Per-node code editor */}
      <CodeEditor node={node} />

      {/* Data preview for dataset nodes */}
      <DataPreview node={node} />

      <div style={{ borderTop: "1px solid var(--border)" }}>
        {canConvert && (
          <div className="px-4 pt-3">
            <button
              onClick={() => convertToCodeBlock(node.id)}
              className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid transparent" }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = "transparent")}
            >Convert to Code Block</button>
          </div>
        )}
        <div className="px-4 py-2 text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>ID: {node.id}</div>
        <div className="px-4 pb-3">
          <button onClick={deleteSelected} className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
            style={{ background: "var(--danger-subtle)", color: "var(--danger)", border: "1px solid transparent" }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--danger)")}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = "transparent")}
          >Delete Node</button>
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// CodeBlockEditor — full editor for code block mode nodes
// ---------------------------------------------------------------------------

function CodeBlockEditor({ node }: { node: ReturnType<typeof useGraphStore.getState>["nodes"][number] }) {
  const cb = node.data.codeBlock!;
  const updateCodeBlock = useGraphStore((s) => s.updateCodeBlock);
  const addCodeBlockHandle = useGraphStore((s) => s.addCodeBlockHandle);
  const removeCodeBlockHandle = useGraphStore((s) => s.removeCodeBlockHandle);
  const revertToStandard = useGraphStore((s) => s.revertToStandard);
  const deleteSelected = useGraphStore((s) => s.deleteSelected);
  const shapeError: ShapeError | undefined = useGraphStore((s) => s.shapeErrors[node.id]);
  const color = CATEGORY_COLORS[node.data.category];

  const [newInputName, setNewInputName] = useState("");
  const [newOutputName, setNewOutputName] = useState("");
  const [probing, setProbing] = useState(false);
  const [probeResult, setProbeResult] = useState<Record<string, number[]> | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);
  const shapeMap = useGraphStore((s) => s.shapeMap);
  const edges = useGraphStore((s) => s.edges);

  const handleProbe = useCallback(async () => {
    setProbing(true);
    setProbeError(null);
    setProbeResult(null);
    try {
      // Gather upstream shapes for each input handle
      const inputSpecs = cb.inputHandles.map((h) => {
        const edge = edges.find((e) => e.target === node.id && e.targetHandle === h.id);
        let shape = [1, 3, 32, 32]; // fallback
        if (edge) {
          const srcShape = shapeMap[edge.source]?.shape;
          if (srcShape) shape = srcShape;
        }
        return { name: h.id, shape, dtype: "float32" };
      });

      const result = await sidecarCall<{ shapes?: Record<string, number[]>; error?: string }>("node.probe", {
        init_code: cb.initBody,
        forward_code: cb.forwardBody,
        inputs: inputSpecs,
      });

      if (result.error) {
        setProbeError(result.error);
      } else if (result.shapes) {
        setProbeResult(result.shapes);
      }
    } catch (e) {
      setProbeError(e instanceof Error ? e.message : "Probe failed");
    }
    setProbing(false);
  }, [cb, node.id, edges, shapeMap]);

  const forwardSig = `def forward(self, ${cb.inputHandles.map((h) => h.id).join(", ")}):`;

  const handleAddInput = useCallback(() => {
    const name = safePythonId(newInputName.trim() || `in_${cb.inputHandles.length}`);
    // Check uniqueness
    const allIds = [...cb.inputHandles, ...cb.outputHandles].map((h) => h.id);
    if (allIds.includes(name)) return;
    addCodeBlockHandle(node.id, "input", { id: name, label: name, dataType: "tensor" });
    setNewInputName("");
  }, [newInputName, cb, node.id, addCodeBlockHandle]);

  const handleAddOutput = useCallback(() => {
    const name = safePythonId(newOutputName.trim() || `out_${cb.outputHandles.length}`);
    const allIds = [...cb.inputHandles, ...cb.outputHandles].map((h) => h.id);
    if (allIds.includes(name)) return;
    addCodeBlockHandle(node.id, "output", { id: name, label: name, dataType: "tensor" });
    setNewOutputName("");
  }, [newOutputName, cb, node.id, addCodeBlockHandle]);

  const inputCls = "w-full px-2.5 py-2 text-xs rounded-md";
  const inputStyle = { background: "transparent", border: "1px solid var(--border)", color: "var(--text-primary)" };

  return (
    <>
      {/* Header */}
      <div className="px-4 py-3 flex items-start gap-3" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-[11px] font-bold flex-shrink-0 mt-0.5" style={{ background: color }}>
          {node.data.label.slice(0, 2).toUpperCase()}
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>{node.data.label}</span>
            <span className="text-[8px] px-1 py-0.5 rounded font-mono flex-shrink-0"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}>&lt;/&gt;</span>
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
            Code Block (was {cb.originalModuleType})
          </div>
        </div>
      </div>

      {shapeError && <ErrorBanner error={shapeError} nodeId={node.id} />}

      <div className="flex-1 overflow-y-auto">
        {/* Input Handles */}
        <div className="px-4 pt-3 pb-2">
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Input Handles</div>
          <div className="space-y-1.5">
            {cb.inputHandles.map((h) => (
              <div key={h.id} className="flex items-center gap-2">
                <span className="flex-1 text-[11px] font-mono" style={{ color: "var(--text-primary)" }}>{h.id}</span>
                <button
                  onClick={() => removeCodeBlockHandle(node.id, "input", h.id)}
                  className="text-[9px] px-1.5 py-0.5 rounded transition-colors"
                  style={{ color: "var(--danger)", background: "var(--danger-subtle)" }}
                >Remove</button>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            <input
              type="text"
              value={newInputName}
              onChange={(e) => setNewInputName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleAddInput(); }}
              placeholder="handle name"
              className="flex-1 px-2 py-1 text-[11px] rounded-lg"
              style={inputStyle}
            />
            <button
              onClick={handleAddInput}
              className="text-[10px] px-2 py-1 rounded-lg font-medium transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}
            >+ Add</button>
          </div>
        </div>

        {/* Output Handles */}
        <div className="px-4 pt-2 pb-2" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Output Handles</div>
          <div className="space-y-1.5">
            {cb.outputHandles.map((h) => (
              <div key={h.id} className="flex items-center gap-2">
                <span className="flex-1 text-[11px] font-mono" style={{ color: "var(--text-primary)" }}>{h.id}</span>
                <button
                  onClick={() => removeCodeBlockHandle(node.id, "output", h.id)}
                  className="text-[9px] px-1.5 py-0.5 rounded transition-colors"
                  style={{ color: "var(--danger)", background: "var(--danger-subtle)" }}
                >Remove</button>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-1.5 mt-2">
            <input
              type="text"
              value={newOutputName}
              onChange={(e) => setNewOutputName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleAddOutput(); }}
              placeholder="handle name"
              className="flex-1 px-2 py-1 text-[11px] rounded-lg"
              style={inputStyle}
            />
            <button
              onClick={handleAddOutput}
              className="text-[10px] px-2 py-1 rounded-lg font-medium transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}
            >+ Add</button>
          </div>
        </div>

        {/* Init Body */}
        <div className="px-4 pt-2 pb-2" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>__init__</div>
          <textarea
            value={cb.initBody}
            onChange={(e) => updateCodeBlock(node.id, { initBody: e.target.value })}
            rows={3}
            className={`${inputCls} font-mono resize-y`}
            style={inputStyle}
            spellCheck={false}
          />
        </div>

        {/* Forward Body */}
        <div className="px-4 pt-2 pb-3" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>forward()</div>
          <div className="text-[10px] font-mono mb-1 px-1" style={{ color: "var(--text-muted)" }}>{forwardSig}</div>
          <textarea
            value={cb.forwardBody}
            onChange={(e) => updateCodeBlock(node.id, { forwardBody: e.target.value })}
            rows={3}
            className={`${inputCls} font-mono resize-y`}
            style={inputStyle}
            spellCheck={false}
          />
        </div>

        {/* Probe Output Shapes */}
        <div className="px-4 pb-3" style={{ borderTop: "1px solid var(--border)" }}>
          <button
            onClick={handleProbe}
            disabled={probing}
            className="w-full py-1.5 mt-2 rounded-lg text-[11px] font-medium transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", opacity: probing ? 0.6 : 1 }}
          >
            {probing ? "Probing..." : "Probe Output Shapes"}
          </button>
          {probeError && (
            <div className="mt-1.5 text-[10px] px-2 py-1 rounded" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
              {probeError}
            </div>
          )}
          {probeResult && (
            <div className="mt-1.5 space-y-0.5">
              {Object.entries(probeResult).map(([handle, shape]) => (
                <div key={handle} className="flex justify-between text-[11px]">
                  <span className="font-mono" style={{ color: "var(--text-secondary)" }}>{handle}</span>
                  <span className="font-mono" style={{ color: "var(--accent)" }}>{JSON.stringify(shape)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div style={{ borderTop: "1px solid var(--border)" }}>
        <div className="px-4 pt-3 space-y-2">
          <button
            onClick={() => revertToStandard(node.id)}
            className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid transparent" }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = "transparent")}
          >Revert to Standard</button>
        </div>
        <div className="px-4 py-2 text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>ID: {node.id}</div>
        <div className="px-4 pb-3">
          <button onClick={deleteSelected} className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
            style={{ background: "var(--danger-subtle)", color: "var(--danger)", border: "1px solid transparent" }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--danger)")}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = "transparent")}
          >Delete Node</button>
        </div>
      </div>
    </>
  );
}

function ParamInput({ param, value, onChange, highlight }: { param: ParamDef; value: unknown; onChange: (v: unknown) => void; highlight?: boolean }) {
  const s = {
    background: "transparent",
    border: highlight ? "1px solid var(--danger)" : "1px solid var(--border)",
    color: "var(--text-primary)",
    boxShadow: highlight ? "0 0 0 2px var(--danger-subtle)" : undefined,
  };
  const c = "w-full px-2.5 py-2 text-xs rounded-md";
  return (
    <div>
      <label className="block text-[11px] font-medium mb-1" style={{ color: highlight ? "var(--danger)" : "var(--text-secondary)" }}>
        {param.label ?? param.name}
        {highlight && <span className="ml-1 text-[9px] font-normal" style={{ color: "var(--danger)" }}>(mismatch)</span>}
      </label>
      {param.type === "bool" ? (
        <label className="flex items-center gap-2.5 cursor-pointer rounded-md px-2.5 py-2" style={{ border: "1px solid var(--border)" }}>
          <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} className="accent-[var(--accent)]" />
          <span className="text-xs" style={{ color: "var(--text-primary)" }}>{value ? "True" : "False"}</span>
        </label>
      ) : param.type === "select" ? (
        <select value={String(value ?? param.default)} onChange={(e) => onChange(e.target.value)} className={c} style={s}>
          {param.options?.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : param.type === "int" || param.type === "float" ? (
        <input type="number" value={value == null ? "" : String(value)} step={param.type === "float" ? "any" : "1"} min={param.min} max={param.max}
          onChange={(e) => { const v = param.type === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value); if (!isNaN(v)) onChange(v); }}
          className={c} style={s} />
      ) : (
        <input type="text" value={value == null ? "" : String(value)} onChange={(e) => onChange(e.target.value)} className={c} style={s} />
      )}
    </div>
  );
}
