import { useState, useMemo, useCallback, useRef, useEffect, useId } from "react";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore, runFraction } from "../store/training-store";
import { MODULE_REGISTRY, CATEGORY_COLORS } from "../lib/registry";
import { computeModelSummary, formatParamCount, formatFlops } from "../lib/model-summary";
import { formatCtorArgs, generateNodeSnippet } from "../lib/decompiler";
import TrainingPanel from "./TrainingPanel";
import GroupPanel from "./GroupPanel";
import { startTraining, stopTraining } from "../lib/training";
import type { ParamDef, DataOutputDef, ScanResult } from "../lib/types";
import { safePythonId, DEFAULT_DATA_OUTPUTS } from "../lib/types";
import { formatShape, type ShapeError } from "../lib/shape-inference";
import { shapeErrorFix } from "../lib/shape-fixes";
import { paramHelp, pytorchDocsUrl, torchName } from "../lib/param-help";
import { openExternal } from "../lib/open-external";
import HelpTip from "./ui/HelpTip";
import { focusNode } from "../lib/canvas-focus";
import { isDatasetNode, DATASET_METADATA, extractDatasetConfig, DATA_PRESETS, type DataDomain, type IntrospectionResult, type ProbeResult } from "../lib/dataset-metadata";
import { sidecarCall } from "../lib/sidecar";
import NumberField from "./NumberField";
import Select from "./ui/Select";
import Button from "./ui/Button";
import Tabs from "./ui/Tabs";
import { CategoryBadge } from "../lib/category-icons";

type Tab = "properties" | "training";

export default function PropertyPanel() {
  const nodes = useGraphStore((s) => s.nodes);
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const selectedGroup = useGraphStore((s) => (s.selectedGroupId ? s.groups.find((g) => g.id === s.selectedGroupId) : undefined));
  const [tab, setTab] = useState<Tab>("properties");
  const trainingStatus = useTrainingStore((s) => s.status);
  const stopRequested = useTrainingStore((s) => s.stopRequested);
  const prevSelectedRef = useRef<string | null>(null);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  // Auto-switch to properties when a NEW node is selected
  useEffect(() => {
    if (selectedNodeId && selectedNodeId !== prevSelectedRef.current) {
      setTab("properties");
    }
    prevSelectedRef.current = selectedNodeId;
  }, [selectedNodeId]);
  const selectedGroupId = selectedGroup?.id;
  useEffect(() => { if (selectedGroupId) setTab("properties"); }, [selectedGroupId]);

  const activeTab = selectedNode ? tab : tab;

  return (
    <div
      className="flex flex-col h-full overflow-hidden"
      style={{
        background: "var(--bg-secondary)",
      }}
    >
      {/* Tab bar */}
      <Tabs
        label="Side panel"
        stretch
        value={activeTab}
        onChange={setTab}
        className="h-9 flex-shrink-0 border-b border-line"
        items={[
          { id: "properties", label: "Properties" },
          { id: "training", label: "Training", live: trainingStatus === "running" },
        ]}
      />

      {/* Tab content */}
      {activeTab === "properties" ? (
        selectedGroup ? (
          <GroupPanel group={selectedGroup} />
        ) : selectedNode ? (
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
              <Button variant="secondary" size="lg" block onClick={stopTraining} disabled={stopRequested}>
                <span className="w-2.5 h-2.5 rounded-sm bg-danger" aria-hidden="true" />
                {stopRequested ? "Stopping…" : "Stop training"}
              </Button>
            ) : trainingStatus === "starting" ? (
              <Button variant="primary" size="lg" block disabled className="!opacity-80" style={{ animation: "pulse-subtle 1.5s ease-in-out infinite" }}>
                Starting…
              </Button>
            ) : (
              <Button variant="primary" size="lg" block onClick={startTraining}>
                Start training
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
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
        <div className="text-[11px] font-semibold uppercase tracking-wider mb-2.5" style={{ color: "var(--text-muted)" }}>
          Model Summary
        </div>
        <div className="space-y-2">
          <SummaryRow label="Parameters" value={formatParamCount(summary.totalParams)} />
          <SummaryRow label="Model Size (FP32)" value={`${summary.estimatedSizeMB.toFixed(1)} MB`} />
          {summary.estimatedFlops > 0 && <SummaryRow label="FLOPs" value={formatFlops(summary.estimatedFlops)} />}
          <SummaryRow label="Est. VRAM" value={`~${summary.estimatedVramMB.toFixed(0)} MB`}
            note={summary.estimatedVramMB > 6144 ? "High: may not fit on GPUs with 8 GB or less"
              : summary.estimatedVramMB > 3072 ? "May not fit on GPUs with 4 GB or less" : undefined}
            noteTone={summary.estimatedVramMB > 6144 ? "danger" : "warning"} />
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
        <div className="text-[11px] font-semibold uppercase tracking-wider mb-2.5" style={{ color: "var(--text-muted)" }}>
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
        <Button variant="primary" block onClick={onGoToTraining}>Configure training</Button>
      </div>
    </div>
  );
}

function SummaryRow({ label, value, note, noteTone = "warning" }: { label: string; value: string; note?: string; noteTone?: "warning" | "danger" }) {
  return (
    <div className="flex items-center justify-between text-[11px]">
      <span style={{ color: "var(--text-secondary)" }}>{label}</span>
      <span className="flex items-center gap-1 font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>
        {note && (
          <span className={noteTone === "danger" ? "text-danger" : "text-warning"} title={note} role="img" aria-label={note}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </span>
        )}
        {value}
      </span>
    </div>
  );
}

function TrainingProgress() {
  const status = useTrainingStore((s) => s.status);
  const currentEpoch = useTrainingStore((s) => s.currentEpoch);
  const epochs = useTrainingStore((s) => s.config.epochs);
  const metrics = useTrainingStore((s) => s.metrics);
  const fraction = useTrainingStore(runFraction);
  const downloading = useTrainingStore((s) => s.downloading);

  if (status !== "running") return null;

  const lastLoss = metrics.length > 0 ? metrics[metrics.length - 1].trainLoss : null;

  return (
    <div className="mb-3 space-y-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-fg-secondary">
          {downloading ? `Downloading ${downloading}…` : currentEpoch === 0 ? "Preparing data…" : `Epoch ${currentEpoch} of ${epochs}`}
        </span>
        {lastLoss !== null && (
          <span className="font-mono tabular-nums text-[11px] text-fg">
            loss {lastLoss.toFixed(4)}
          </span>
        )}
      </div>
      <div className="w-full h-1 rounded-full bg-surface-raised overflow-hidden">
        <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${fraction * 100}%` }} />
      </div>
    </div>
  );
}

// --- Error Banner ---
function ErrorBanner({ error, nodeId }: { error: ShapeError; nodeId: string }) {
  const sourceLabel = useGraphStore((s) => s.nodes.find((n) => n.id === error.sourceNodeId)?.data.label ?? error.sourceNodeId);
  const params = useGraphStore((s) => s.nodes.find((n) => n.id === nodeId)?.data.params);
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam);
  const fix = params ? shapeErrorFix(error, params) : null;

  if (error.type === "root") {
    return (
      <div role="alert" className="mx-4 mt-3 px-3 py-2.5 rounded-lg text-[11px] leading-relaxed bg-danger-subtle border border-danger text-danger">
        <div className="flex items-start gap-1.5 font-semibold">
          <AlertIcon />
          <span>{error.message}</span>
        </div>
        {error.receivedShape && error.expectedShape && (
          <div className="font-mono mt-1 ml-[18px] opacity-90">
            Gets {formatShape(error.receivedShape)}, expects {formatShape(error.expectedShape)}
          </div>
        )}
        {fix && (
          <Button size="sm" onClick={() => updateNodeParam(nodeId, fix.param, fix.to)} className="mt-2 ml-[18px]">
            Set {fix.param} to {fix.to}
          </Button>
        )}
      </div>
    );
  }

  // Further down from a problem: this layer's input shape isn't known until that one is fixed
  return (
    <div className="mx-4 mt-3 px-3 py-2.5 rounded-lg text-[11px] leading-relaxed bg-surface-sunken border border-line text-fg-secondary">
      <div>
        The input shape here isn't known until <span className="font-semibold text-fg">{sourceLabel}</span> is fixed.
      </div>
      <Button size="sm" onClick={() => focusNode(error.sourceNodeId)} className="mt-2">
        Go to {sourceLabel}
      </Button>
    </div>
  );
}

function AlertIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-[2px]" aria-hidden="true">
      <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
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
        className="w-full flex items-center justify-between px-4 py-2 text-[11px] font-semibold uppercase tracking-wider"
        style={{ color: "var(--text-muted)" }}
      >
        <span className="flex items-center gap-1.5">
          Code
          {isCustom && (
            <span className="text-[11px] px-1 py-0.5 rounded normal-case tracking-normal font-medium"
              style={{ background: "var(--accent-subtle)", color: "var(--accent-text)" }}>edited</span>
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
                <div className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>__init__</div>
              )}
              <textarea
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                rows={isDataset ? 3 : 1}
                className="w-full rounded-lg px-3 py-2 text-[11px] font-mono leading-relaxed resize-y"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--accent)" }}
                spellCheck={false}
              />
              {/* Show forward line as grayed-out read-only when editing a non-dataset node */}
              {!isDataset && snippet && (
                <div>
                  <div className="text-[11px] font-medium mt-1 mb-1" style={{ color: "var(--text-muted)" }}>forward (from graph)</div>
                  <div
                    className="rounded-lg px-3 py-2 text-[11px] font-mono leading-relaxed"
                    style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)", border: "1px dashed var(--border)" }}
                  >
                    <code style={{ whiteSpace: "pre-wrap" }}>{snippet.forwardCode}</code>
                  </div>
                </div>
              )}
              <div className="flex gap-1.5">
                <Button variant="primary" size="sm" onClick={handleSave}>Save</Button>
                <Button size="sm" onClick={() => setEditing(false)}>Cancel</Button>
                {isCustom && (
                  <Button variant="ghost" size="sm" onClick={handleReset} className="ml-auto">Reset to default</Button>
                )}
              </div>
            </div>
          ) : (
            <div
              className="relative rounded-lg px-3 py-2 text-[11px] font-mono leading-relaxed overflow-x-auto"
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
                    className="px-1.5 py-0.5 rounded text-[11px] font-sans transition-colors bg-surface-elevated text-fg-secondary border border-line hover:text-fg"
                    title="Edit init code"
                  >Edit</button>
                )}
                <button
                  onClick={handleCopy}
                  className={`px-1.5 py-0.5 rounded text-[11px] font-sans transition-colors border border-line ${
                    copied ? "bg-success-subtle text-success" : "bg-surface-elevated text-fg-secondary hover:text-fg"
                  }`}
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
        className="w-full flex items-center justify-between px-4 py-2 text-[11px] font-semibold uppercase tracking-wider"
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
          <Button size="sm" block onClick={handlePreview} disabled={loading} className="mb-2">
            {loading ? "Loading…" : "Preview data"}
          </Button>
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
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-primary)" }}>{JSON.stringify(preview.sample_shape)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Classes</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{preview.num_classes}</span>
                </div>
                <div className="flex justify-between">
                  <span>Dtype</span>
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-primary)" }}>{preview.sample_dtype}</span>
                </div>
                <div className="mt-1.5 pt-1.5" style={{ borderTop: "1px solid var(--border)" }}>
                  <div className="text-[11px] font-semibold uppercase tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>Stats</div>
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
        <CategoryBadge category="data" color={color} size={32} className="mt-0.5" />
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
          <Select
            value={source}
            onChange={handleSourceChange}
            placeholder="Select a source…"
            aria-label="Data source"
            className={inputCls}
            style={inputStyle}
            options={[
              ...PRESET_SOURCES.map((s) => ({ value: s, label: DATA_PRESETS[s]?.label ?? s, group: "Presets" })),
              ...CUSTOM_SOURCES.map((s) => ({ value: s, label: CUSTOM_SOURCE_LABELS[s] ?? s, group: "Custom" })),
            ]}
          />
          {isPreset && (
            <div className="mt-1.5 text-[11px]" style={{ color: "var(--text-muted)" }}>
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
            <Button size="sm" block onClick={handleIntrospect} disabled={introspecting}>
              {introspecting ? "Inspecting…" : "Inspect data"}
            </Button>
            {introspectError && (
              <div className="mt-1.5 text-[11px] px-2 py-1 rounded" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
                {introspectError}
              </div>
            )}
          </div>
        )}

        {/* Introspection results */}
        {introspected && (
          <div className="px-4 pb-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-wider mt-2 mb-2" style={{ color: "var(--text-muted)" }}>
              Introspection
            </div>
            <div className="space-y-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>
              {introspected.columns.length > 0 && (
                <div className="flex justify-between">
                  <span>Columns</span>
                  <span className="font-mono text-[11px] truncate ml-2" style={{ color: "var(--text-primary)", maxWidth: "140px" }}>
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
                  <span className="font-mono text-[11px] truncate ml-2" style={{ color: "var(--text-primary)", maxWidth: "140px" }}>
                    {introspected.label_values.join(", ")}
                  </span>
                </div>
              )}
              {introspected.sample_shape.length > 0 && (
                <div className="flex justify-between">
                  <span>Sample shape</span>
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-primary)" }}>{JSON.stringify(introspected.sample_shape)}</span>
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
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-primary)" }}>{introspected.splits.join(", ")}</span>
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
                <div className="mt-2 px-2.5 py-2 rounded-lg text-[11px] leading-relaxed"
                  style={{ background: "var(--warning-subtle)", border: "1px solid var(--warning)", color: "var(--warning)" }}>
                  <div className="font-semibold mb-1">Preprocessing required</div>
                  <div className="mb-1.5" style={{ opacity: 0.85 }}>
                    This dataset has nested/complex columns ({complexCols.join(", ")}).
                    Add a preprocessing script below to extract features and labels.
                  </div>
                  {example && (
                    <pre className="font-mono text-[11px] p-1.5 rounded overflow-x-auto"
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
            <Select
              value={String(p.domain ?? "auto")}
              onChange={(v) => updateNodeParam(node.id, "domain", v)}
              aria-label="Domain"
              className={inputCls}
              style={inputStyle}
              options={[
                { value: "auto", label: "Auto-detect" },
                { value: "vision", label: "Vision" },
                { value: "text", label: "Text" },
                { value: "tabular", label: "Tabular" },
                { value: "audio", label: "Audio" },
              ]}
            />
          </div>
        )}

        {/* Domain-specific params */}
        {source && (domain === "vision" || (isPreset && preset.domain === "vision")) && (
          <div className="px-4 pb-3 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-wider mt-2" style={{ color: "var(--text-muted)" }}>Vision</div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Image Size</label>
              <NumberField value={p.image_size} integer min={8} max={1024}
                onChange={(v) => updateNodeParam(node.id, "image_size", v)}
                className={inputCls} style={inputStyle} />
            </div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Channels</label>
              <Select value={String(p.channels ?? "rgb")} onChange={(v) => updateNodeParam(node.id, "channels", v)}
                aria-label="Channels" className={inputCls} style={inputStyle}
                options={[{ value: "rgb", label: "RGB" }, { value: "grayscale", label: "Grayscale" }]} />
            </div>
          </div>
        )}

        {source && (domain === "text" || (isPreset && preset.domain === "text")) && (
          <div className="px-4 pb-3 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-wider mt-2" style={{ color: "var(--text-muted)" }}>Text</div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Max Length</label>
              <NumberField value={p.max_length} integer min={8} max={2048}
                onChange={(v) => updateNodeParam(node.id, "max_length", v)}
                className={inputCls} style={inputStyle} />
            </div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Vocab Size</label>
              <NumberField value={p.vocab_size} integer min={100} max={100000}
                onChange={(v) => updateNodeParam(node.id, "vocab_size", v)}
                className={inputCls} style={inputStyle} />
            </div>
            {/* Column dropdowns from introspection */}
            {!isPreset && (
              <>
                <div>
                  <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Text Column</label>
                  {columns.length > 0 ? (
                    <Select value={String(p.text_column ?? "")} onChange={(v) => updateNodeParam(node.id, "text_column", v)}
                      aria-label="Text column" className={inputCls} style={inputStyle}
                      options={[{ value: "", label: "Auto-detect" }, ...columns]} />
                  ) : (
                    <input type="text" value={String(p.text_column ?? "")} onChange={(e) => updateNodeParam(node.id, "text_column", e.target.value)}
                      placeholder="auto-detect" className={inputCls} style={inputStyle} />
                  )}
                </div>
                <div>
                  <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Label Column</label>
                  {columns.length > 0 ? (
                    <Select value={String(p.label_column ?? "")} onChange={(v) => updateNodeParam(node.id, "label_column", v)}
                      aria-label="Label column" className={inputCls} style={inputStyle}
                      options={[{ value: "", label: "Auto-detect" }, ...columns]} />
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
            <div className="text-[11px] font-semibold uppercase tracking-wider mt-2" style={{ color: "var(--text-muted)" }}>Tabular</div>
            <div>
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Target Column</label>
              {columns.length > 0 ? (
                <Select value={String(p.target_column ?? "")} onChange={(v) => updateNodeParam(node.id, "target_column", v)}
                  aria-label="Target column" className={inputCls} style={inputStyle}
                  options={[{ value: "", label: "Auto (last column)" }, ...columns]} />
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
                  <Select value={String(p.delimiter ?? ",")} onChange={(v) => updateNodeParam(node.id, "delimiter", v)}
                    aria-label="Delimiter" className={inputCls} style={inputStyle}
                    options={[{ value: ",", label: "Comma (,)" }, { value: ";", label: "Semicolon (;)" }, { value: "|", label: "Pipe (|)" }, { value: "\t", label: "Tab" }]} />
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
              <Select value={String(p.normalization ?? "none")} onChange={(v) => updateNodeParam(node.id, "normalization", v)}
                aria-label="Normalization" className={inputCls} style={inputStyle}
                options={[{ value: "none", label: "None" }, { value: "standard", label: "Standard (z-score)" }, { value: "minmax", label: "Min-Max" }]} />
            </div>
          </div>
        )}

        {/* HuggingFace vision: image column */}
        {source === "HuggingFace" && domain === "vision" && !isPreset && (
          <div className="px-4 pb-3">
            <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Image Column</label>
            {columns.length > 0 ? (
              <Select value={String(p.image_column ?? "")} onChange={(v) => updateNodeParam(node.id, "image_column", v)}
                aria-label="Image column" className={inputCls} style={inputStyle}
                options={[{ value: "", label: "Auto-detect" }, ...columns]} />
            ) : (
              <input type="text" value={String(p.image_column ?? "")} onChange={(e) => updateNodeParam(node.id, "image_column", e.target.value)}
                placeholder="auto-detect" className={inputCls} style={inputStyle} />
            )}
          </div>
        )}


        {/* Augmentation — vision only */}
        {source && (domain === "vision" || (isPreset && preset.domain === "vision")) && (
          <div className="px-4 pb-3 space-y-2" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="text-[11px] font-semibold uppercase tracking-wider mt-2 mb-1" style={{ color: "var(--text-muted)" }}>Augmentation</div>
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
              <NumberField value={p.rotation ?? 0} integer min={0} max={180}
                onChange={(v) => updateNodeParam(node.id, "rotation", v)}
                className={inputCls} style={inputStyle} />
            </div>
          </div>
        )}

        {/* Val split — shown when source is selected */}
        {source && (
          <div className="px-4 pb-3" style={{ borderTop: "1px solid var(--border)" }}>
            <div className="mt-2">
              <label className="block text-[11px] font-medium mb-1.5" style={{ color: "var(--text-secondary)" }}>Val Split</label>
              <NumberField value={p.val_split ?? 0.2} step="any" min={0} max={0.5}
                onChange={(v) => updateNodeParam(node.id, "val_split", v)}
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
            <Button size="sm" block onClick={handleProbe} disabled={probing}>
              {probing ? "Probing…" : "Probe output shape"}
            </Button>
            {probeError && (
              <div className="mt-1.5 text-[11px] px-2 py-1 rounded" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
                {probeError}
              </div>
            )}
            {probed && (
              <div className="mt-2 space-y-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>
                <div className="flex justify-between">
                  <span>Output shape</span>
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-primary)" }}>[1, {probed.sample_shape.join(", ")}]</span>
                </div>
                <div className="flex justify-between">
                  <span>Dtype</span>
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-primary)" }}>{probed.sample_dtype}</span>
                </div>
                <div className="flex justify-between">
                  <span>Classes</span>
                  <span className="font-medium tabular-nums" style={{ color: "var(--text-primary)" }}>{probed.num_classes}</span>
                </div>
                <div className="flex justify-between">
                  <span>Label dtype</span>
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-primary)" }}>{probed.label_dtype}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Data preview */}
      <DataPreview node={node} />

      <div style={{ borderTop: "1px solid var(--border)" }}>
        <div className="px-4 py-3">
          <Button variant="danger" block onClick={deleteSelected}>Delete layer</Button>
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
        <Button size="sm" block onClick={() => onScan(path)}>Scan data</Button>
      )}
      {scanning && (
        <div className="text-[11px] text-center py-1" style={{ color: "var(--text-muted)" }}>Scanning...</div>
      )}
      {scanResult && (
        <div className="flex items-center justify-between text-[11px]" style={{ color: "var(--text-muted)" }}>
          <span className="tabular-nums">{scanResult.total_samples.toLocaleString()} samples · {scanResult.tensors.length} tensors</span>
          <button onClick={() => onScan(path)} className="text-[11px]" style={{ color: "var(--accent-text)" }}>rescan</button>
        </div>
      )}
      {scanError && (
        <div className="text-[11px] px-2 py-1.5 rounded-md" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
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
                  <Select
                    value={role}
                    onChange={(v) => handleRoleChange(t.name, v as Role, t)}
                    options={[...ROLES]}
                    aria-label={`Use of ${t.name}`}
                    className="text-[11px] px-1.5 py-0.5 rounded"
                    style={{
                      background: "transparent",
                      border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
                      color: isActive ? "var(--accent-text)" : "var(--text-muted)",
                    }}
                  />
                </div>
                <div className="mt-1 text-[11px] flex items-center gap-2" style={{ color: "var(--text-muted)" }}>
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
            <div className="text-[11px] px-2 py-1.5 rounded-md" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
              {sampleMismatch}
            </div>
          )}
          {!outputs.some((o) => o.column && o.id !== "y") && (
            <div className="text-[11px] px-2 py-1.5 rounded-md" style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}>
              No tensor assigned as Model Input
            </div>
          )}
          {!outputs.find((o) => o.id === "y")?.column && (
            <div className="text-[11px] px-2 py-1.5 rounded-md" style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}>
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
          <div className="text-[11px]" style={{ color: "var(--text-muted)" }}>
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
                <Select value={o.column} onChange={(v) => updateDataOutput(node.id, o.id, { column: v })}
                  options={[{ value: "", label: "auto" }, ...columns]}
                  aria-label={i === 0 ? "Model input column" : i === 1 ? "Loss target column" : `${o.id} column`}
                  className="flex-1 min-w-0 px-2 py-1.5 text-[11px] rounded-md"
                  style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-primary)" }} />
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
        className="w-full flex items-center justify-between px-4 py-2 text-[11px] font-semibold uppercase tracking-wider"
        style={{ color: "var(--text-muted)" }}
      >
        <span className="flex items-center gap-1.5">
          Preprocessing
          {value && (
            <span className="text-[11px] px-1 py-0.5 rounded normal-case tracking-normal font-medium"
              style={{ background: "var(--accent-subtle)", color: "var(--accent-text)" }}>custom</span>
          )}
        </span>
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.15s" }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>
      {expanded && (
        <div className="px-4 pb-3">
          <div className="text-[11px] mb-2 leading-relaxed" style={{ color: "var(--text-muted)" }}>
            Python function body. Receives <code style={{ color: "var(--accent-text)" }}>item</code> (dict), must return <code style={{ color: "var(--accent-text)" }}>(tensor, label)</code>.
          </div>
          {editing ? (
            <div className="space-y-2">
              <textarea
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                rows={8}
                className="w-full rounded-lg px-3 py-2 text-[11px] font-mono leading-relaxed resize-y"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--accent)" }}
                spellCheck={false}
                placeholder={placeholder}
              />
              <div className="flex gap-1.5">
                <Button variant="primary" size="sm" onClick={handleSave}>Save</Button>
                <Button size="sm" onClick={() => { setEditValue(value); setEditing(false); }}>Cancel</Button>
                {value && (
                  <Button variant="ghost" size="sm" onClick={() => { onChange(""); setEditValue(""); setEditing(false); }} className="ml-auto">Clear</Button>
                )}
              </div>
            </div>
          ) : (
            <div>
              {value ? (
                <div
                  className="relative rounded-lg px-3 py-2 text-[11px] font-mono leading-relaxed overflow-x-auto cursor-pointer"
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
        <CategoryBadge category={node.data.category} color={color} size={32} className="mt-0.5" />
        <div className="min-w-0">
          <div className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>{node.data.label}</div>
          <TorchCaption moduleType={node.data.moduleType} />
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
                moduleType={node.data.moduleType}
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
            <Button block onClick={() => convertToCodeBlock(node.id)}>Convert to Code Block</Button>
          </div>
        )}
        <div className="px-4 py-3">
          <Button variant="danger" block onClick={deleteSelected}>Delete layer</Button>
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
        <CategoryBadge category={node.data.category} color={color} size={32} className="mt-0.5" />
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>{node.data.label}</span>
            <span className="text-[11px] px-1 py-0.5 rounded font-mono flex-shrink-0"
              style={{ background: "var(--accent-subtle)", color: "var(--accent-text)" }}>&lt;/&gt;</span>
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
          <div className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Input Handles</div>
          <div className="space-y-1.5">
            {cb.inputHandles.map((h) => (
              <div key={h.id} className="flex items-center gap-2">
                <span className="flex-1 text-[11px] font-mono" style={{ color: "var(--text-primary)" }}>{h.id}</span>
                <Button variant="danger" size="sm" className="!h-6" onClick={() => removeCodeBlockHandle(node.id, "input", h.id)}>Remove</Button>
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
            <Button size="sm" onClick={handleAddInput}>Add</Button>
          </div>
        </div>

        {/* Output Handles */}
        <div className="px-4 pt-2 pb-2" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>Output Handles</div>
          <div className="space-y-1.5">
            {cb.outputHandles.map((h) => (
              <div key={h.id} className="flex items-center gap-2">
                <span className="flex-1 text-[11px] font-mono" style={{ color: "var(--text-primary)" }}>{h.id}</span>
                <Button variant="danger" size="sm" className="!h-6" onClick={() => removeCodeBlockHandle(node.id, "output", h.id)}>Remove</Button>
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
            <Button size="sm" onClick={handleAddOutput}>Add</Button>
          </div>
        </div>

        {/* Init Body */}
        <div className="px-4 pt-2 pb-2" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>__init__</div>
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
          <div className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>forward()</div>
          <div className="text-[11px] font-mono mb-1 px-1" style={{ color: "var(--text-muted)" }}>{forwardSig}</div>
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
          <Button size="sm" block onClick={handleProbe} disabled={probing} className="mt-2">
            {probing ? "Probing…" : "Probe output shapes"}
          </Button>
          {probeError && (
            <div className="mt-1.5 text-[11px] px-2 py-1 rounded" style={{ background: "var(--danger-subtle)", color: "var(--danger)" }}>
              {probeError}
            </div>
          )}
          {probeResult && (
            <div className="mt-1.5 space-y-0.5">
              {Object.entries(probeResult).map(([handle, shape]) => (
                <div key={handle} className="flex justify-between text-[11px]">
                  <span className="font-mono" style={{ color: "var(--text-secondary)" }}>{handle}</span>
                  <span className="font-mono" style={{ color: "var(--accent-text)" }}>{JSON.stringify(shape)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div style={{ borderTop: "1px solid var(--border)" }}>
        <div className="px-4 pt-3 space-y-2">
          <Button block onClick={() => revertToStandard(node.id)}>Revert to Standard</Button>
        </div>
        <div className="px-4 py-3">
          <Button variant="danger" block onClick={deleteSelected}>Delete layer</Button>
        </div>
      </div>
    </>
  );
}

/** "torch.nn.Linear", linking to its page in the PyTorch docs */
function TorchCaption({ moduleType }: { moduleType: string }) {
  const name = isDatasetNode(moduleType) ? `dataset.${moduleType}` : torchName(moduleType);
  const url = isDatasetNode(moduleType) ? null : pytorchDocsUrl(moduleType);
  if (!name) return null;
  if (!url) return <div className="text-[11px] mt-0.5 font-mono text-fg-muted">{name}</div>;
  return (
    <button type="button" onClick={() => void openExternal(url)} title="Open the PyTorch documentation"
      className="inline-flex items-center gap-1 text-[11px] mt-0.5 font-mono text-fg-muted hover:text-accent-text hover:underline underline-offset-2">
      {name}
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" />
      </svg>
      <span className="sr-only">(opens the PyTorch documentation)</span>
    </button>
  );
}

function ParamInput({ moduleType, param, value, onChange, highlight }: { moduleType: string; param: ParamDef; value: unknown; onChange: (v: unknown) => void; highlight?: boolean }) {
  const s = {
    background: "transparent",
    border: highlight ? "1px solid var(--danger)" : "1px solid var(--border)",
    color: "var(--text-primary)",
    boxShadow: highlight ? "0 0 0 2px var(--danger-subtle)" : undefined,
  };
  const c = "w-full px-2.5 py-2 text-xs rounded-md";
  const id = useId();
  const help = paramHelp(moduleType, param.name);
  const helpId = help ? `${id}-help` : undefined;
  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label htmlFor={id} className="text-[11px] font-medium" style={{ color: highlight ? "var(--danger)" : "var(--text-secondary)" }}>
          {param.label ?? param.name}
          {highlight && <span className="ml-1 text-[11px] font-normal" style={{ color: "var(--danger)" }}>(mismatch)</span>}
        </label>
        {help && helpId && <HelpTip id={helpId} text={help} label={`About ${param.label ?? param.name}`} />}
      </div>
      {param.type === "bool" ? (
        <label className="flex items-center gap-2.5 cursor-pointer rounded-md px-2.5 py-2" style={{ border: "1px solid var(--border)" }}>
          <input id={id} type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} aria-describedby={helpId} className="accent-[var(--accent)]" />
          <span className="text-xs" style={{ color: "var(--text-primary)" }}>{value ? "True" : "False"}</span>
        </label>
      ) : param.type === "select" ? (
        <Select value={String(value ?? param.default)} onChange={onChange} options={param.options ?? []}
          aria-label={param.label ?? param.name} aria-describedby={helpId} className={c} style={s} />
      ) : param.type === "int" || param.type === "float" ? (
        <NumberField id={id} aria-describedby={helpId} value={value} integer={param.type === "int"} step={param.type === "float" ? "any" : "1"} min={param.min} max={param.max}
          onChange={onChange}
          className={c} style={s} />
      ) : (
        <input id={id} aria-describedby={helpId} type="text" value={value == null ? "" : String(value)} onChange={(e) => onChange(e.target.value)} className={c} style={s} />
      )}
    </div>
  );
}
