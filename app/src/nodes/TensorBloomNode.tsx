import { memo, useState, useRef, useEffect, useCallback, useMemo } from "react";
import { Handle, Position, useStore, type NodeProps, type Node } from "@xyflow/react";
import { MODULE_REGISTRY, CATEGORY_COLORS } from "../lib/registry";
import { resolveHandlesWithRegistry } from "../lib/types";
import { useGraphStore, type NodeData } from "../store/graph-store";
import { useTrainingStore } from "../store/training-store";
import { formatShape } from "../lib/shape-inference";
import { formatParamCount } from "../lib/model-summary";
import { DATASET_METADATA, DATA_PRESETS, resolveDataNodeShape } from "../lib/dataset-metadata";
import { quickParamCount, getParamSummary } from "../lib/node-summary";

type TLNodeProps = NodeProps<Node<NodeData>>;

// Shapes that are trivial pass-throughs and don't add useful info
const TRIVIAL_SHAPES = new Set(["?", "unknown", ""]);

function isTrivialShape(shape: number[] | string | undefined): boolean {
  if (!shape) return true;
  if (typeof shape === "string") return TRIVIAL_SHAPES.has(shape);
  return false;
}

// Below this zoom the detail lines are unreadable, so nodes show just a larger title
const COMPACT_ZOOM = 0.5;

function TensorBloomNode({ id, data, selected }: TLNodeProps) {
  const color = CATEGORY_COLORS[data.category];
  const compact = useStore((s) => s.transform[2] < COMPACT_ZOOM);
  const resolved = resolveHandlesWithRegistry(data, MODULE_REGISTRY);
  const inputHandles = resolved.inputHandles;
  const outputHandles = resolved.outputHandles;
  const isCodeBlock = !!data.codeBlock;
  const showHandleLabels = !compact && (isCodeBlock || data.category === "loss" || !!data.dataOutputs || inputHandles.length > 1 || outputHandles.length > 1);
  const paramSummary = getParamSummary(data);
  const shapeMap = useGraphStore((s) => s.shapeMap);
  const shapeDisplayBatch = useGraphStore((s) => s.shapeDisplayBatch);
  const batchSize = useTrainingStore((s) => s.config.batchSize);
  const shapeInfo = shapeMap[id];
  const renameNode = useGraphStore((s) => s.renameNode);
  const effectiveBatch = shapeDisplayBatch ? batchSize : undefined;

  const paramCount = quickParamCount(data.moduleType, data.params);

  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(data.label);
  const [hovered, setHovered] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const commitRename = useCallback(() => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== data.label) renameNode(id, trimmed);
    else setEditValue(data.label);
    setEditing(false);
  }, [editValue, data.label, id, renameNode]);


  const showShape = shapeInfo && !isTrivialShape(shapeInfo.shape);

  const maxHandles = Math.max(inputHandles.length, outputHandles.length);
  const isDataNode = !!data.dataOutputs;
  const hasTargetHandle = data.category === "loss" && inputHandles.some((h) => h.id === "target");
  // Data nodes need extra width for right-side handle labels (e.g. "y [1] →")
  // Loss nodes with target handle on right also need extra width
  const nodeWidth = isDataNode ? 240 : hasTargetHandle ? 220 : maxHandles > 2 ? 200 + (maxHandles - 2) * 40 : 200;
  // Data nodes need enough height for all right-side handles
  const rightHandleCount = isDataNode ? Math.max(outputHandles.length - 1, 0) : 0;
  // and room under the summary line so the right-side labels don't run into it
  const nodeMinHeight = rightHandleCount > 2 ? 66 + rightHandleCount * 22 : isDataNode && !compact ? 88 : undefined;

  return (
    <div
      className="relative rounded-lg overflow-hidden"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: nodeWidth,
        minHeight: nodeMinHeight,
        background: "var(--bg-secondary)",
        border: `1.5px solid ${selected ? color : hovered ? "var(--bg-elevated)" : "var(--border)"}`,
        boxShadow: selected
          ? `var(--node-shadow-hover), 0 0 0 2px ${color}20`
          : hovered
            ? "var(--node-shadow-hover)"
            : "var(--node-shadow)",
        transition: "border-color 0.15s ease, box-shadow 0.2s ease",
      }}
    >
      {/* Color accent bar */}
      <div className="h-[3px]" style={{ background: color }} />

      {/* Header */}
      <div className={`flex items-center gap-2 px-3 font-semibold ${compact ? "py-3 text-[26px]" : "pt-2 pb-1.5 text-[14px]"}`}>
        <div
          className={`${compact ? "hidden" : "flex"} w-[18px] h-[18px] rounded-[4px] items-center justify-center text-[8px] font-bold text-white flex-shrink-0 opacity-90`}
          style={{ background: color }}
        >
          {data.label.slice(0, 2).toUpperCase()}
        </div>
        {editing ? (
          <input
            ref={inputRef}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") { setEditValue(data.label); setEditing(false); }
            }}
            className="flex-1 min-w-0 px-1 py-0 text-[14px] font-semibold rounded outline-none"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--accent)" }}
          />
        ) : (
          <>
            <span
              className="truncate cursor-text"
              style={{ color: "var(--text-primary)" }}
              onDoubleClick={() => { setEditValue(data.label); setEditing(true); }}
              title="Double-click to rename"
            >
              {data.label}
            </span>
            {isCodeBlock && (
              <span
                className="text-[10px] px-1 py-0.5 rounded font-mono flex-shrink-0"
                style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}
                title="Code block mode"
              >
                &lt;/&gt;
              </span>
            )}
          </>
        )}
      </div>

      {/* Param summary + count */}
      {!compact && (paramSummary || paramCount > 0) && (
        <div className={`px-3 ${data.dataOutputs ? "pb-4" : "pb-1"}`}>
          {paramSummary && (
            <div className={`text-[12px] font-mono leading-snug ${paramSummary.includes("\n") ? "whitespace-pre-wrap" : "truncate"}`} style={{ color: "var(--text-secondary)" }}>
              {paramSummary}
            </div>
          )}
          {paramCount > 0 && (
            <div className="text-[11px] tabular-nums mt-0.5" style={{ color: "var(--text-muted)" }}>
              {formatParamCount(paramCount)} params
            </div>
          )}
        </div>
      )}

      {/* Output shape — only shown when non-trivial, hidden for Data nodes (handles show shapes) */}
      {!compact && showShape && !data.dataOutputs && (
        <div className="px-3 pb-2.5 flex items-center gap-1.5">
          <span className="text-[11px] font-mono" style={{ color: "var(--text-muted)" }}>
            {formatShape(shapeInfo.shape, effectiveBatch)}
          </span>
          {shapeDisplayBatch && shapeInfo.shape.length >= 2 && (
            <span className="text-[10px] font-mono" style={{ color: "var(--text-muted)", opacity: 0.7 }}>
              N={batchSize}
            </span>
          )}
        </div>
      )}

      {/* Dataset compatibility badge — Input nodes only */}
      {!compact && data.moduleType === "Input" && <DatasetBadge nodeParams={data.params} />}

      {/* Input handles */}
      {inputHandles.map((h) => {
        const isLossNode = data.category === "loss";
        // For Loss nodes: "input"/logits on top, "target" on right
        if (isLossNode && h.id === "target") {
          return (
            <span key={`in-${h.id}`}>
              <Handle type="target" position={Position.Right} id={h.id}
                style={{ top: "50%" }} title={h.label} />
              {showHandleLabels && (
                <span
                  className="absolute text-[10px] font-mono pointer-events-none"
                  style={{ right: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", whiteSpace: "nowrap" }}
                >
                  {h.label} →
                </span>
              )}
            </span>
          );
        }
        // Default: top handles
        const nonTargetHandles = isLossNode ? inputHandles.filter((hh) => hh.id !== "target") : inputHandles;
        const idx = nonTargetHandles.indexOf(h);
        const leftPct = ((idx + 1) / (nonTargetHandles.length + 1)) * 100;
        return (
          <span key={`in-${h.id}`}>
            <Handle type="target" position={Position.Top} id={h.id}
              style={{ left: `${leftPct}%`, top: 0 }} title={h.label} />
            {showHandleLabels && (
              <span
                className="absolute text-[10px] font-mono pointer-events-none"
                style={{ left: `${leftPct}%`, top: 4, transform: "translateX(-50%)", color: "var(--text-muted)", whiteSpace: "nowrap" }}
              >
                {h.label}
              </span>
            )}
          </span>
        );
      })}

      {/* Output handles */}
      {outputHandles.map((h, i) => {
        const handleShape = shapeMap[`${id}:${h.id}`]?.shape;
        const shapeStr = handleShape ? ` ${formatShape(handleShape, effectiveBatch)}` : "";
        // For Data nodes: first handle (x) goes bottom, remaining (y, mask, etc.) go right side
        const isDataNode = !!data.dataOutputs;
        const isFirstHandle = i === 0;
        if (isDataNode && !isFirstHandle) {
          // Right-side handle for y/target outputs — distribute evenly in 50-90% range
          const rightHandleCount = outputHandles.length - 1;
          const topPct = rightHandleCount === 1 ? 72 : 55 + (i - 1) * (35 / Math.max(rightHandleCount - 1, 1));
          return (
            <span key={`out-${h.id}`}>
              <Handle type="source" position={Position.Right} id={h.id}
                style={{ top: `${topPct}%` }} title={`${h.label}${shapeStr}`} />
              {showHandleLabels && (
                <span
                  className="absolute text-[10px] font-mono pointer-events-none"
                  style={{ right: 10, top: `${topPct}%`, transform: "translateY(-50%)", color: "var(--text-muted)", whiteSpace: "nowrap" }}
                >
                  {h.label}{shapeStr} →
                </span>
              )}
            </span>
          );
        }
        // Bottom handle (default for x / non-data nodes)
        const leftPct = isDataNode ? 50 : ((i + 1) / (outputHandles.length + 1)) * 100;
        return (
          <span key={`out-${h.id}`}>
            <Handle type="source" position={Position.Bottom} id={h.id}
              style={{ left: `${leftPct}%` }} title={`${h.label}${shapeStr}`} />
            {showHandleLabels && (
              <span
                className="absolute text-[10px] font-mono pointer-events-none"
                style={{ left: `${leftPct}%`, bottom: 4, transform: "translateX(-50%)", color: "var(--text-muted)", whiteSpace: "nowrap" }}
              >
                {h.label}{shapeStr}
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

function DatasetBadge({ nodeParams }: { nodeParams: Record<string, unknown> }) {
  const nodes = useGraphStore((s) => s.nodes);

  // Find the Data node in the graph — prefer it over training store default
  const graphDataNode = nodes.find((n) => n.data.moduleType === "Data");
  const trainingDatasetType = useTrainingStore((s) => s.config.dataset.type);

  const info = useMemo(() => {
    if (graphDataNode) {
      const source = String(graphDataNode.data.params.source ?? "");
      if (!source) return null;
      const resolved = resolveDataNodeShape(graphDataNode.data.params);
      const preset = DATA_PRESETS[source];
      if (resolved) {
        return { label: preset?.label ?? source, inputShape: resolved.inputShape };
      }
      // Custom source with no resolved shape — can't validate
      return { label: preset?.label ?? source, inputShape: null };
    }
    // Fall back to training store config
    const meta = DATASET_METADATA[trainingDatasetType];
    if (!meta) return null;
    return { label: trainingDatasetType, inputShape: meta.inputShape };
  }, [graphDataNode, trainingDatasetType]);

  if (!info) return null;

  const match = useMemo(() => {
    if (!info.inputShape) return null; // Can't validate — no known shape
    const raw = String(nodeParams.shape ?? "");
    let shape: number[];
    try { shape = JSON.parse(raw.replace(/\(/g, "[").replace(/\)/g, "]")); } catch { return null; }
    return JSON.stringify(shape) === JSON.stringify(info.inputShape);
  }, [nodeParams.shape, info?.inputShape]);

  if (match === null && info.inputShape === null) {
    // Custom dataset with unknown shape — show name only
    return (
      <div className="px-3 pb-1.5 text-[11px] font-medium" style={{ color: "var(--text-muted)" }}>
        {info.label}
      </div>
    );
  }

  if (match === null) return null;

  return (
    <div className="px-3 pb-1.5 text-[11px] font-medium" style={{ color: match ? "rgb(34, 197, 94)" : "rgb(245, 158, 11)" }}>
      {match ? `${info.label} \u2713` : `${info.label} expects ${JSON.stringify(info.inputShape)}`}
    </div>
  );
}

export default memo(TensorBloomNode);
