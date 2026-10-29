import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { useGraphStore } from "../store/graph-store";
import { useHistoryStore } from "../store/history-store";
import { useTrainingStore } from "../store/training-store";
import { formatShape } from "../lib/shape-inference";
import { formatParamCount } from "../lib/model-summary";
import { quickParamCount } from "../lib/node-summary";
import { FRAME_HEADER, type GroupNodeData } from "../lib/groups";

/** A layer group: a titled frame behind its layers when open, a single card in their place when collapsed */
function LayerGroupNode({ id, data }: NodeProps<Node<GroupNodeData>>) {
  const { group } = data;
  const selected = useGraphStore((s) => s.selectedGroupId === id);
  const renaming = useGraphStore((s) => s.renamingGroupId === id);
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const shapeMap = useGraphStore((s) => s.shapeMap);
  const shapeErrors = useGraphStore((s) => s.shapeErrors);
  const showBatch = useGraphStore((s) => s.shapeDisplayBatch);
  const batchSize = useTrainingStore((s) => s.config.batchSize);
  const batch = showBatch ? batchSize : undefined;

  const info = useMemo(() => {
    const inside = new Set(group.nodeIds);
    let params = 0;
    for (const n of nodes) if (inside.has(n.id)) params += quickParamCount(n.data.moduleType, n.data.params);
    // The shapes going in and out across the group's border
    const shapeOf = (nodeId: string, handle?: string | null) => (shapeMap[`${nodeId}:${handle}`] ?? shapeMap[nodeId])?.shape;
    const inEdge = edges.find((e) => inside.has(e.target) && !inside.has(e.source));
    const outEdge = edges.find((e) => inside.has(e.source) && !inside.has(e.target));
    const problems = group.nodeIds.filter((n) => shapeErrors[n]?.type === "root").length;
    return {
      params,
      inShape: inEdge ? shapeOf(inEdge.source, inEdge.sourceHandle) : undefined,
      outShape: outEdge ? shapeOf(outEdge.source, outEdge.sourceHandle) : undefined,
      problems,
    };
  }, [group.nodeIds, nodes, edges, shapeMap, shapeErrors]);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    useHistoryStore.getState().pushSnapshot();
    useGraphStore.getState().setGroupCollapsed(id, !group.collapsed);
  };

  const meta = `${data.layerCount} layer${data.layerCount === 1 ? "" : "s"}${info.params > 0 ? ` · ${formatParamCount(info.params)} params` : ""}`;
  const title = <GroupTitle id={id} name={group.name} renaming={renaming} />;

  if (group.collapsed) {
    return (
      <div
        className={`w-full h-full rounded-lg border-2 bg-surface-raised flex flex-col justify-center gap-1 px-3 cursor-grab ${selected ? "border-accent" : info.problems ? "border-danger" : "border-line"}`}
        style={{ boxShadow: "var(--node-shadow), 4px 4px 0 -1px var(--bg-secondary), 4px 4px 0 0 var(--border)" }}
        onDoubleClick={toggle}
      >
        <Handle type="target" position={Position.Top} id="in" isConnectable={false} />
        <div className="flex items-center gap-2 min-w-0">
          <GroupIcon />
          {title}
          <ToggleButton collapsed onClick={toggle} />
        </div>
        <div className="flex items-center gap-2 text-[11px] text-fg-muted tabular-nums min-w-0">
          <span className="truncate">{meta}</span>
          {info.problems > 0 && (
            <span className="ml-auto flex-shrink-0 flex items-center gap-1 text-danger font-medium">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {info.problems} problem{info.problems === 1 ? "" : "s"} inside
            </span>
          )}
        </div>
        {(info.inShape || info.outShape) && (
          <div className="text-[11px] font-mono text-fg-muted truncate">
            {info.inShape ? formatShape(info.inShape, batch) : "?"} → {info.outShape ? formatShape(info.outShape, batch) : "?"}
          </div>
        )}
        <Handle type="source" position={Position.Bottom} id="out" isConnectable={false} />
      </div>
    );
  }

  // Open: only the title bar takes the pointer, so the layers inside and the canvas behind stay usable
  return (
    <div
      className={`w-full h-full rounded-xl border-[1.5px] ${selected ? "border-accent" : "border-dashed border-line"}`}
      style={{ background: "color-mix(in srgb, var(--accent) 3%, transparent)", pointerEvents: "none" }}
    >
      <div
        className="group-drag-handle flex items-center gap-2 px-2.5 rounded-t-xl cursor-grab min-w-0"
        style={{ height: FRAME_HEADER, pointerEvents: "auto" }}
      >
        <ToggleButton collapsed={false} onClick={toggle} />
        {title}
        {/* The counts give way to the name in a narrow frame */}
        <span className="ml-auto min-w-0 truncate text-[11px] text-fg-muted tabular-nums" title={meta}>{meta}</span>
      </div>
    </div>
  );
}

/** The group's name; double-click it (or pick Rename) to edit it in place */
function GroupTitle({ id, name, renaming }: { id: string; name: string; renaming: boolean }) {
  const [value, setValue] = useState(name);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!renaming) return;
    setValue(name);
    // Wait a frame so the input exists and the click that started renaming is over
    requestAnimationFrame(() => { inputRef.current?.focus(); inputRef.current?.select(); });
  }, [renaming]); // eslint-disable-line react-hooks/exhaustive-deps -- only when renaming starts

  const finish = (save: boolean) => {
    const s = useGraphStore.getState();
    if (save && value.trim() && value.trim() !== name) {
      useHistoryStore.getState().pushSnapshot();
      s.renameGroup(id, value);
    }
    s.setRenamingGroupId(null);
  };

  if (renaming) {
    return (
      <input
        ref={inputRef}
        value={value}
        aria-label="Group name"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter") finish(true);
          if (e.key === "Escape") finish(false);
        }}
        onBlur={() => finish(true)}
        onMouseDown={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
        className="nodrag min-w-0 flex-1 px-1 text-[13px] font-semibold rounded outline-none bg-surface border border-accent text-fg"
        style={{ pointerEvents: "auto" }}
      />
    );
  }
  return (
    <span
      className="flex-shrink-0 max-w-[75%] truncate text-[13px] font-semibold text-fg cursor-text"
      title="Double-click to rename"
      onDoubleClick={(e) => { e.stopPropagation(); useGraphStore.getState().setRenamingGroupId(id); }}
    >
      {name}
    </span>
  );
}

function ToggleButton({ collapsed, onClick }: { collapsed: boolean; onClick: (e: React.MouseEvent) => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      aria-label={collapsed ? "Expand group" : "Collapse group"}
      aria-expanded={!collapsed}
      title={collapsed ? "Expand" : "Collapse"}
      className={`nodrag flex-shrink-0 w-5 h-5 rounded flex items-center justify-center text-fg-muted hover:text-fg hover:bg-surface-raised ${collapsed ? "ml-auto" : ""}`}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
        style={{ transform: collapsed ? "rotate(-90deg)" : undefined, transition: "transform 0.15s" }}>
        <polyline points="6 9 12 15 18 9" />
      </svg>
    </button>
  );
}

function GroupIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 text-accent-text" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="3" strokeDasharray="4 3" /><rect x="7" y="7" width="10" height="4" rx="1" /><rect x="7" y="13" width="10" height="4" rx="1" />
    </svg>
  );
}

export default memo(LayerGroupNode);
