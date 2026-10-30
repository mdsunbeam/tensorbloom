import { useState, useCallback, useMemo } from "react";
import { BaseEdge, getBezierPath, Position, type EdgeProps } from "@xyflow/react";
import { useGraphStore } from "../store/graph-store";
import { formatShape } from "../lib/shape-inference";

const R = 8;            // corner radius for orthogonal turns
const PAD = 30;         // base clearance outside node bounds
const LANE_GAP = 20;    // spacing between parallel channel lanes

export default function ShapeEdge(props: EdgeProps) {
  const { id, source, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, selected } = props;
  const [hovered, setHovered] = useState(false);

  const sourceError = useGraphStore((s) => s.shapeErrors[source]);
  const sourceShape = useGraphStore((s) => s.shapeMap[source]);
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);

  // ── Compute node bounds + lane assignments for all channel edges ──
  const { bounds, lane, side } = useMemo(() => {
    // Node bounds
    let left = Infinity, right = -Infinity;
    const nodeMap = new Map<string, typeof nodes[number]>();
    for (const n of nodes) {
      nodeMap.set(n.id, n);
      const w = n.measured?.width ?? (n.width as number | undefined) ?? 200;
      left = Math.min(left, n.position.x);
      right = Math.max(right, n.position.x + w);
    }
    const bds = { left, right };

    // Classify every edge as right-channel, left-channel, or normal.
    // Channel edges: skip connections (Bottom→Top, large dy, small dx)
    //                and Right→Right edges (Data.y → Loss.target)
    type CEdge = { id: string; side: "right" | "left"; minY: number };
    const channelEdges: CEdge[] = [];

    for (const e of edges) {
      const sn = nodeMap.get(e.source);
      const tn = nodeMap.get(e.target);
      if (!sn || !tn) continue;

      const sw = sn.measured?.width ?? (sn.width as number | undefined) ?? 200;
      const sh = sn.measured?.height ?? (sn.height as number | undefined) ?? 80;
      const tw = tn.measured?.width ?? (tn.width as number | undefined) ?? 200;

      // Data → Loss.target edges (both Right→Right for y, and Bottom→Right for x)
      const isDataSource = !!sn.data.dataOutputs;
      if (isDataSource && e.targetHandle === "target") {
        channelEdges.push({ id: e.id, side: "right", minY: sn.position.y });
        continue;
      }

      // Skip connection: detect by vertical gap between node CENTERS
      // (not handle positions, which vary by handle index on multi-input nodes)
      const srcCenterX = sn.position.x + sw / 2;
      const tgtCenterX = tn.position.x + tw / 2;
      const nodeCenterDx = Math.abs(tgtCenterX - srcCenterX);
      const nodeGapY = tn.position.y - (sn.position.y + sh);

      if (nodeGapY > 100 && nodeCenterDx < 150) {
        // Determine channel side from target handle ID:
        // Second handles ("b", "tgt", "input2") are right of center (67%)
        // First handles ("a", "src", "input1") are left of center (33%)
        const rightHandles = new Set(["b", "tgt", "input2", "positive", "negative"]);
        const side: "right" | "left" = rightHandles.has(e.targetHandle ?? "") ? "right" : "left";
        channelEdges.push({
          id: e.id,
          side,
          minY: sn.position.y + sh,
        });
      }
    }

    // Assign lanes per side: sort by minY, sequential lane index
    const rightEdges = channelEdges.filter((e) => e.side === "right").sort((a, b) => a.minY - b.minY);
    const leftEdges = channelEdges.filter((e) => e.side === "left").sort((a, b) => a.minY - b.minY);

    const laneMap = new Map<string, { lane: number; side: "right" | "left" }>();
    rightEdges.forEach((e, i) => laneMap.set(e.id, { lane: i, side: "right" }));
    leftEdges.forEach((e, i) => laneMap.set(e.id, { lane: i, side: "left" }));

    const info = laneMap.get(id);
    return {
      bounds: bds,
      lane: info?.lane ?? 0,
      side: info?.side ?? null,
    };
  }, [nodes, edges, id]);

  // ── Edge path computation ──
  let edgePath: string;
  let labelX: number | undefined;

  const dx = targetX - sourceX;
  const adx = Math.abs(dx);
  const dy = targetY - sourceY;

  if (sourcePosition === Position.Bottom && targetPosition === Position.Top) {
    if (side !== null) {
      // Skip connection — route through assigned lane outside all nodes
      const signX = side === "right" ? 1 : -1;
      const channelX = side === "right"
        ? bounds.right + PAD + lane * LANE_GAP
        : bounds.left - PAD - lane * LANE_GAP;
      const stub = 20;
      const r = Math.min(R, stub);
      const y1 = sourceY + stub;
      const y2 = targetY - stub;
      edgePath = `M ${sourceX} ${sourceY} `
        + `L ${sourceX} ${y1 - r} `
        + `Q ${sourceX} ${y1} ${sourceX + signX * r} ${y1} `
        + `L ${channelX - signX * r} ${y1} `
        + `Q ${channelX} ${y1} ${channelX} ${y1 + r} `
        + `L ${channelX} ${y2 - r} `
        + `Q ${channelX} ${y2} ${channelX - signX * r} ${y2} `
        + `L ${targetX + signX * r} ${y2} `
        + `Q ${targetX} ${y2} ${targetX} ${y2 + r} `
        + `L ${targetX} ${targetY}`;
    } else if (adx < 2) {
      // Aligned — straight vertical line
      edgePath = `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`;
    } else {
      // Adjacent layers / different columns — midpoint horizontal jog
      const midY = (sourceY + targetY) / 2;
      const r = Math.min(R, adx, Math.abs(dy) / 4);
      const signX = dx > 0 ? 1 : -1;
      edgePath = `M ${sourceX} ${sourceY} `
        + `L ${sourceX} ${midY - r} `
        + `Q ${sourceX} ${midY} ${sourceX + signX * r} ${midY} `
        + `L ${targetX - signX * r} ${midY} `
        + `Q ${targetX} ${midY} ${targetX} ${midY + r} `
        + `L ${targetX} ${targetY}`;
    }
  } else if (sourcePosition === Position.Bottom && targetPosition === Position.Right) {
    // Bottom→Right: e.g., Data.x → Loss.target (reconstruction)
    // Route through right-side channel: down → right to channel → down → left to target
    const channelX = bounds.right + PAD + lane * LANE_GAP;
    const stub = 20;
    const r = Math.min(R, stub);
    const y1 = sourceY + stub;
    edgePath = `M ${sourceX} ${sourceY} `
      + `L ${sourceX} ${y1 - r} `
      + `Q ${sourceX} ${y1} ${sourceX + r} ${y1} `
      + `L ${channelX - r} ${y1} `
      + `Q ${channelX} ${y1} ${channelX} ${y1 + r} `
      + `L ${channelX} ${targetY - r} `
      + `Q ${channelX} ${targetY} ${channelX - r} ${targetY} `
      + `L ${targetX} ${targetY}`;
  } else if (sourcePosition === Position.Right && targetPosition === Position.Right) {
    // Right→Right: assigned lane on the right margin
    const channelX = bounds.right + PAD + lane * LANE_GAP;
    const goingDown = targetY > sourceY;
    const d = goingDown ? R : -R;
    edgePath = `M ${sourceX} ${sourceY} `
      + `L ${channelX - R} ${sourceY} `
      + `Q ${channelX} ${sourceY} ${channelX} ${sourceY + d} `
      + `L ${channelX} ${targetY - d} `
      + `Q ${channelX} ${targetY} ${channelX - R} ${targetY} `
      + `L ${targetX} ${targetY}`;
    labelX = channelX;
  } else {
    // Fallback for unusual handle combos
    [edgePath] = getBezierPath({
      sourceX, sourceY, targetX, targetY,
      sourcePosition, targetPosition,
    });
  }

  // ── Error state ──
  const isRootError = sourceError?.type === "root";
  const isCascadedError = sourceError?.type === "cascaded";
  const hasError = isRootError || isCascadedError;
  const errorClass = isRootError ? "edge-error-root" : isCascadedError ? "edge-error-cascaded" : "";

  // Midpoint for hover label
  const midX = labelX ?? (sourceX + targetX) / 2;
  const midY = (sourceY + targetY) / 2;

  const onMouseEnter = useCallback(() => setHovered(true), []);
  const onMouseLeave = useCallback(() => setHovered(false), []);

  // Build hover label content
  let hoverLabel = "";
  if (sourceShape) {
    hoverLabel = formatShape(sourceShape.shape);
  }
  if (sourceError) {
    if (sourceError.receivedShape && sourceError.expectedShape) {
      hoverLabel = `${formatShape(sourceError.receivedShape)} \u2192 expects ${formatShape(sourceError.expectedShape)}`;
    } else if (sourceError.message) {
      hoverLabel = sourceError.message;
    }
  }

  return (
    <g onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      {/* Wider invisible hit-area for easier hover targeting */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={16}
        style={{ cursor: "pointer" }}
      />
      <BaseEdge id={id} path={edgePath} style={style} className={`${selected ? "selected" : ""} ${errorClass}`} />

      {/* Hover label at midpoint */}
      {hovered && hoverLabel && (
        <foreignObject
          x={midX - 100}
          y={midY - 14}
          width={200}
          height={28}
          style={{ pointerEvents: "none", overflow: "visible" }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              width: "100%",
              height: "100%",
            }}
          >
            <div
              style={{
                background: hasError ? (isRootError ? "var(--danger)" : "var(--warning)") : "var(--bg-elevated)",
                color: hasError ? "#fff" : "var(--text-primary)",
                fontSize: "9px",
                fontFamily: "monospace",
                padding: "2px 8px",
                borderRadius: "4px",
                whiteSpace: "nowrap",
                border: hasError ? "none" : "1px solid var(--border)",
                boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
                maxWidth: "200px",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {hoverLabel}
            </div>
          </div>
        </foreignObject>
      )}
    </g>
  );
}
