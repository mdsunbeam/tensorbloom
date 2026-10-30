import { useState, useCallback, useMemo } from "react";
import { BaseEdge, getBezierPath, Position, type EdgeProps } from "@xyflow/react";
import { useGraphStore } from "../store/graph-store";
import { formatShape } from "../lib/shape-inference";

const R = 14;           // corner radius for the turns around a block
const PAD = 30;         // clearance outside the whole graph, for Data → Loss target edges
const SKIP_PAD = 26;    // clearance beside the block a skip connection goes around
const LANE_GAP = 14;    // spacing between skip connections that run side by side

type Side = "right" | "left";

export default function ShapeEdge(props: EdgeProps) {
  const { id, source, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, selected } = props;
  const [hovered, setHovered] = useState(false);

  const sourceError = useGraphStore((s) => s.shapeErrors[source]);
  const sourceShape = useGraphStore((s) => s.shapeMap[source]);
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);

  // ── Which edges go around the graph, and where ──
  const { channelX, side, isSkip } = useMemo(() => {
    const nodeMap = new Map<string, typeof nodes[number]>();
    const size = (n: typeof nodes[number]) => ({
      w: n.measured?.width ?? (n.width as number | undefined) ?? 200,
      h: n.measured?.height ?? (n.height as number | undefined) ?? 80,
    });
    let right = -Infinity;
    for (const n of nodes) {
      nodeMap.set(n.id, n);
      right = Math.max(right, n.position.x + size(n).w);
    }

    // Channel edges: Data → Loss target edges run down the graph's right margin.
    // Skip connections (a long gap between nodes stacked in one column) run just beside the nodes they pass.
    type CEdge = { id: string; side: Side; top: number; bottom: number; skip: boolean; edge: number };
    const channel: CEdge[] = [];
    const rightHandles = new Set(["b", "tgt", "input2", "positive", "negative"]);

    for (const e of edges) {
      const sn = nodeMap.get(e.source);
      const tn = nodeMap.get(e.target);
      if (!sn || !tn) continue;
      const ss = size(sn), ts = size(tn);

      if (sn.data.dataOutputs && e.targetHandle === "target") {
        channel.push({ id: e.id, side: "right", top: sn.position.y, bottom: tn.position.y, skip: false, edge: right });
        continue;
      }

      // Detect by the gap between node CENTERS (handle positions vary on multi-input nodes)
      const nodeCenterDx = Math.abs(tn.position.x + ts.w / 2 - (sn.position.x + ss.w / 2));
      const top = sn.position.y + ss.h;
      const bottom = tn.position.y;
      if (bottom - top > 100 && nodeCenterDx < 150) {
        // Second handles ("b", "tgt", …) sit right of center, so their skip runs on the right
        const sd: Side = rightHandles.has(e.targetHandle ?? "") ? "right" : "left";
        // The outer edge of the nodes this connection passes, not of the whole graph
        let edgeX = sd === "right" ? -Infinity : Infinity;
        for (const n of nodes) {
          const ns = size(n);
          if (n.position.y + ns.h < sn.position.y || n.position.y > tn.position.y + ts.h) continue;
          const nx = n.position.x + ns.w / 2;
          if (Math.abs(nx - (sn.position.x + ss.w / 2)) > 200) continue;
          edgeX = sd === "right" ? Math.max(edgeX, n.position.x + ns.w) : Math.min(edgeX, n.position.x);
        }
        channel.push({ id: e.id, side: sd, top, bottom, skip: true, edge: edgeX });
      }
    }

    // Lanes: connections whose spans overlap on the same side sit next to each other; others share lane 0
    const laneOf = new Map<string, number>();
    for (const sd of ["right", "left"] as const) {
      const list = channel.filter((c) => c.side === sd).sort((a, b) => a.top - b.top);
      const laneEnds: number[] = [];
      for (const c of list) {
        let lane = laneEnds.findIndex((end) => end < c.top);
        if (lane === -1) { lane = laneEnds.length; laneEnds.push(c.bottom); } else laneEnds[lane] = c.bottom;
        laneOf.set(c.id, lane);
      }
    }

    const mine = channel.find((c) => c.id === id);
    const lane = laneOf.get(id) ?? 0;
    const sign = mine?.side === "left" ? -1 : 1;
    const pad = mine?.skip ? SKIP_PAD : PAD;
    return {
      side: mine?.side ?? null,
      isSkip: !!mine?.skip,
      channelX: mine ? mine.edge + sign * (pad + lane * LANE_GAP) : right + PAD,
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
      // Skip connection: out to the side of the block it skips, down, and back in
      const signX = side === "right" ? 1 : -1;
      const stub = 20;
      const r = Math.min(R, stub, Math.abs(channelX - sourceX) / 2, Math.abs(channelX - targetX) / 2);
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
      <BaseEdge id={id} path={edgePath} style={style} className={`${selected ? "selected" : ""} ${errorClass || (isSkip ? "edge-skip" : "")}`} />

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
