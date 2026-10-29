import { useMemo } from "react";
import type { Template } from "../lib/templates";
import { templateGraph } from "../lib/templates";
import { nodeSize } from "../lib/auto-layout";
import { CATEGORY_COLORS } from "../lib/registry";
import type { NodeCategory } from "../lib/types";

interface Bar { id: string; category: NodeCategory; x: number; y: number; w: number; h: number }
interface Preview { bars: Bar[]; links: string[]; width: number; height: number; layers: number }

const cache = new Map<string, Preview>();

/**
 * The template's real layout turned on its side, so the model reads left to right as a strip of
 * layer colors: what it's made of and how deep it is, at a glance.
 */
export function templatePreview(tpl: Template): Preview {
  const hit = cache.get(tpl.name);
  if (hit) return hit;
  const { nodes, edges } = templateGraph(tpl);
  const byId = new Map<string, Bar>();
  for (const n of nodes) {
    const { width, height } = nodeSize(n.data);
    // Rotate a quarter turn: the canvas's top-to-bottom flow becomes left to right
    byId.set(n.id, { id: n.id, category: n.data.category, x: n.position.y, y: n.position.x, w: height, h: width });
  }
  const bars = [...byId.values()];
  const minX = Math.min(...bars.map((b) => b.x));
  const minY = Math.min(...bars.map((b) => b.y));
  for (const b of bars) { b.x -= minX; b.y -= minY; }
  const links: string[] = [];
  for (const e of edges) {
    // Labels into the loss would run the length of the strip; the canvas draws those to the side
    if (e.targetHandle === "target") continue;
    const s = byId.get(e.source);
    const t = byId.get(e.target);
    if (!s || !t) continue;
    const x1 = s.x + s.w, y1 = s.y + s.h / 2, x2 = t.x, y2 = t.y + t.h / 2;
    const mid = (x1 + x2) / 2;
    links.push(`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`);
  }
  const preview = {
    bars,
    links,
    width: Math.max(...bars.map((b) => b.x + b.w)),
    height: Math.max(...bars.map((b) => b.y + b.h)),
    layers: nodes.length,
  };
  cache.set(tpl.name, preview);
  return preview;
}

export default function TemplatePreview({ template, height = 56, className = "" }: {
  template: Template;
  height?: number;
  className?: string;
}) {
  const p = useMemo(() => templatePreview(template), [template]);
  // Pad by a node's breadth so short strips don't stretch edge to edge
  const pad = 60;
  return (
    <svg
      viewBox={`${-pad} ${-pad} ${p.width + pad * 2} ${p.height + pad * 2}`}
      preserveAspectRatio="xMidYMid meet"
      className={`block w-full ${className}`}
      style={{ height }}
      aria-hidden="true"
    >
      {p.links.map((d, i) => (
        <path key={i} d={d} fill="none" stroke="var(--edge-color)" strokeWidth={10} strokeOpacity={0.7} />
      ))}
      {p.bars.map((b) => (
        <rect key={b.id} x={b.x} y={b.y} width={b.w} height={b.h} rx={14}
          fill={CATEGORY_COLORS[b.category] ?? "var(--node-reshape)"} />
      ))}
    </svg>
  );
}
