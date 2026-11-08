/**
 * 3D architecture diagrams for papers and slides.
 *
 * Turns the graph into a figure in the style researchers draw by hand: every layer is a shaded
 * block sized by its output tensor (channels set its depth along the flow, height and width its
 * face), BatchNorm / ReLU / pooling fold onto the layer before them as bands, skip connections
 * arc over the blocks they skip, and side branches run in lanes below the main path.
 *
 * The figure is built once as a list of flat 2D primitives (a "scene") that is then written out
 * as SVG or as a vector PDF, so both files look the same. Nothing here touches the DOM.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { NodeCategory } from "./types";
import type { ShapeMap } from "./shape-inference";
import type { LayerGroup } from "./groups";
import { MODULE_REGISTRY, CATEGORY_LABELS } from "./registry";

// ─── Options ────────────────────────────────────────────────────────────────

export type DiagramProjection = "oblique" | "isometric" | "flat";
export type DiagramPalette = "paper" | "tensorbloom" | "grayscale";
export type DiagramLabelMode = "type" | "name" | "none";

export interface DiagramOptions {
  projection: DiagramProjection;
  palette: DiagramPalette;
  /** Draw BatchNorm, activations, dropout and pooling as bands on the layer before them */
  foldLayers: boolean;
  /** Leave out Flatten, Permute, Squeeze and other layers that only reshape */
  hideReshape: boolean;
  /** Draw each collapsed group as one block named after the group */
  collapseGroups: boolean;
  /** Draw identical layers in a row (such as 12 encoder layers) as one stacked block marked ×12 */
  stackRepeats: boolean;
  labels: DiagramLabelMode;
  showShapes: boolean;
  showLegend: boolean;
  background: "white" | "transparent";
}

export const DEFAULT_DIAGRAM_OPTIONS: DiagramOptions = {
  projection: "oblique",
  palette: "paper",
  foldLayers: true,
  hideReshape: true,
  collapseGroups: true,
  stackRepeats: true,
  labels: "type",
  showShapes: true,
  showLegend: true,
  background: "white",
};

// ─── Scene ──────────────────────────────────────────────────────────────────

export type Pt = [number, number];

export type ScenePrim =
  | { kind: "poly"; pts: Pt[]; fill: string | null; stroke: string | null; width: number }
  | { kind: "line"; pts: Pt[]; stroke: string; width: number; dash?: number[] }
  | { kind: "text"; x: number; y: number; text: string; size: number; bold: boolean; color: string; anchor: "middle" | "start" };

export interface DiagramScene {
  width: number;
  height: number;
  background: string | null;
  prims: ScenePrim[];
  /** Number of blocks drawn; 0 means there was nothing to draw */
  blockCount: number;
}

// ─── Colors ─────────────────────────────────────────────────────────────────

const PALETTES: Record<DiagramPalette, Record<NodeCategory, string>> = {
  paper: {
    io: "#B8C4CE", data: "#B8C4CE",
    conv: "#F3C969", vision: "#F3C969", pool: "#E07A6F",
    norm: "#9BC88B", activation: "#E3995B", dropout: "#C9C9C9",
    linear: "#7EA6DD", embedding: "#78C2B4",
    recurrent: "#B99BD8", transformer: "#E69BC0",
    reshape: "#D5DBE0", math: "#FFFFFF", loss: "#E57373",
  },
  tensorbloom: {
    io: "#2DB3A6", data: "#2DB3A6",
    conv: "#9B72F0", vision: "#9B72F0", pool: "#B497F4",
    norm: "#84B83A", activation: "#E6A03A", dropout: "#A7C46B",
    linear: "#4F86EE", embedding: "#7BA6F3",
    recurrent: "#E55497", transformer: "#E55497",
    reshape: "#8592A6", math: "#FFFFFF", loss: "#E35252",
  },
  grayscale: {
    io: "#C4C4C4", data: "#C4C4C4",
    conv: "#DADADA", vision: "#DADADA", pool: "#9E9E9E",
    norm: "#EFEFEF", activation: "#7D7D7D", dropout: "#B5B5B5",
    linear: "#BDBDBD", embedding: "#CFCFCF",
    recurrent: "#AAAAAA", transformer: "#AAAAAA",
    reshape: "#E6E6E6", math: "#FFFFFF", loss: "#8A8A8A",
  },
};

const INK = "#1F2937";
const INK_MUTED = "#6B7280";
const EDGE_INK = "#4B5563";

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  return "#" + [r, g, b].map((v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join("").toUpperCase();
}

/** Mixes a color toward white (amount > 0) or black (amount < 0) */
export function shade(hex: string, amount: number): string {
  const [r, g, b] = hexToRgb(hex);
  const t = amount > 0 ? 255 : 0;
  const a = Math.abs(amount);
  return rgbToHex([r + (t - r) * a, g + (t - g) * a, b + (t - b) * a]);
}

// ─── Projection ─────────────────────────────────────────────────────────────

type V3 = [number, number, number];

interface Projection {
  /** World (x along the flow, y up, z away from the viewer) to screen (y down) */
  project: (p: V3) => Pt;
  /** Points from the scene toward the viewer; faces whose normal agrees with it are visible */
  view: V3;
}

const COS30 = Math.cos(Math.PI / 6);
const SIN30 = 0.5;
const OBL_A = 0.55;
const OBL_ANGLE = (38 * Math.PI) / 180;

function makeProjection(kind: DiagramProjection): Projection {
  if (kind === "isometric") {
    return {
      project: ([x, y, z]) => [(x + z) * COS30, (x - z) * SIN30 - y],
      view: [1, 1, -1],
    };
  }
  if (kind === "flat") {
    return { project: ([x, y]) => [x, -y], view: [0, 0, -1] };
  }
  const cx = OBL_A * Math.cos(OBL_ANGLE);
  const sy = OBL_A * Math.sin(OBL_ANGLE);
  return {
    project: ([x, y, z]) => [x + z * cx, -y - z * sy],
    view: [cx, sy, -1],
  };
}

// ─── Labels & sizes ─────────────────────────────────────────────────────────

function kernelText(v: unknown): string {
  if (Array.isArray(v)) return v.length && v.every((n) => n === v[0]) ? `${v[0]}×${v[0]}` : v.join("×");
  if (typeof v === "number") return `${v}×${v}`;
  return "";
}

/** A short, paper-style name for a layer, such as "Conv 3×3" or "FC" */
export function shortLayerLabel(data: NodeData): string {
  const mt = data.moduleType;
  const p = data.params ?? {};
  if (mt === "Data" || mt === "Input") return data.label && data.label !== "Data" ? data.label : "Input";
  if (/^Conv[123]d$/.test(mt)) {
    const s = Number(p.stride ?? 1);
    return `Conv ${mt === "Conv1d" ? String(p.kernel_size ?? "") : kernelText(p.kernel_size)}${s > 1 ? ` /${s}` : ""}`.trim();
  }
  if (/^ConvTranspose[123]d$/.test(mt)) return `Deconv ${mt === "ConvTranspose1d" ? String(p.kernel_size ?? "") : kernelText(p.kernel_size)}`.trim();
  if (mt === "Linear") return "FC";
  if (/^BatchNorm/.test(mt)) return "BN";
  if (mt === "LayerNorm") return "LN";
  if (mt === "GroupNorm") return "GN";
  if (/^Dropout/.test(mt)) return "Dropout";
  if (/^MaxPool/.test(mt)) return "MaxPool";
  if (/^AvgPool/.test(mt)) return "AvgPool";
  if (/^AdaptiveAvgPool/.test(mt)) return "GAP";
  if (/^AdaptiveMaxPool/.test(mt)) return "GMP";
  if (mt === "Embedding") return "Embed";
  if (mt === "TransformerEncoderLayer") return "Encoder";
  if (mt === "TransformerDecoderLayer") return "Decoder";
  if (mt === "TransformerEncoder") return "Encoder stack";
  if (mt === "TransformerDecoder") return "Decoder stack";
  if (mt === "MultiheadAttention") return "Attention";
  return MODULE_REGISTRY[mt]?.label ?? mt;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const log2 = (n: number) => Math.log2(Math.max(1, n));
const chanSize = (c: number) => clamp(3 + 3.6 * log2(c + 1), 4, 48);
const spatialSize = (n: number) => clamp(8 + 15 * log2(n), 10, 140);
const featSize = (n: number) => clamp(16 + 11 * log2(n), 16, 140);

interface BoxSize { dx: number; h: number; w: number }

/** Block size from an output shape that includes the batch dimension */
export function blockSize(shape: number[] | null, category: NodeCategory): BoxSize {
  const d = shape ? shape.slice(1) : [];
  if (d.length >= 3) {
    const [c, h, w] = [d[0], d[d.length - 2], d[d.length - 1]];
    return { dx: chanSize(c), h: spatialSize(h), w: spatialSize(w) };
  }
  if (d.length === 2) {
    if (category === "conv" || category === "pool" || category === "vision") {
      return { dx: chanSize(d[0]), h: spatialSize(d[1]), w: 10 };
    }
    // [tokens, features]
    return { dx: 10, h: featSize(d[1]), w: spatialSize(d[0]) };
  }
  if (d.length === 1) return { dx: 8, h: featSize(d[0]), w: 8 };
  return { dx: 14, h: 40, w: 40 };
}

export function shapeCaption(shape: number[] | null): string {
  if (!shape || shape.length < 2) return "";
  return shape.slice(1).join("×");
}

// Helvetica advance widths (1/1000 em) for ASCII 32–126, used to size labels and to center PDF text
const HELV = [
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
  556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
  1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
  667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
  333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
  556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
];

export function textWidth(text: string, size: number, bold = false): number {
  let w = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    w += c >= 32 && c <= 126 ? HELV[c - 32] : 584;
  }
  return (w * size * (bold ? 1.07 : 1)) / 1000;
}

// ─── Block graph ────────────────────────────────────────────────────────────

interface Part {
  category: NodeCategory;
  shape: number[] | null;
  size: BoxSize;
  /** Empty space before this part, used to draw repeated layers as a stack */
  gap?: number;
}

interface Block {
  id: string;
  nodeIds: string[];
  title: string;
  folded: string[];
  parts: Part[];
  shape: number[] | null;
  /** An element-wise op (Add, Multiply, Concatenate) drawn as a small circle */
  op?: "+" | "×" | "C" | "@";
  /** How many identical layers in a row this block stands for */
  repeat: number;
}

const FOLDABLE = new Set<NodeCategory>(["activation", "norm", "dropout"]);
// Bands go on layers people draw that way; a final LayerNorm after a transformer stack stays its own slab
const HOSTS = new Set<NodeCategory>(["conv", "vision", "linear", "pool", "norm", "activation"]);
const OPS: Record<string, Block["op"]> = { Add: "+", Multiply: "×", Concatenate: "C", MatMul: "@" };

function nodeShape(shapes: ShapeMap, n: Node<NodeData>): number[] | null {
  if (n.data.moduleType === "Data" || n.data.dataOutputs) {
    return shapes[`${n.id}:x`]?.shape ?? shapes[n.id]?.shape ?? null;
  }
  return shapes[n.id]?.shape ?? shapes[`${n.id}:out`]?.shape ?? null;
}

function topoOrder(ids: string[], edges: { s: string; t: string }[]): string[] {
  const indeg = new Map(ids.map((id) => [id, 0]));
  const out = new Map<string, string[]>();
  for (const e of edges) {
    indeg.set(e.t, (indeg.get(e.t) ?? 0) + 1);
    out.set(e.s, [...(out.get(e.s) ?? []), e.t]);
  }
  const queue = ids.filter((id) => indeg.get(id) === 0);
  const order: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const t of out.get(id) ?? []) {
      indeg.set(t, indeg.get(t)! - 1);
      if (indeg.get(t) === 0) queue.push(t);
    }
  }
  // A cycle shouldn't happen; keep whatever is left in its original order
  for (const id of ids) if (!order.includes(id)) order.push(id);
  return order;
}

function buildBlocks(
  nodes: Node<NodeData>[],
  edges: Edge[],
  shapes: ShapeMap,
  groups: LayerGroup[],
  opts: DiagramOptions,
): { blocks: Map<string, Block>; links: { s: string; t: string }[]; order: string[] } {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const isBypass = (n: Node<NodeData>) =>
    n.data.moduleType === "Output" || (opts.hideReshape && n.data.category === "reshape");
  const isDropped = (n: Node<NodeData>) => n.data.category === "loss";

  // Node-level links, with the target labels and loss layers dropped and reshapes bypassed
  let links = edges
    .filter((e) => byId.has(e.source) && byId.has(e.target))
    .filter((e) => !(byId.get(e.source)!.data.moduleType === "Data" && e.sourceHandle && e.sourceHandle !== "x"))
    .map((e) => ({ s: e.source, t: e.target, h: e.targetHandle ?? "" }));
  const kept = nodes.filter((n) => !isDropped(n)).map((n) => n.id);
  links = links.filter((l) => kept.includes(l.s) && kept.includes(l.t));
  for (const id of topoOrder(kept, links)) {
    const n = byId.get(id)!;
    if (!isBypass(n)) continue;
    const ins = links.filter((l) => l.t === id);
    const outs = links.filter((l) => l.s === id);
    links = links.filter((l) => l.s !== id && l.t !== id);
    for (const i of ins) for (const o of outs) links.push({ s: i.s, t: o.t, h: o.h });
  }
  const visible = kept.filter((id) => !isBypass(byId.get(id)!));

  const blocks = new Map<string, Block>();
  const owner = new Map<string, string>();
  const makePart = (n: Node<NodeData>): Part => {
    const shape = nodeShape(shapes, n);
    const size = blockSize(shape, n.data.category);
    // A BN or ReLU on its own is a thin slab, not a layer-sized block
    if (FOLDABLE.has(n.data.category)) size.dx = Math.max(5, Math.min(10, size.dx * 0.3));
    return { category: n.data.category, shape, size };
  };
  const titleOf = (n: Node<NodeData>) => (opts.labels === "name" ? n.data.label : shortLayerLabel(n.data));

  // Collapsed groups become one block
  if (opts.collapseGroups) {
    const order = topoOrder(visible, links);
    for (const g of groups) {
      if (!g.collapsed) continue;
      const members = order.filter((id) => g.nodeIds.includes(id));
      if (!members.length) continue;
      const exits = members.filter((id) => links.some((l) => l.s === id && !g.nodeIds.includes(l.t)));
      const last = byId.get(exits[exits.length - 1] ?? members[members.length - 1])!;
      const counts = new Map<NodeCategory, number>();
      for (const id of members) {
        const c = byId.get(id)!.data.category;
        if (!FOLDABLE.has(c)) counts.set(c, (counts.get(c) ?? 0) + 1);
      }
      const category = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? last.data.category;
      const shape = nodeShape(shapes, last);
      const bid = `group:${g.id}`;
      blocks.set(bid, { id: bid, nodeIds: members, title: g.name, folded: [], parts: [{ category, shape, size: blockSize(shape, category) }], shape, repeat: 1 });
      for (const id of members) owner.set(id, bid);
    }
  }
  for (const id of visible) {
    if (owner.has(id)) continue;
    const n = byId.get(id)!;
    const op = n.data.category === "math" ? OPS[n.data.moduleType] : undefined;
    blocks.set(id, { id, nodeIds: [id], title: titleOf(n), folded: [], parts: [makePart(n)], shape: nodeShape(shapes, n), op, repeat: 1 });
    owner.set(id, id);
  }

  // Block-level links, without duplicates or links inside a block
  const seen = new Set<string>();
  let blinks: { s: string; t: string; h: string }[] = [];
  for (const l of links) {
    const s = owner.get(l.s), t = owner.get(l.t);
    if (!s || !t || s === t || seen.has(`${s}>${t}`)) continue;
    seen.add(`${s}>${t}`);
    blinks.push({ s, t, h: l.h });
  }

  // Fold BN / activation / dropout / pooling onto a single-output layer just before them
  if (opts.foldLayers) {
    for (const id of topoOrder([...blocks.keys()], blinks)) {
      const b = blocks.get(id);
      if (!b || b.op || b.nodeIds.length !== 1 || id.startsWith("group:")) continue;
      const cat = b.parts[0].category;
      const ins = blinks.filter((l) => l.t === id);
      if (ins.length !== 1) continue;
      const host = blocks.get(ins[0].s)!;
      if (host.op || blinks.filter((l) => l.s === host.id).length !== 1) continue;
      const hostCat = host.parts[0].category;
      const prev = host.parts[host.parts.length - 1];
      let part: Part | null;
      if (cat === "dropout") {
        // Dropout doesn't change the tensor, so it gets a mention in the caption but no band
        part = null;
      } else if (FOLDABLE.has(cat) && HOSTS.has(hostCat)) {
        part = { ...b.parts[0], size: { ...prev.size, dx: Math.max(3, Math.min(8, host.parts[0].size.dx * 0.22)) } };
      } else if (cat === "pool" && (hostCat === "conv" || hostCat === "vision") && !/^Adaptive/.test(byId.get(b.nodeIds[0])!.data.moduleType)) {
        part = { ...b.parts[0], size: { ...b.parts[0].size, dx: Math.max(4, prev.size.dx * 0.5) } };
      } else continue;
      if (part) host.parts.push(part);
      host.nodeIds.push(...b.nodeIds);
      host.folded.push(b.title);
      host.shape = b.shape ?? host.shape;
      blocks.delete(id);
      blinks = blinks
        .filter((l) => l.t !== id)
        .map((l) => (l.s === id ? { ...l, s: host.id } : l));
    }
  }

  // Runs of identical layers become one stacked block
  if (opts.stackRepeats) {
    const sig = (b: Block) => b.op || b.id.startsWith("group:") ? null
      : [b.title, b.folded.join(","), shapeCaption(b.shape), b.parts.map((p) => p.category).join(",")].join("|");
    for (const id of topoOrder([...blocks.keys()], blinks)) {
      const a = blocks.get(id);
      if (!a || sig(a) === null) continue;
      for (;;) {
        const outs = blinks.filter((l) => l.s === a.id);
        if (outs.length !== 1) break;
        const b = blocks.get(outs[0].t)!;
        if (sig(b) !== sig(a) || blinks.filter((l) => l.t === b.id).length !== 1) break;
        a.repeat += b.repeat;
        a.nodeIds.push(...b.nodeIds);
        blocks.delete(b.id);
        blinks = blinks.filter((l) => l.t !== b.id).map((l) => (l.s === b.id ? { ...l, s: a.id } : l));
      }
    }
    for (const b of blocks.values()) {
      if (b.repeat < 2) continue;
      const one = b.parts;
      b.parts = [];
      for (let k = 0; k < Math.min(b.repeat, 3); k++) b.parts.push(...one.map((p, i) => ({ ...p, gap: k > 0 && i === 0 ? 4 : 0 })));
      b.title = `${b.title} ×${b.repeat}`;
    }
  }

  return { blocks, links: blinks, order: topoOrder([...blocks.keys()], blinks) };
}

// ─── Geometry helpers ───────────────────────────────────────────────────────

function convexHull(points: Pt[]): Pt[] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: Pt[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function insideConvex(p: Pt, hull: Pt[]): boolean {
  if (hull.length < 3) return false;
  let sign = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    const c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (Math.abs(c) < 1e-9) continue;
    const s = Math.sign(c);
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

/** Where the segment p→q leaves a convex polygon that contains p */
function exitPoint(p: Pt, q: Pt, hull: Pt[]): Pt {
  let best = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    const r: Pt = [q[0] - p[0], q[1] - p[1]];
    const s: Pt = [b[0] - a[0], b[1] - a[1]];
    const den = r[0] * s[1] - r[1] * s[0];
    if (Math.abs(den) < 1e-9) continue;
    const t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / den;
    const u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den;
    if (t >= 0 && t <= 1 && u >= -1e-6 && u <= 1 + 1e-6) best = Math.max(best, t);
  }
  return [p[0] + (q[0] - p[0]) * best, p[1] + (q[1] - p[1]) * best];
}

/** Cuts off the start of a polyline that lies inside a convex polygon */
function trimStart(pts: Pt[], hull: Pt[]): Pt[] {
  if (!insideConvex(pts[0], hull)) return pts;
  for (let i = 0; i < pts.length - 1; i++) {
    if (insideConvex(pts[i + 1], hull)) continue;
    return [exitPoint(pts[i], pts[i + 1], hull), ...pts.slice(i + 1)];
  }
  return pts;
}

const trimEnd = (pts: Pt[], hull: Pt[]) => trimStart([...pts].reverse(), hull).reverse();

function circlePoly(c: Pt, r: number, n = 20): Pt[] {
  return Array.from({ length: n }, (_, i) => [c[0] + r * Math.cos((2 * Math.PI * i) / n), c[1] + r * Math.sin((2 * Math.PI * i) / n)] as Pt);
}

// ─── Scene construction ─────────────────────────────────────────────────────

interface Placed {
  block: Block;
  rank: number;
  lane: number;
  x0: number; // world x where the block starts
  x1: number; // world x where it ends
  y: number; // world y of the lane's axis
  hull: Pt[];
  top: number; // highest world y of the block
  bottom: number;
  /** Extra height the block's back edge adds on screen */
  rise: number;
}

const OP_R = 9;
const GAP = 30;
const TITLE_SIZE = 11;
const SUB_SIZE = 9.5;

function captionLines(b: Block, opts: DiagramOptions): { text: string; size: number; bold: boolean; color: string }[] {
  if (b.op) return [];
  const lines: { text: string; size: number; bold: boolean; color: string }[] = [];
  if (opts.labels !== "none") {
    lines.push({ text: b.title, size: TITLE_SIZE, bold: true, color: INK });
    if (b.folded.length) lines.push({ text: "+ " + b.folded.join(" · "), size: SUB_SIZE, bold: false, color: INK_MUTED });
  }
  if (opts.showShapes) {
    const s = shapeCaption(b.shape);
    if (s) lines.push({ text: s, size: SUB_SIZE, bold: false, color: INK_MUTED });
  }
  return lines;
}

const totalDx = (b: Block) => (b.op ? OP_R * 2 : b.parts.reduce((a, p) => a + (p.gap ?? 0) + p.size.dx, 0));
const maxH = (b: Block) => (b.op ? OP_R * 2 : Math.max(...b.parts.map((p) => p.size.h)));

/** Builds the diagram's primitives from the graph */
export function buildDiagramScene(
  nodes: Node<NodeData>[],
  edges: Edge[],
  shapes: ShapeMap,
  groups: LayerGroup[],
  options: Partial<DiagramOptions> = {},
): DiagramScene {
  const opts = { ...DEFAULT_DIAGRAM_OPTIONS, ...options };
  const proj = makeProjection(opts.projection);
  const palette = PALETTES[opts.palette];
  const { blocks, links, order } = buildBlocks(nodes, edges, shapes, groups, opts);
  const prims: ScenePrim[] = [];
  const background = opts.background === "white" ? "#FFFFFF" : null;
  if (!blocks.size) return { width: 0, height: 0, background, prims, blockCount: 0 };

  const preds = new Map<string, string[]>();
  const succs = new Map<string, string[]>();
  for (const l of links) {
    preds.set(l.t, [...(preds.get(l.t) ?? []), l.s]);
    succs.set(l.s, [...(succs.get(l.s) ?? []), l.t]);
  }

  // Main path: the longest chain of blocks
  const dist = new Map<string, number>();
  for (const id of order) dist.set(id, Math.max(0, ...(preds.get(id) ?? []).map((p) => dist.get(p)! + 1)));
  let end = order[0];
  for (const id of order) if (dist.get(id)! > dist.get(end)!) end = id;
  const main = new Set<string>([end]);
  for (let cur = end; (preds.get(cur) ?? []).length; ) {
    const ps = preds.get(cur)!;
    cur = ps.reduce((a, b) => (dist.get(b)! > dist.get(a)! ? b : a));
    main.add(cur);
  }

  // Ranks: the main path as early as possible, side branches as late as possible, next to where they join
  const rank = new Map<string, number>();
  for (const id of order) rank.set(id, dist.get(id)!);
  for (const id of [...order].reverse()) {
    if (main.has(id)) continue;
    const ss = succs.get(id) ?? [];
    if (ss.length) rank.set(id, Math.max(dist.get(id)!, Math.min(...ss.map((s) => rank.get(s)!)) - 1));
  }

  // Lanes: 0 for the main path, side branches below it, keeping a branch in one lane when it can
  const lane = new Map<string, number>();
  const used = new Map<number, Set<number>>();
  const take = (r: number, l: number) => { if (!used.has(r)) used.set(r, new Set()); used.get(r)!.add(l); };
  const free = (r: number, l: number) => !used.get(r)?.has(l);
  for (const id of order) if (main.has(id)) { lane.set(id, 0); take(rank.get(id)!, 0); }
  for (const id of order) {
    if (main.has(id)) continue;
    const r = rank.get(id)!;
    const sidePred = (preds.get(id) ?? []).find((p) => !main.has(p) && lane.has(p));
    let l = sidePred ? lane.get(sidePred)! : 1;
    if (!free(r, l)) { l = 1; while (!free(r, l)) l++; }
    lane.set(id, l);
    take(r, l);
  }

  // Lane heights and their axes
  const laneCount = Math.max(...lane.values()) + 1;
  const laneHalf = Array.from({ length: laneCount }, () => 10);
  const laneCaption = Array.from({ length: laneCount }, () => 0);
  for (const [id, b] of blocks) {
    const l = lane.get(id)!;
    laneHalf[l] = Math.max(laneHalf[l], maxH(b) / 2);
    laneCaption[l] = Math.max(laneCaption[l], captionLines(b, opts).length * 13);
  }
  const laneY: number[] = [0];
  for (let l = 1; l < laneCount; l++) laneY[l] = laneY[l - 1] - laneHalf[l - 1] - laneCaption[l - 1] - laneHalf[l] - 34;

  // Columns: each rank starts where the previous one's blocks and captions end on screen
  const blockCorners = (b: Block, x0: number, y: number): V3[] => {
    const pts: V3[] = [];
    let x = x0;
    for (const p of b.op ? [{ size: { dx: OP_R * 2, h: OP_R * 2, w: 0 }, gap: 0 }] : b.parts) {
      x += p.gap ?? 0;
      for (const xx of [x, x + p.size.dx]) for (const yy of [y - p.size.h / 2, y + p.size.h / 2]) for (const zz of [-p.size.w / 2, p.size.w / 2]) pts.push([xx, yy, zz]);
      x += p.size.dx;
    }
    return pts;
  };
  const px = proj.project([1, 0, 0])[0];
  // How far up the screen a block's back edge sits per unit of depth, so arcs clear it
  const zRise = Math.max(0, -proj.project([0, 0, 1])[1]);
  const maxRank = Math.max(...rank.values());
  const colStart: number[] = [];
  /** World x of the middle of the empty gap before each column, where links change lanes */
  const gapBefore: number[] = [];
  let cursor = 0;
  for (let r = 0; r <= maxRank; r++) {
    const ids = [...blocks.keys()].filter((id) => rank.get(id) === r);
    let lo = 0, hi = 0, maxDx = 0;
    for (const id of ids) {
      const b = blocks.get(id)!;
      const dx = totalDx(b);
      maxDx = Math.max(maxDx, dx);
      const xs = blockCorners(b, -dx / 2, laneY[lane.get(id)!]).map((c) => proj.project(c)[0]);
      const cap = Math.max(0, ...captionLines(b, opts).map((c) => textWidth(c.text, c.size, c.bold)));
      lo = Math.min(lo, ...xs, -cap / 2);
      hi = Math.max(hi, ...xs, cap / 2);
    }
    // World x of this column's center
    const center = r === 0 ? 0 : cursor - lo / px;
    colStart[r] = center;
    gapBefore[r] = cursor - GAP / (2 * px);
    cursor = center + (hi + GAP) / px;
    void maxDx;
  }

  // Place blocks
  const placed = new Map<string, Placed>();
  for (const [id, b] of blocks) {
    const r = rank.get(id)!, l = lane.get(id)!;
    const dx = totalDx(b);
    const x0 = colStart[r] - dx / 2;
    const y = laneY[l];
    const hull = b.op ? circlePoly(proj.project([x0 + OP_R, y, 0]), OP_R) : convexHull(blockCorners(b, x0, y).map(proj.project));
    const maxW = b.op ? 0 : Math.max(...b.parts.map((p) => p.size.w));
    placed.set(id, { block: b, rank: r, lane: l, x0, x1: x0 + dx, y, hull, top: y + maxH(b) / 2, bottom: y - maxH(b) / 2, rise: (maxW / 2) * zRise });
  }

  // Blocks, far ones first
  const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  interface Drawable { depth: number; draw: () => void }
  const drawables: Drawable[] = [];
  const stroke = (fill: string) => shade(fill, -0.45);
  for (const p of placed.values()) {
    const b = p.block;
    if (b.op) {
      const c = proj.project([p.x0 + OP_R, p.y, 0]);
      drawables.push({
        depth: dot([p.x0 + OP_R, p.y, 0], proj.view),
        draw: () => {
          prims.push({ kind: "poly", pts: circlePoly(c, OP_R, 32), fill: "#FFFFFF", stroke: EDGE_INK, width: 1.1 });
          const s = 4.5;
          if (b.op === "+") {
            prims.push({ kind: "line", pts: [[c[0] - s, c[1]], [c[0] + s, c[1]]], stroke: EDGE_INK, width: 1.3 });
            prims.push({ kind: "line", pts: [[c[0], c[1] - s], [c[0], c[1] + s]], stroke: EDGE_INK, width: 1.3 });
          } else if (b.op === "×") {
            const d = s * 0.75;
            prims.push({ kind: "line", pts: [[c[0] - d, c[1] - d], [c[0] + d, c[1] + d]], stroke: EDGE_INK, width: 1.3 });
            prims.push({ kind: "line", pts: [[c[0] - d, c[1] + d], [c[0] + d, c[1] - d]], stroke: EDGE_INK, width: 1.3 });
          } else {
            prims.push({ kind: "text", x: c[0], y: c[1] + 3.6, text: b.op!, size: 10, bold: true, color: EDGE_INK, anchor: "middle" });
          }
        },
      });
      continue;
    }
    let x = p.x0;
    for (const part of b.parts) {
      x += part.gap ?? 0;
      const { dx, h, w } = part.size;
      const xa = x, xb = x + dx, ya = p.y - h / 2, yb = p.y + h / 2, za = -w / 2, zb = w / 2;
      const base = palette[part.category] ?? palette.reshape;
      const faces: { n: V3; pts: V3[]; fill: string }[] = [
        { n: [0, 0, -1], pts: [[xa, ya, za], [xb, ya, za], [xb, yb, za], [xa, yb, za]], fill: base },
        { n: [0, 0, 1], pts: [[xa, ya, zb], [xb, ya, zb], [xb, yb, zb], [xa, yb, zb]], fill: base },
        { n: [0, 1, 0], pts: [[xa, yb, za], [xb, yb, za], [xb, yb, zb], [xa, yb, zb]], fill: shade(base, 0.38) },
        { n: [0, -1, 0], pts: [[xa, ya, za], [xb, ya, za], [xb, ya, zb], [xa, ya, zb]], fill: shade(base, -0.25) },
        { n: [1, 0, 0], pts: [[xb, ya, za], [xb, yb, za], [xb, yb, zb], [xb, ya, zb]], fill: shade(base, -0.14) },
        { n: [-1, 0, 0], pts: [[xa, ya, za], [xa, yb, za], [xa, yb, zb], [xa, ya, zb]], fill: shade(base, -0.14) },
      ];
      const center: V3 = [(xa + xb) / 2, p.y, 0];
      drawables.push({
        depth: dot(center, proj.view),
        draw: () => {
          for (const f of faces) {
            if (dot(f.n, proj.view) <= 1e-9) continue;
            prims.push({ kind: "poly", pts: f.pts.map(proj.project), fill: f.fill, stroke: stroke(base), width: 0.8 });
          }
        },
      });
      x = xb;
    }
  }
  drawables.sort((a, b) => a.depth - b.depth).forEach((d) => d.draw());

  // Connections, routed in world space so they follow the projection, then cut at the blocks' outlines
  const arcLevels: { r0: number; r1: number; level: number }[] = [];
  const midGap = (p: Placed, side: "after" | "before") =>
    side === "after"
      ? (gapBefore[p.rank + 1] ?? p.x1 + GAP / (2 * px))
      : (p.rank > 0 ? gapBefore[p.rank] : p.x0 - GAP / (2 * px));
  for (const l of links) {
    const a = placed.get(l.s)!, b = placed.get(l.t)!;
    let world: V3[];
    const between = [...placed.values()].filter((p) => p.lane === a.lane && p.rank > a.rank && p.rank < b.rank);
    if (a.lane === b.lane && !between.length) {
      world = [[a.x1, a.y, 0], [b.x0, b.y, 0]];
    } else if (a.lane === b.lane && a.lane === 0) {
      const level = arcLevels.filter((s) => s.r0 < b.rank && s.r1 > a.rank).reduce((m, s) => Math.max(m, s.level + 1), 0);
      arcLevels.push({ r0: a.rank, r1: b.rank, level });
      const yArc = Math.max(a.top + a.rise, b.top + b.rise, ...between.map((p) => p.top + p.rise)) + 18 + level * 14;
      const ax = (a.x0 + a.x1) / 2, bx = (b.x0 + b.x1) / 2;
      world = [[ax, a.top, 0], [ax, yArc, 0], [bx, yArc, 0], [bx, b.top, 0]];
    } else if (a.lane === b.lane) {
      const yArc = Math.min(a.bottom, b.bottom, ...between.map((p) => p.bottom)) - 40;
      const ax = (a.x0 + a.x1) / 2, bx = (b.x0 + b.x1) / 2;
      world = [[ax, a.bottom, 0], [ax, yArc, 0], [bx, yArc, 0], [bx, b.bottom, 0]];
    } else if (a.lane < b.lane) {
      // Down into a side branch: leave just after the source, drop between columns, run along the branch's lane
      const xm = midGap(a, "after");
      world = [[a.x1, a.y, 0], [xm, a.y, 0], [xm, b.y, 0], [b.x0, b.y, 0]];
    } else {
      // Back up from a side branch: run along its lane, then rise just before the target
      const xm = midGap(b, "before");
      world = [[a.x1, a.y, 0], [xm, a.y, 0], [xm, b.y, 0], [b.x0, b.y, 0]];
    }
    let pts = world.map(proj.project);
    pts = trimStart(pts, a.hull);
    pts = trimEnd(pts, b.hull);
    pushArrow(prims, pts);
  }

  // Captions under each block
  for (const p of placed.values()) {
    const lines = captionLines(p.block, opts);
    if (!lines.length) continue;
    const anchor = proj.project([(p.x0 + p.x1) / 2, p.y, 0]);
    const lowest = Math.max(...p.hull.map((h) => h[1]));
    let y = lowest + 14;
    for (const line of lines) {
      prims.push({ kind: "text", x: anchor[0], y, text: line.text, size: line.size, bold: line.bold, color: line.color, anchor: "middle" });
      y += line.size + 3.5;
    }
  }

  // Bounds, then the legend below everything
  let [minX, minY, maxX, maxY] = bounds(prims);
  if (opts.showLegend) {
    const cats: NodeCategory[] = [];
    for (const p of placed.values()) for (const part of p.block.op ? [] : p.block.parts) if (!cats.includes(part.category)) cats.push(part.category);
    const ly = maxY + 26;
    let lx = minX;
    for (const c of cats) {
      const base = palette[c];
      prims.push({ kind: "poly", pts: [[lx, ly - 9], [lx + 11, ly - 9], [lx + 11, ly + 2], [lx, ly + 2]], fill: base, stroke: stroke(base), width: 0.8 });
      const label = CATEGORY_LABELS[c] ?? c;
      prims.push({ kind: "text", x: lx + 16, y: ly, text: label, size: 10, bold: false, color: INK, anchor: "start" });
      lx += 16 + textWidth(label, 10) + 18;
    }
    [minX, minY, maxX, maxY] = bounds(prims);
  }

  // Move everything so the figure starts at (pad, pad)
  const pad = 18;
  const ox = pad - minX, oy = pad - minY;
  for (const p of prims) {
    if (p.kind === "text") { p.x += ox; p.y += oy; }
    else p.pts = p.pts.map(([x, y]) => [x + ox, y + oy] as Pt);
  }
  return {
    width: Math.ceil(maxX - minX + pad * 2),
    height: Math.ceil(maxY - minY + pad * 2),
    background,
    prims,
    blockCount: blocks.size,
  };
}

function pushArrow(prims: ScenePrim[], pts: Pt[]) {
  if (pts.length < 2) return;
  const tip = pts[pts.length - 1];
  const prev = pts[pts.length - 2];
  const len = Math.hypot(tip[0] - prev[0], tip[1] - prev[1]);
  if (len < 1e-6) return;
  const ux = (tip[0] - prev[0]) / len, uy = (tip[1] - prev[1]) / len;
  const head = Math.min(7, len * 0.8), half = 3;
  const base: Pt = [tip[0] - ux * head, tip[1] - uy * head];
  prims.push({ kind: "line", pts: [...pts.slice(0, -1), base], stroke: EDGE_INK, width: 1.1 });
  prims.push({
    kind: "poly",
    pts: [tip, [base[0] - uy * half, base[1] + ux * half], [base[0] + uy * half, base[1] - ux * half]],
    fill: EDGE_INK,
    stroke: null,
    width: 0,
  });
}

function bounds(prims: ScenePrim[]): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const add = (x: number, y: number) => { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); };
  for (const p of prims) {
    if (p.kind === "text") {
      const w = textWidth(p.text, p.size, p.bold);
      const x0 = p.anchor === "middle" ? p.x - w / 2 : p.x;
      add(x0, p.y - p.size);
      add(x0 + w, p.y + p.size * 0.25);
    } else for (const [x, y] of p.pts) add(x, y);
  }
  return [minX, minY, maxX, maxY];
}

// ─── Writers ────────────────────────────────────────────────────────────────

const f2 = (n: number) => (Math.round(n * 100) / 100).toString();

function xmlEscape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export function sceneToSvg(scene: DiagramScene): string {
  const out: string[] = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${scene.width}" height="${scene.height}" viewBox="0 0 ${scene.width} ${scene.height}">`);
  if (scene.background) out.push(`<rect width="100%" height="100%" fill="${scene.background}"/>`);
  out.push(`<g stroke-linejoin="round" stroke-linecap="round" font-family="Helvetica, Arial, sans-serif">`);
  for (const p of scene.prims) {
    if (p.kind === "poly") {
      out.push(`<polygon points="${p.pts.map(([x, y]) => `${f2(x)},${f2(y)}`).join(" ")}" fill="${p.fill ?? "none"}"${p.stroke ? ` stroke="${p.stroke}" stroke-width="${p.width}"` : ""}/>`);
    } else if (p.kind === "line") {
      out.push(`<polyline points="${p.pts.map(([x, y]) => `${f2(x)},${f2(y)}`).join(" ")}" fill="none" stroke="${p.stroke}" stroke-width="${p.width}"${p.dash ? ` stroke-dasharray="${p.dash.join(" ")}"` : ""}/>`);
    } else {
      out.push(`<text x="${f2(p.x)}" y="${f2(p.y)}" font-size="${p.size}"${p.bold ? ` font-weight="bold"` : ""} fill="${p.color}" text-anchor="${p.anchor}">${xmlEscape(p.text)}</text>`);
    }
  }
  out.push(`</g></svg>`);
  return out.join("\n");
}

/** A string literal for a PDF content stream, in WinAnsi, with anything else replaced by "?" */
function pdfString(s: string): string {
  let out = "(";
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (ch === "(" || ch === ")" || ch === "\\") out += "\\" + ch;
    else if (c >= 32 && c <= 126) out += ch;
    else if (ch === "×") out += "\\327";
    else if (ch === "·") out += "\\267";
    else out += "?";
  }
  return out + ")";
}

function pdfColor(hex: string, op: "rg" | "RG"): string {
  const [r, g, b] = hexToRgb(hex);
  return `${f2(r / 255)} ${f2(g / 255)} ${f2(b / 255)} ${op}`;
}

/** A one-page vector PDF of the scene, 1 px = 1 pt, using the built-in Helvetica fonts */
export function sceneToPdf(scene: DiagramScene): string {
  const H = scene.height;
  const ops: string[] = ["1 J 1 j"];
  if (scene.background) ops.push(`${pdfColor(scene.background, "rg")} 0 0 ${scene.width} ${H} re f`);
  const path = (pts: Pt[]) => pts.map(([x, y], i) => `${f2(x)} ${f2(H - y)} ${i ? "l" : "m"}`).join(" ");
  for (const p of scene.prims) {
    if (p.kind === "poly") {
      const parts: string[] = [];
      if (p.fill) parts.push(pdfColor(p.fill, "rg"));
      if (p.stroke) parts.push(pdfColor(p.stroke, "RG"), `${p.width} w`);
      parts.push(path(p.pts), "h", p.fill && p.stroke ? "B" : p.fill ? "f" : "S");
      ops.push(parts.join(" "));
    } else if (p.kind === "line") {
      ops.push(`${pdfColor(p.stroke, "RG")} ${p.width} w ${p.dash ? `[${p.dash.join(" ")}] 0 d ` : ""}${path(p.pts)} S${p.dash ? " [] 0 d" : ""}`);
    } else {
      const w = textWidth(p.text, p.size, p.bold);
      const x = p.anchor === "middle" ? p.x - w / 2 : p.x;
      ops.push(`BT ${pdfColor(p.color, "rg")} /${p.bold ? "F2" : "F1"} ${p.size} Tf ${f2(x)} ${f2(H - p.y)} Td ${pdfString(p.text)} Tj ET`);
    }
  }
  const content = ops.join("\n");
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${scene.width} ${H}] /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>`,
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
  ];
  let pdf = "%PDF-1.4\n";
  const offsets: number[] = [];
  objects.forEach((o, i) => {
    offsets.push(pdf.length);
    pdf += `${i + 1} 0 obj\n${o}\nendobj\n`;
  });
  const xref = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const off of offsets) pdf += `${String(off).padStart(10, "0")} 00000 n \n`;
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return pdf;
}
