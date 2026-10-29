/**
 * Layer groups: named frames around a set of layers, for organizing a big model.
 *
 * A group only changes how the canvas looks. Its layers stay ordinary nodes in the graph, so code
 * generation, shape inference, training and export never see groups. A collapsed group hides its
 * layers behind one card, and the edges that cross its border are drawn to that card instead.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";

export interface LayerGroup {
  id: string;
  name: string;
  /** The layers in the group. A layer belongs to at most one group. */
  nodeIds: string[];
  collapsed: boolean;
}

/** Room around the layers inside an open group, and for its title bar */
export const FRAME_PAD = 16;
export const FRAME_HEADER = 30;
/** Size of a collapsed group's card */
export const CARD_WIDTH = 220;
export const CARD_HEIGHT = 74;

/** The node type the canvas draws groups with */
export const GROUP_NODE_TYPE = "layerGroup";

const DEFAULT_SIZE = { w: 200, h: 80 };

function sizeOf(n: Node<NodeData>) {
  return {
    w: n.measured?.width ?? (n.width as number | undefined) ?? DEFAULT_SIZE.w,
    h: n.measured?.height ?? (n.height as number | undefined) ?? DEFAULT_SIZE.h,
  };
}

/** The box around a group's layers, in canvas coordinates, or null when it has none */
export function layersBounds(group: LayerGroup, nodes: Node<NodeData>[]): { x: number; y: number; w: number; h: number } | null {
  const ids = new Set(group.nodeIds);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of nodes) {
    if (!ids.has(n.id)) continue;
    const s = sizeOf(n);
    x0 = Math.min(x0, n.position.x);
    y0 = Math.min(y0, n.position.y);
    x1 = Math.max(x1, n.position.x + s.w);
    y1 = Math.max(y1, n.position.y + s.h);
  }
  return x0 === Infinity ? null : { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Where a group is drawn: a frame around its layers when open, a card at their top left when collapsed */
export function groupRect(group: LayerGroup, nodes: Node<NodeData>[]): { x: number; y: number; w: number; h: number } | null {
  const b = layersBounds(group, nodes);
  if (!b) return null;
  if (group.collapsed) return { x: b.x + (b.w - CARD_WIDTH) / 2, y: b.y, w: CARD_WIDTH, h: CARD_HEIGHT };
  return { x: b.x - FRAME_PAD, y: b.y - FRAME_PAD - FRAME_HEADER, w: b.w + FRAME_PAD * 2, h: b.h + FRAME_PAD * 2 + FRAME_HEADER };
}

/** Maps each grouped layer to its group's id */
export function groupIndex(groups: LayerGroup[]): Map<string, string> {
  const m = new Map<string, string>();
  for (const g of groups) for (const id of g.nodeIds) m.set(id, g.id);
  return m;
}

/** Drops layers that no longer exist, and groups left with none */
export function pruneGroups(groups: LayerGroup[], nodes: Node<NodeData>[]): LayerGroup[] {
  const ids = new Set(nodes.map((n) => n.id));
  let changed = false;
  const out: LayerGroup[] = [];
  for (const g of groups) {
    const kept = g.nodeIds.filter((id) => ids.has(id));
    if (kept.length !== g.nodeIds.length) changed = true;
    if (kept.length === 0) continue;
    out.push(kept.length === g.nodeIds.length ? g : { ...g, nodeIds: kept });
  }
  return changed ? out : groups;
}

/** Adds a group of the given layers, taking them out of any group they were in */
export function addGroup(groups: LayerGroup[], nodeIds: string[], name: string, id: string): LayerGroup[] {
  const taken = new Set(nodeIds);
  const rest = groups
    .map((g) => (g.nodeIds.some((n) => taken.has(n)) ? { ...g, nodeIds: g.nodeIds.filter((n) => !taken.has(n)) } : g))
    .filter((g) => g.nodeIds.length > 0);
  return [...rest, { id, name, nodeIds: [...nodeIds], collapsed: false }];
}

/** Groups read from a file, keeping only well-formed ones and giving each layer to one group */
export function normalizeGroups(raw: unknown): LayerGroup[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: LayerGroup[] = [];
  for (const g of raw) {
    if (!g || typeof g !== "object") continue;
    const { id, name, nodeIds, collapsed } = g as Partial<LayerGroup>;
    if (typeof id !== "string" || !Array.isArray(nodeIds) || out.some((o) => o.id === id)) continue;
    const ids = nodeIds.filter((n): n is string => typeof n === "string" && !seen.has(n));
    ids.forEach((n) => seen.add(n));
    if (ids.length === 0) continue;
    out.push({ id, name: typeof name === "string" && name.trim() ? name : "Group", nodeIds: ids, collapsed: collapsed === true });
  }
  return out;
}

/** A name that isn't taken yet: "Block 1", "Block 2", ... */
export function nextGroupName(groups: LayerGroup[]): string {
  const names = new Set(groups.map((g) => g.name));
  for (let i = 1; ; i++) if (!names.has(`Block ${i}`)) return `Block ${i}`;
}

export interface GroupNodeData extends Record<string, unknown> {
  group: LayerGroup;
  layerCount: number;
}

/** An edge as the canvas draws it, remembering the layers it really connects */
export interface ViewEdgeData extends Record<string, unknown> {
  origSource?: string;
  origTarget?: string;
}

/**
 * What the canvas shows: the graph's nodes and edges, plus a node per group. A collapsed group's
 * layers are hidden, edges inside it are hidden, and edges crossing its border attach to its card.
 */
export function canvasView(
  nodes: Node<NodeData>[],
  edges: Edge[],
  groups: LayerGroup[],
): { nodes: Node[]; edges: Edge[] } {
  if (groups.length === 0) return { nodes, edges };
  const index = groupIndex(groups);
  const collapsed = new Set(groups.filter((g) => g.collapsed).map((g) => g.id));

  const groupNodes: Node[] = [];
  for (const g of groups) {
    const r = groupRect(g, nodes);
    if (!r) continue;
    groupNodes.push({
      id: g.id,
      type: GROUP_NODE_TYPE,
      position: { x: r.x, y: r.y },
      width: r.w,
      height: r.h,
      data: { group: g, layerCount: g.nodeIds.length } satisfies GroupNodeData,
      selectable: false,
      connectable: false,
      deletable: false,
      // An open group sits behind its layers and is dragged by its title bar
      zIndex: g.collapsed ? 0 : -1,
      ...(g.collapsed ? {} : { dragHandle: ".group-drag-handle" }),
    });
  }

  const viewNodes: Node[] = nodes.map((n) => {
    const g = index.get(n.id);
    return g && collapsed.has(g) ? { ...n, hidden: true } : n;
  });

  const viewEdges: Edge[] = [];
  for (const e of edges) {
    const gs = index.get(e.source);
    const gt = index.get(e.target);
    const hideSource = gs && collapsed.has(gs);
    const hideTarget = gt && collapsed.has(gt);
    if (!hideSource && !hideTarget) { viewEdges.push(e); continue; }
    if (gs && gs === gt) { viewEdges.push({ ...e, hidden: true }); continue; }
    viewEdges.push({
      ...e,
      source: hideSource ? gs! : e.source,
      sourceHandle: hideSource ? "out" : e.sourceHandle,
      target: hideTarget ? gt! : e.target,
      targetHandle: hideTarget ? "in" : e.targetHandle,
      data: { ...e.data, origSource: e.source, origTarget: e.target } satisfies ViewEdgeData,
    });
  }

  return { nodes: [...groupNodes, ...viewNodes], edges: viewEdges };
}
