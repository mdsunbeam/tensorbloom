/**
 * Marks the assistant's pending changes on the canvas: added layers and connections in green, changed
 * layers in amber, and removed ones shown as faded red ghosts where they were.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../../store/graph-store";
import type { GraphDoc } from "../graph-ops";
import { visibleParams } from "./describe";

const edgeKey = (e: Edge) => `${e.source}:${e.sourceHandle ?? "out"}>${e.target}:${e.targetHandle ?? "in"}`;

function layerChanged(a: Node<NodeData>, b: Node<NodeData>): boolean {
  return a.data.moduleType !== b.data.moduleType
    || a.data.label !== b.data.label
    || JSON.stringify(visibleParams(a.data.params)) !== JSON.stringify(visibleParams(b.data.params))
    || JSON.stringify(a.data.codeBlock ?? null) !== JSON.stringify(b.data.codeBlock ?? null);
}

export function previewMarks(
  view: { nodes: Node[]; edges: Edge[] },
  base: GraphDoc,
  nodes: Node<NodeData>[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const before = new Map(base.nodes.map((n) => [n.id, n]));
  const now = new Set(nodes.map((n) => n.id));
  const marked = view.nodes.map((n) => {
    if (n.type !== "tensorLabNode") return n;
    const old = before.get(n.id);
    const mark = !old ? "tb-preview-added" : layerChanged(old, n as Node<NodeData>) ? "tb-preview-changed" : null;
    return mark ? { ...n, className: [n.className, mark].filter(Boolean).join(" ") } : n;
  });
  const ghosts: Node[] = base.nodes
    .filter((n) => !now.has(n.id))
    .map((n) => ({ ...n, id: `__removed_${n.id}`, className: "tb-preview-removed", draggable: false, selectable: false, connectable: false, deletable: false, zIndex: -1 }));

  const oldEdges = new Set(base.edges.map(edgeKey));
  const newEdges = new Set(edges.map(edgeKey));
  const markedEdges = view.edges.map((e) => (oldEdges.has(edgeKey(e)) ? e : { ...e, className: [e.className, "tb-preview-added-edge"].filter(Boolean).join(" ") }));
  const ghostOf = (id: string) => (now.has(id) ? id : `__removed_${id}`);
  const goneEdges: Edge[] = base.edges
    .filter((e) => !newEdges.has(edgeKey(e)))
    .map((e) => ({ ...e, id: `__removed_${e.id}`, source: ghostOf(e.source), target: ghostOf(e.target), className: "tb-preview-removed-edge", selectable: false, deletable: false, focusable: false }));
  return { nodes: [...ghosts, ...marked], edges: [...markedEdges, ...goneEdges] };
}
