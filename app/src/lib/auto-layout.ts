/**
 * Auto-layout algorithm for node graphs.
 * Places nodes in a clean vertical flow, all centered on the same X.
 * Multi-input nodes (like Add/Concat) offset their secondary inputs to the side.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { getParamSummary, quickParamCount } from "./node-summary";

const GRID = 20;
const NODE_WIDTH = 220;
const GAP_Y = 40;

// Loss nodes that have a "target" handle on the right side (wider)
const LOSS_WITH_TARGET = new Set([
  "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "BCELoss",
  "L1Loss", "HuberLoss", "NLLLoss", "KLDivLoss",
  "CosineEmbeddingLoss", "CTCLoss", "SmoothL1Loss",
]);

/** Compute visual width of a node (must match TensorBloomNode.tsx logic) */
function nodeWidth(data: NodeData): number {
  if (data.dataOutputs) return 240;
  if (LOSS_WITH_TARGET.has(data.moduleType)) return 220;
  return 200;
}

/** Rough rendered height of a node (must track TensorBloomNode.tsx), so every layer gets the same gap below it */
function nodeHeight(data: NodeData): number {
  if (data.dataOutputs) return 90;
  let h = 42; // color bar + title row
  if (getParamSummary(data)) h += 18;
  if (quickParamCount(data.moduleType, data.params) > 0) h += 18;
  return h + 28; // output shape line
}

function snap(v: number): number {
  return Math.round(v / GRID) * GRID;
}

export function autoLayout(
  nodes: Node<NodeData>[],
  edges: Edge[],
): Node<NodeData>[] {
  if (nodes.length === 0) return nodes;

  // Build adjacency
  const childrenOf = new Map<string, string[]>();
  const parentsOf = new Map<string, string[]>();
  for (const n of nodes) {
    childrenOf.set(n.id, []);
    parentsOf.set(n.id, []);
  }
  for (const e of edges) {
    childrenOf.get(e.source)?.push(e.target);
    parentsOf.get(e.target)?.push(e.source);
  }

  // Assign layers (Y position) via longest path
  const layerOf = new Map<string, number>();
  const roots = nodes.filter((n) => (parentsOf.get(n.id)?.length ?? 0) === 0);

  function assignLayer(id: string, depth: number) {
    const current = layerOf.get(id) ?? -1;
    if (depth > current) {
      layerOf.set(id, depth);
      for (const child of childrenOf.get(id) ?? []) {
        assignLayer(child, depth + 1);
      }
    }
  }
  for (const r of roots) assignLayer(r.id, 0);
  for (const n of nodes) { if (!layerOf.has(n.id)) layerOf.set(n.id, 0); }

  // Find nodes that are "secondary" inputs (skip connections)
  // A node is secondary if it connects to a multi-input node (Add, Concat, etc.)
  // via a non-primary edge (i.e., it already has another parent in the main chain)
  const multiInputNodes = new Set<string>();
  const secondaryParents = new Set<string>(); // nodes that feed skip connections

  for (const n of nodes) {
    const parents = parentsOf.get(n.id) ?? [];
    if (parents.length > 1) {
      multiInputNodes.add(n.id);
      // The parent with the highest layer diff is the "skip" source
      let maxDiff = 0;
      let skipParent = "";
      for (const p of parents) {
        const diff = (layerOf.get(n.id) ?? 0) - (layerOf.get(p) ?? 0);
        if (diff > maxDiff) {
          maxDiff = diff;
          skipParent = p;
        }
      }
      if (skipParent && maxDiff > 1) {
        secondaryParents.add(skipParent);
      }
    }
  }

  // Position: all bottom/top handles aligned on the same vertical axis.
  // The handle is at 50% of node width, so we fix the handle X and derive
  // the node X from it (no X-snapping, which would misalign different widths).
  const nodeMap = new Map<string, Node<NodeData>>();
  for (const n of nodes) nodeMap.set(n.id, n);

  const handleX = snap(300) + NODE_WIDTH / 2; // fixed handle center line
  const positions = new Map<string, { x: number; y: number }>();

  // Group by layer
  const layers = new Map<number, string[]>();
  for (const [id, l] of layerOf) {
    if (!layers.has(l)) layers.set(l, []);
    layers.get(l)!.push(id);
  }

  const maxLayer = Math.max(...layers.keys(), 0);

  let y = 40;
  for (let l = 0; l <= maxLayer; l++) {
    const ids = layers.get(l) ?? [];
    if (l > 0) {
      const above = layers.get(l - 1) ?? [];
      const tallest = Math.max(0, ...above.map((id) => nodeHeight(nodeMap.get(id)!.data)));
      y += Math.ceil((tallest + GAP_Y) / GRID) * GRID;
    }

    if (ids.length === 1) {
      // Single node — place so its center handle sits exactly on handleX
      const w = nodeWidth(nodeMap.get(ids[0])!.data);
      positions.set(ids[0], { x: handleX - w / 2, y });
    } else {
      // Multiple nodes in same layer — spread horizontally, centered around handleX
      const totalWidth = ids.length * NODE_WIDTH + (ids.length - 1) * 40;
      const startCenter = handleX - totalWidth / 2 + NODE_WIDTH / 2;
      ids.forEach((id, i) => {
        const w = nodeWidth(nodeMap.get(id)!.data);
        const slotCenter = startCenter + i * (NODE_WIDTH + 40);
        positions.set(id, { x: slotCenter - w / 2, y });
      });
    }
  }

  // Build result
  const result: Node<NodeData>[] = [];

  for (const n of nodes) {
    const w = nodeWidth(n.data);
    const pos = positions.get(n.id) ?? { x: handleX - w / 2, y: 0 };
    result.push({
      ...n,
      position: pos,
    });
  }

  return result;
}
