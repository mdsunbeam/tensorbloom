/**
 * Composite nodes — group selected nodes into a reusable sub-graph block.
 * The block appears as a single node that can be expanded/collapsed.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { NodeCategory } from "./types";

export interface CompositeBlock {
  id: string;
  name: string;
  description: string;
  innerNodes: Node<NodeData>[];
  innerEdges: Edge[];
  inputNodeIds: string[]; // nodes that receive external inputs
  outputNodeIds: string[]; // nodes that produce external outputs
}

let compositeCounter = 0;

/**
 * Creates a composite block from selected nodes and their interconnecting edges.
 * Returns the new collapsed node + updated node/edge arrays.
 */
export function createCompositeBlock(
  allNodes: Node<NodeData>[],
  allEdges: Edge[],
  selectedIds: Set<string>,
  name: string,
): {
  nodes: Node<NodeData>[];
  edges: Edge[];
  block: CompositeBlock;
} {
  const selectedNodes = allNodes.filter((n) => selectedIds.has(n.id));
  const unselectedNodes = allNodes.filter((n) => !selectedIds.has(n.id));

  // Edges fully inside the selection
  const innerEdges = allEdges.filter(
    (e) => selectedIds.has(e.source) && selectedIds.has(e.target),
  );

  // Edges crossing the boundary
  const incomingEdges = allEdges.filter(
    (e) => !selectedIds.has(e.source) && selectedIds.has(e.target),
  );
  const outgoingEdges = allEdges.filter(
    (e) => selectedIds.has(e.source) && !selectedIds.has(e.target),
  );
  const externalEdges = allEdges.filter(
    (e) => !selectedIds.has(e.source) && !selectedIds.has(e.target),
  );

  const inputNodeIds = [...new Set(incomingEdges.map((e) => e.target))];
  const outputNodeIds = [...new Set(outgoingEdges.map((e) => e.source))];

  // Compute center position
  const avgX = selectedNodes.reduce((s, n) => s + n.position.x, 0) / selectedNodes.length;
  const avgY = selectedNodes.reduce((s, n) => s + n.position.y, 0) / selectedNodes.length;

  const blockId = `composite_${++compositeCounter}`;

  const block: CompositeBlock = {
    id: blockId,
    name,
    description: `${selectedNodes.length} layers grouped`,
    innerNodes: selectedNodes,
    innerEdges,
    inputNodeIds,
    outputNodeIds,
  };

  // Create the collapsed node
  const compositeNode: Node<NodeData> = {
    id: blockId,
    type: "tensorLabNode",
    position: { x: avgX, y: avgY },
    data: {
      moduleType: "Composite",
      label: name,
      category: "linear" as NodeCategory,
      params: {
        _innerNodeCount: selectedNodes.length,
        _blockId: blockId,
      },
    },
  };

  // Rewire external edges to point to/from the composite node
  const rewiredEdges: Edge[] = [
    ...externalEdges,
    ...incomingEdges.map((e) => ({
      ...e,
      id: `${e.id}_to_${blockId}`,
      target: blockId,
      targetHandle: "in",
    })),
    ...outgoingEdges.map((e) => ({
      ...e,
      id: `${e.id}_from_${blockId}`,
      source: blockId,
      sourceHandle: "out",
    })),
  ];

  return {
    nodes: [...unselectedNodes, compositeNode],
    edges: rewiredEdges,
    block,
  };
}

/**
 * Expand a composite block back into its individual nodes.
 */
export function expandCompositeBlock(
  allNodes: Node<NodeData>[],
  allEdges: Edge[],
  blockId: string,
  block: CompositeBlock,
): {
  nodes: Node<NodeData>[];
  edges: Edge[];
} {
  // Remove the composite node
  const otherNodes = allNodes.filter((n) => n.id !== blockId);

  // Rewire edges that point to/from the composite back to the inner nodes
  const compositePos = allNodes.find((n) => n.id === blockId)?.position ?? { x: 0, y: 0 };
  const offsetX = compositePos.x - (block.innerNodes[0]?.position.x ?? 0);
  const offsetY = compositePos.y - (block.innerNodes[0]?.position.y ?? 0);

  const restoredNodes = block.innerNodes.map((n) => ({
    ...n,
    position: { x: n.position.x + offsetX, y: n.position.y + offsetY },
  }));

  // Separate edges
  const edgesToComposite = allEdges.filter((e) => e.target === blockId);
  const edgesFromComposite = allEdges.filter((e) => e.source === blockId);
  const otherEdges = allEdges.filter((e) => e.source !== blockId && e.target !== blockId);

  // Rewire incoming edges to the first input node
  const rewiredIn = edgesToComposite.map((e) => ({
    ...e,
    target: block.inputNodeIds[0] ?? block.innerNodes[0]?.id ?? e.target,
    targetHandle: "in",
    id: `${e.id}_expanded`,
  }));

  // Rewire outgoing edges from the last output node
  const rewiredOut = edgesFromComposite.map((e) => ({
    ...e,
    source: block.outputNodeIds[0] ?? block.innerNodes[block.innerNodes.length - 1]?.id ?? e.source,
    sourceHandle: "out",
    id: `${e.id}_expanded`,
  }));

  return {
    nodes: [...otherNodes, ...restoredNodes],
    edges: [...otherEdges, ...block.innerEdges, ...rewiredIn, ...rewiredOut],
  };
}
