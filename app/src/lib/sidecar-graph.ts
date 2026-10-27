/**
 * Serializes the editor graph into the JSON the Python sidecar builds models from.
 * Shared by training and export so both build the exact same nn.Module.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { isDatasetNode, resolveDataNodeShape } from "./dataset-metadata";
import { getCustomBlock } from "./custom-blocks";

export interface SidecarGraphNode {
  id: string;
  type: string;
  moduleType: string;
  label: string;
  params: Record<string, unknown>;
  codeBlock?: unknown;
  dataOutputs?: unknown;
  customCode?: string;
}

export interface SidecarGraphEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
}

export interface SidecarGraph {
  nodes: SidecarGraphNode[];
  edges: SidecarGraphEdge[];
}

const LOSS_TYPES = new Set([
  "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "BCELoss",
  "L1Loss", "HuberLoss", "NLLLoss", "KLDivLoss",
  "TripletMarginLoss", "CosineEmbeddingLoss", "CTCLoss", "SmoothL1Loss",
]);

export function buildSidecarGraph(nodes: Node<NodeData>[], edges: Edge[]): SidecarGraph {
  const datasetNode = nodes.find((n) => isDatasetNode(n.data.moduleType));

  // Filter Dataset nodes from the sidecar graph — they're frontend-only
  const datasetNodeIds = new Set(
    nodes.filter((n) => isDatasetNode(n.data.moduleType)).map((n) => n.id)
  );

  // Identify loss nodes — edges from Data→Loss must be preserved for target routing
  const lossNodeIds = new Set(nodes.filter((n) => LOSS_TYPES.has(n.data.moduleType)).map((n) => n.id));

  // Auto-inject a virtual Input node when Data connects directly to model layers
  // (so users don't need to manually add an Input node when using the Data block)
  const hasExplicitInput = nodes.some((n) => n.data.moduleType === "Input");
  const virtualNodes: SidecarGraphNode[] = [];
  const virtualEdges: SidecarGraphEdge[] = [];
  const rerouted = new Set<string>(); // edge IDs rerouted through virtual Input

  if (!hasExplicitInput && datasetNode) {
    const dataEdgesToLayers = edges.filter((e) =>
      datasetNodeIds.has(e.source) && !lossNodeIds.has(e.target) && !datasetNodeIds.has(e.target)
    );
    if (dataEdgesToLayers.length > 0) {
      const resolved = resolveDataNodeShape(datasetNode.data.params);
      const shape = resolved?.inputShape ?? [1, 10];
      const virtualId = "_auto_input";
      virtualNodes.push({
        id: virtualId,
        type: "placeholder",
        moduleType: "Input",
        label: "Input",
        params: { shape: JSON.stringify(shape), dtype: "float32" },
      });
      for (const e of dataEdgesToLayers) {
        rerouted.add(e.id);
        virtualEdges.push({
          id: `e_${virtualId}_${e.target}`,
          source: virtualId,
          target: e.target,
          sourceHandle: "out",
          targetHandle: e.targetHandle ?? "in",
        });
      }
    }
  }

  return {
    nodes: [
      ...virtualNodes,
      ...nodes
        .filter((n) => !datasetNodeIds.has(n.id))
        .map((n) => {
          const customCode = n.data.codeBlock ? undefined : getCustomBlock(n.data.moduleType)?.code;
          return {
            id: n.id,
            type: n.data.moduleType === "Input" ? "placeholder"
              : n.data.moduleType === "Output" ? "output"
              : "call_module",
            moduleType: n.data.moduleType,
            label: n.data.label,
            params: n.data.params,
            ...(n.data.codeBlock ? { codeBlock: n.data.codeBlock } : {}),
            ...(n.data.dataOutputs ? { dataOutputs: n.data.dataOutputs } : {}),
            ...(customCode ? { customCode } : {}),
          };
        }),
    ],
    edges: [
      ...virtualEdges,
      ...edges
        .filter((e) => {
          if (rerouted.has(e.id)) return false;
          if (datasetNodeIds.has(e.source) && lossNodeIds.has(e.target)) return true;
          return !datasetNodeIds.has(e.source) && !datasetNodeIds.has(e.target);
        })
        .map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle ?? "out",
          targetHandle: e.targetHandle ?? "in",
        })),
    ],
  };
}

/** Parse a shape param like "[1, 3, 224, 224]" or "(1, 3, 224, 224)". */
function parseShape(raw: unknown): number[] | null {
  try {
    const parsed = JSON.parse(String(raw).replace(/\(/g, "[").replace(/\)/g, "]"));
    return Array.isArray(parsed) && parsed.every((d) => typeof d === "number") ? parsed : null;
  } catch {
    return null;
  }
}

/** Input shapes for every model input, in graph order (matches the forward() argument order). */
export function graphInputShapes(graph: SidecarGraph): Record<string, number[]> {
  const inputs = graph.nodes
    .filter((n) => n.type === "placeholder")
    .map((n) => parseShape(n.params.shape) ?? [1, 3, 224, 224]);
  if (inputs.length <= 1) return { x: inputs[0] ?? [1, 3, 224, 224] };
  return Object.fromEntries(inputs.map((shape, i) => [`input_${i}`, shape]));
}
