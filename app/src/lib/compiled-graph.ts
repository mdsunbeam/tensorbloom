import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { NodeCategory } from "./types";
import { MODULE_REGISTRY } from "./registry";
import { autoLayout } from "./auto-layout";

/** What the sidecar's "compile" method returns for a piece of PyTorch code */
export interface CompiledGraph {
  nodes: Array<{ id: string; type: string; moduleType?: string; label: string; position: { x: number; y: number }; params: Record<string, unknown> }>;
  edges: Array<{ id: string; source: string; target: string; sourceHandle: string; targetHandle: string }>;
}

/** Turns a compiled model into canvas layers, laid out top to bottom */
export function compiledToGraph(result: CompiledGraph): { nodes: Node<NodeData>[]; edges: Edge[] } {
  const nodes: Node<NodeData>[] = result.nodes.map((n) => {
    const mt = n.moduleType ?? "Identity";
    const def = MODULE_REGISTRY[mt];
    return {
      id: n.id, type: "tensorLabNode", position: n.position,
      data: { moduleType: mt, label: n.label, category: (def?.category ?? "linear") as NodeCategory, params: n.params },
    };
  });
  const edges: Edge[] = result.edges.map((e) => ({ ...e, type: "smoothstep" }));
  return { nodes: autoLayout(nodes, edges), edges };
}
