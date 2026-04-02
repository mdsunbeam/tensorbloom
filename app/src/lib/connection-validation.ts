/**
 * Connection validation — prevents incompatible connections.
 */

import type { Connection, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import { MODULE_REGISTRY } from "./registry";
import { resolveHandlesWithRegistry } from "./types";
import type { Node } from "@xyflow/react";

export function isValidConnection(
  connection: Connection,
  nodes: Node<NodeData>[],
  edges: Edge[],
): boolean {
  const { source, target, sourceHandle, targetHandle } = connection;

  // No self-loops
  if (source === target) return false;

  // No null references
  if (!source || !target) return false;

  // No duplicate edges
  const duplicate = edges.some(
    (e) =>
      e.source === source &&
      e.target === target &&
      e.sourceHandle === sourceHandle &&
      e.targetHandle === targetHandle,
  );
  if (duplicate) return false;

  // Check that target handle isn't already occupied (single-input handles)
  const targetNode = nodes.find((n) => n.id === target);
  if (targetNode) {
    const { inputHandles } = resolveHandlesWithRegistry(targetNode.data, MODULE_REGISTRY);
    const handleDef = inputHandles.find((h) => h.id === (targetHandle ?? "in"));
    if (handleDef) {
      // Check if this specific handle already has a connection
      const existingToHandle = edges.some(
        (e) => e.target === target && e.targetHandle === (targetHandle ?? "in"),
      );
      // Multi-input nodes (Add, Concat, MatMul, loss functions) have multiple input handles
      // so only block if the *same* handle is already occupied
      if (existingToHandle) return false;
    }
  }

  // DataType compatibility
  const sourceNode = nodes.find((n) => n.id === source);
  if (sourceNode && targetNode) {
    const srcResolved = resolveHandlesWithRegistry(sourceNode.data, MODULE_REGISTRY);
    const tgtResolved = resolveHandlesWithRegistry(targetNode.data, MODULE_REGISTRY);

    const srcHandle = srcResolved.outputHandles.find((h) => h.id === (sourceHandle ?? "out"));
    const tgtHandle = tgtResolved.inputHandles.find((h) => h.id === (targetHandle ?? "in"));

    if (srcHandle && tgtHandle) {
      // "any" is compatible with everything
      if (srcHandle.dataType !== "any" && tgtHandle.dataType !== "any") {
        if (srcHandle.dataType !== tgtHandle.dataType) return false;
      }
    }
  }

  return true;
}
