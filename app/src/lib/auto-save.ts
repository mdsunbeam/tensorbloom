/**
 * Auto-save to localStorage every 30 seconds.
 * Restores on app load if no explicit project is opened.
 */

import { useGraphStore, type NodeData } from "../store/graph-store";
import type { Node, Edge } from "@xyflow/react";

const STORAGE_KEY = "tensorbloom-autosave";
const INTERVAL_MS = 30_000;

interface AutoSaveData {
  timestamp: number;
  projectName: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
}

export function startAutoSave(): () => void {
  const timer = setInterval(() => {
    const { nodes, edges, projectName } = useGraphStore.getState();
    if (nodes.length === 0) return;

    const data: AutoSaveData = {
      timestamp: Date.now(),
      projectName,
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: n.data,
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
        type: e.type,
      })),
    };

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
      // Storage full or unavailable
    }
  }, INTERVAL_MS);

  return () => clearInterval(timer);
}

export function restoreAutoSave(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;

    const data: AutoSaveData = JSON.parse(raw);

    // Only restore if less than 24 hours old
    if (Date.now() - data.timestamp > 24 * 60 * 60 * 1000) {
      localStorage.removeItem(STORAGE_KEY);
      return false;
    }

    if (data.nodes.length === 0) return false;

    useGraphStore.getState().loadGraph(
      data.nodes as Node<NodeData>[],
      data.edges as Edge[],
    );
    useGraphStore.getState().setProjectName(data.projectName);
    return true;
  } catch {
    return false;
  }
}

export function clearAutoSave() {
  localStorage.removeItem(STORAGE_KEY);
}
