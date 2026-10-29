/**
 * Auto-save to localStorage every 30 seconds while the project has unsaved changes.
 * On the next launch the app offers to restore it (see readAutoSave).
 */

import { useGraphStore, type NodeData } from "../store/graph-store";
import { useDocumentStore } from "../store/document-store";
import { useRunStore, type RunRecord } from "../store/run-store";
import type { LayerGroup } from "./groups";
import type { Node, Edge } from "@xyflow/react";

const STORAGE_KEY = "tensorbloom-autosave";
const INTERVAL_MS = 30_000;
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;

export interface AutoSaveData {
  timestamp: number;
  projectName: string;
  filePath?: string | null;
  nodes: Node<NodeData>[];
  edges: Edge[];
  runs?: RunRecord[];
  groups?: LayerGroup[];
}

/** Writes the project now if it has unsaved work, and removes the old copy if it doesn't */
export function writeAutoSave() {
  const { nodes, edges, projectName } = useGraphStore.getState();
  const { dirty, filePath } = useDocumentStore.getState();
  try {
    // Saved or empty work needs no recovery copy, and a stale one would come back on the next launch
    if (!dirty || nodes.length === 0) {
      localStorage.removeItem(STORAGE_KEY);
      return;
    }
    const data: AutoSaveData = {
      timestamp: Date.now(),
      projectName,
      filePath,
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
      runs: useRunStore.getState().runs,
      groups: useGraphStore.getState().groups,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    // Storage full or unavailable
  }
}

export function startAutoSave(): () => void {
  const timer = setInterval(writeAutoSave, INTERVAL_MS);
  return () => clearInterval(timer);
}

/** The unsaved work left by the last session, if there is any worth offering back */
export function readAutoSave(): AutoSaveData | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data: AutoSaveData = JSON.parse(raw);
    if (Date.now() - data.timestamp > MAX_AGE_MS || !data.nodes?.length) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

export function clearAutoSave() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage unavailable
  }
}
