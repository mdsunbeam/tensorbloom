/**
 * Undo/redo via snapshot history.
 * Subscribes to graph-store and captures snapshots on meaningful changes.
 */

import { create } from "zustand";
import { useGraphStore, type NodeData } from "./graph-store";
import type { Node, Edge } from "@xyflow/react";

interface Snapshot {
  nodes: Node<NodeData>[];
  edges: Edge[];
}

interface HistoryState {
  past: Snapshot[];
  future: Snapshot[];
  canUndo: boolean;
  canRedo: boolean;
  pushSnapshot: () => void;
  undo: () => void;
  redo: () => void;
  clear: () => void;
}

const MAX_HISTORY = 50;

export const useHistoryStore = create<HistoryState>((set, get) => ({
  past: [],
  future: [],
  canUndo: false,
  canRedo: false,

  pushSnapshot: () => {
    const { nodes, edges } = useGraphStore.getState();
    const snapshot: Snapshot = {
      nodes: structuredClone(nodes),
      edges: structuredClone(edges),
    };
    const past = [...get().past, snapshot].slice(-MAX_HISTORY);
    set({ past, future: [], canUndo: past.length > 0, canRedo: false });
  },

  undo: () => {
    const { past, future } = get();
    if (past.length === 0) return;

    // Save current as future
    const { nodes, edges } = useGraphStore.getState();
    const currentSnapshot: Snapshot = {
      nodes: structuredClone(nodes),
      edges: structuredClone(edges),
    };

    const prev = past[past.length - 1];
    const newPast = past.slice(0, -1);

    useGraphStore.getState().loadGraph(prev.nodes, prev.edges);
    set({
      past: newPast,
      future: [currentSnapshot, ...future],
      canUndo: newPast.length > 0,
      canRedo: true,
    });
  },

  redo: () => {
    const { past, future } = get();
    if (future.length === 0) return;

    const { nodes, edges } = useGraphStore.getState();
    const currentSnapshot: Snapshot = {
      nodes: structuredClone(nodes),
      edges: structuredClone(edges),
    };

    const next = future[0];
    const newFuture = future.slice(1);

    useGraphStore.getState().loadGraph(next.nodes, next.edges);
    set({
      past: [...past, currentSnapshot],
      future: newFuture,
      canUndo: true,
      canRedo: newFuture.length > 0,
    });
  },

  clear: () => set({ past: [], future: [], canUndo: false, canRedo: false }),
}));
