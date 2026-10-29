/**
 * Undo/redo via snapshot history.
 *
 * The graph's last settled state is kept as a baseline. When the graph changes and then stays still
 * for COMMIT_DELAY_MS, the old baseline is pushed onto the undo stack and the new state becomes the
 * baseline. So each user action (add, delete, connect, a drag, a burst of typing in one field) is
 * one undo step, and selection or measurement changes are never steps at all.
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
  /** Records any pending edit as its own step now (call before an action so it isn't merged with earlier typing) */
  pushSnapshot: () => void;
  undo: () => void;
  redo: () => void;
  clear: () => void;
}

const MAX_HISTORY = 50;
export const COMMIT_DELAY_MS = 400;

let baseline: Snapshot | null = null;
let baselineKey = "";
let pendingTimer: ReturnType<typeof setTimeout> | null = null;
// True while undo/redo is loading a snapshot, so that load isn't recorded as a new edit
let applyingHistory = false;

/** The parts of the graph that are the user's work: no selection, drag or measurement state */
function snapshotOf(nodes: Node<NodeData>[], edges: Edge[]): Snapshot {
  return {
    nodes: nodes.map((n) => {
      const { selected, dragging, measured, resizing, ...rest } = n;
      return structuredClone(rest);
    }),
    edges: edges.map((e) => {
      const { selected, animated, ...rest } = e;
      return structuredClone(rest);
    }),
  };
}

const keyOf = (s: Snapshot) => JSON.stringify(s);

function currentSnapshot(): Snapshot {
  const { nodes, edges } = useGraphStore.getState();
  return snapshotOf(nodes, edges);
}

function apply(s: Snapshot) {
  applyingHistory = true;
  useGraphStore.getState().loadGraph(structuredClone(s.nodes), structuredClone(s.edges));
  applyingHistory = false;
  baseline = s;
  baselineKey = keyOf(s);
}

export const useHistoryStore = create<HistoryState>((set, get) => {
  const commit = () => {
    if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
    const snap = currentSnapshot();
    const key = keyOf(snap);
    if (baseline === null) { baseline = snap; baselineKey = key; return; }
    if (key === baselineKey) return;
    const past = [...get().past, baseline].slice(-MAX_HISTORY);
    baseline = snap;
    baselineKey = key;
    set({ past, future: [], canUndo: true, canRedo: false });
  };

  return {
    past: [],
    future: [],
    canUndo: false,
    canRedo: false,

    pushSnapshot: commit,

    undo: () => {
      commit();
      const { past, future } = get();
      if (past.length === 0 || !baseline) return;
      const prev = past[past.length - 1];
      const newFuture = [baseline, ...future];
      apply(prev);
      set({ past: past.slice(0, -1), future: newFuture, canUndo: past.length > 1, canRedo: true });
    },

    redo: () => {
      commit();
      const { past, future } = get();
      if (future.length === 0 || !baseline) return;
      const next = future[0];
      const newPast = [...past, baseline].slice(-MAX_HISTORY);
      apply(next);
      set({ past: newPast, future: future.slice(1), canUndo: true, canRedo: future.length > 1 });
    },

    clear: () => {
      if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
      baseline = currentSnapshot();
      baselineKey = keyOf(baseline);
      set({ past: [], future: [], canUndo: false, canRedo: false });
    },
  };
});

/** Starts recording graph edits as undo steps. Returns a function that stops it. */
export function startHistoryTracking(): () => void {
  useHistoryStore.getState().clear();
  const unsub = useGraphStore.subscribe((state, prev) => {
    if (applyingHistory) return;
    if (state.nodes === prev.nodes && state.edges === prev.edges) return;
    // Wait for a drag to finish so the whole move is one step
    if (state.nodes.some((n) => n.dragging)) {
      if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
      return;
    }
    if (pendingTimer) clearTimeout(pendingTimer);
    pendingTimer = setTimeout(() => useHistoryStore.getState().pushSnapshot(), COMMIT_DELAY_MS);
  });
  return () => {
    unsub();
    if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
  };
}

/** Serialized form of the user's work in the graph, for telling whether it changed */
export function graphKey(): string {
  return keyOf(currentSnapshot());
}
