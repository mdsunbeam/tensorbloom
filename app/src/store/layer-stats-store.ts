import { create } from "zustand";
import {
  appendHistory,
  type FeatureMaps,
  type LayerOverlay,
  type LayerPoint,
  type LayerSnapshot,
  type RunLayerStats,
} from "../lib/layer-health";
import type { RunRecord } from "./run-store";

const OVERLAY_KEY = "tensorbloom.layerOverlay";

function savedOverlay(): LayerOverlay {
  try {
    const v = localStorage.getItem(OVERLAY_KEY);
    if (v === "off" || v === "gradients" || v === "activations") return v;
  } catch { /* storage unavailable */ }
  return "gradients";
}

/** Where the stats on the graph come from: the run training now, or a past run */
export type LayerStatsSource = { kind: "live" } | { kind: "run"; runId: string; runNumber: number; modelKey: string };

interface LayerStatsState {
  /** What the graph is colored by. Kept between sessions. */
  overlay: LayerOverlay;
  source: LayerStatsSource | null;
  /** The latest captured batch */
  snapshot: LayerSnapshot | null;
  history: Record<string, LayerPoint[]>;
  /** The latest epoch's feature maps. Kept for this session only, by run. */
  maps: FeatureMaps | null;
  mapsByRun: Record<string, FeatureMaps>;
  setOverlay: (overlay: LayerOverlay) => void;
  /** A run is starting: clear the graph for its stats */
  startLive: () => void;
  addSnapshot: (snapshot: LayerSnapshot) => void;
  setMaps: (maps: FeatureMaps) => void;
  /** The live run was saved as this run, so its stats now belong to it */
  liveBecameRun: (run: RunRecord) => void;
  /** Show a past run's stats, or clear the graph when it kept none */
  showRun: (run: RunRecord | undefined) => void;
  clear: () => void;
}

export const useLayerStatsStore = create<LayerStatsState>((set, get) => ({
  overlay: savedOverlay(),
  source: null,
  snapshot: null,
  history: {},
  maps: null,
  mapsByRun: {},

  setOverlay: (overlay) => {
    try { localStorage.setItem(OVERLAY_KEY, overlay); } catch { /* storage unavailable */ }
    set({ overlay });
  },
  startLive: () => set({ source: { kind: "live" }, snapshot: null, history: {}, maps: null }),
  addSnapshot: (snapshot) => {
    if (get().source?.kind !== "live") return;
    set({ snapshot, history: appendHistory(get().history, snapshot) });
  },
  setMaps: (maps) => {
    if (get().source?.kind !== "live") return;
    set({ maps });
  },
  liveBecameRun: (run) => {
    const { source, maps, mapsByRun } = get();
    if (source?.kind !== "live") return;
    set({
      source: { kind: "run", runId: run.id, runNumber: run.number, modelKey: run.model.key },
      mapsByRun: maps ? { ...mapsByRun, [run.id]: maps } : mapsByRun,
    });
  },
  showRun: (run) => {
    if (!run?.layers) {
      set({ source: null, snapshot: null, history: {}, maps: null });
      return;
    }
    set({
      source: { kind: "run", runId: run.id, runNumber: run.number, modelKey: run.model.key },
      snapshot: run.layers.snapshot,
      history: run.layers.history,
      maps: get().mapsByRun[run.id] ?? null,
    });
  },
  clear: () => set({ source: null, snapshot: null, history: {}, maps: null, mapsByRun: {} }),
}));

/** The stats a run keeps when it's saved, or nothing when none were captured */
export function liveRunLayerStats(): RunLayerStats | undefined {
  const { source, snapshot, history } = useLayerStatsStore.getState();
  if (source?.kind !== "live" || !snapshot) return undefined;
  return { snapshot, history };
}

/** A project's runs were loaded: show the latest run's stats */
export function showLatestRunLayers(runs: RunRecord[]) {
  const store = useLayerStatsStore.getState();
  store.clear();
  store.showRun([...runs].reverse().find((r) => r.layers));
}
