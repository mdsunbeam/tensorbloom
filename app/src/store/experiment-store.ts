import { create } from "zustand";
import { useMemo } from "react";
import { useRunStore, type RunRecord } from "./run-store";
import { useGraphStore } from "./graph-store";
import { useTrainingStore } from "./training-store";
import { replaceGraph } from "../lib/document-actions";
import { setupDiff, type SetupDiff } from "../lib/experiment-diff";
import { useLayerStatsStore } from "./layer-stats-store";

/**
 * A "change one thing" experiment: start from a run's model and settings, change one thing,
 * train, and see what that change did next to the run it started from.
 */
interface ExperimentState {
  /** The run the experiment starts from, while one is open */
  baseRunId: string | null;
  /** Layers on the canvas that differ from the starting run, so they can be marked */
  marks: Map<string, "added" | "changed">;
  setMarks: (marks: Map<string, "added" | "changed">) => void;
  start: (run: RunRecord) => void;
  end: () => void;
}

const NO_MARKS = new Map<string, "added" | "changed">();

export const useExperimentStore = create<ExperimentState>((set) => ({
  baseRunId: null,
  marks: NO_MARKS,
  setMarks: (marks) => set({ marks: marks.size ? marks : NO_MARKS }),
  start: (run) => {
    const graph = useGraphStore.getState();
    const training = useTrainingStore.getState();
    // Begin from exactly what the run trained, unless that's already on the canvas
    const current = setupDiff({ ...run.graph, config: run.config }, { nodes: graph.nodes, edges: graph.edges, config: training.config });
    if (current.layers.length || current.connections.length) {
      replaceGraph(run.graph.nodes, run.graph.edges, graph.projectName, run.graph.groups);
    }
    if (current.settings.length) {
      // The device is this computer's, not the run's
      training.updateConfig({ ...run.config, device: training.config.device });
    }
    // The graph shows what training did to the starting run's layers
    useLayerStatsStore.getState().showRun(run);
    set({ baseRunId: run.id });
  },
  end: () => set({ baseRunId: null, marks: NO_MARKS }),
}));

/** The run the open experiment starts from, if it's still kept */
export function useBaseRun(): RunRecord | null {
  const baseRunId = useExperimentStore((s) => s.baseRunId);
  const runs = useRunStore((s) => s.runs);
  return useMemo(() => runs.find((r) => r.id === baseRunId) ?? null, [runs, baseRunId]);
}

/** What's on the canvas and in the Training panel now, against the experiment's starting run */
export function useExperimentDiff(): { base: RunRecord; diff: SetupDiff } | null {
  const base = useBaseRun();
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const config = useTrainingStore((s) => s.config);
  return useMemo(
    () => (base ? { base, diff: setupDiff({ ...base.graph, config: base.config }, { nodes, edges, config }) } : null),
    [base, nodes, edges, config],
  );
}

/** The diff between a finished run and the run it was an experiment on */
export function runDiff(base: RunRecord, run: RunRecord): SetupDiff {
  return setupDiff({ ...base.graph, config: base.config }, { ...run.graph, config: run.config });
}
