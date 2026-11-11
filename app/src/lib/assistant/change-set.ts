/**
 * The assistant's pending changes. Edit tools change the real graph (and training settings) right away,
 * so the canvas, shapes and code all show the result, but undo history is paused and the starting point
 * is kept. Apply records everything as one undo step; Discard puts the starting point back.
 */

import { create } from "zustand";
import { useGraphStore } from "../../store/graph-store";
import { useTrainingStore, type TrainingConfig } from "../../store/training-store";
import { pauseHistory, useHistoryStore } from "../../store/history-store";
import { inferShapesClient } from "../shape-inference";
import type { GraphDoc } from "../graph-ops";
import { configDiff, graphDiff, friendlyDiff, shapeProblems } from "./describe";

export type Decision = "applied" | "discarded";

interface ChangeSetState {
  /** The graph before the assistant's changes; null when nothing is pending */
  base: GraphDoc | null;
  baseTraining: TrainingConfig | null;
  /** Bumped on every change, so views can follow along */
  version: number;
  /** What the user did with the last change set, until the assistant has been told */
  lastDecision: Decision | null;
}

export const useChangeSetStore = create<ChangeSetState>(() => ({
  base: null,
  baseTraining: null,
  version: 0,
  lastDecision: null,
}));

export function currentDoc(): GraphDoc {
  const { nodes, edges, groups } = useGraphStore.getState();
  return { nodes, edges, groups };
}

export function hasPendingChanges(): boolean {
  return useChangeSetStore.getState().base !== null;
}

function begin() {
  if (hasPendingChanges()) return;
  pauseHistory(true);
  useChangeSetStore.setState({
    base: structuredClone(stripUi(currentDoc())),
    baseTraining: structuredClone(useTrainingStore.getState().config),
    lastDecision: null,
  });
}

/** The graph without selection and drag state, so the restored graph doesn't carry it */
function stripUi(doc: GraphDoc): GraphDoc {
  return {
    nodes: doc.nodes.map(({ selected: _s, dragging: _d, ...n }) => n),
    edges: doc.edges.map(({ selected: _s, ...e }) => e),
    groups: doc.groups,
  };
}

/** Puts a new graph on the canvas as part of the pending changes, and updates its shapes at once */
export function writeGraph(doc: GraphDoc) {
  begin();
  const { shapes, errors } = inferShapesClient(doc.nodes, doc.edges);
  useGraphStore.setState({ nodes: doc.nodes, edges: doc.edges, groups: doc.groups, shapeMap: shapes, shapeErrors: errors, selectedNodeId: null });
  useChangeSetStore.setState((s) => ({ version: s.version + 1 }));
}

export function writeTraining(patch: Partial<TrainingConfig>) {
  begin();
  useTrainingStore.getState().updateConfig(patch);
  useChangeSetStore.setState((s) => ({ version: s.version + 1 }));
}

/** What the pending changes do, for the review card and for the assistant */
export function pendingSummary(): { model: string[]; readable: string[]; training: string[]; newProblems: ReturnType<typeof shapeProblems> } | null {
  const { base, baseTraining } = useChangeSetStore.getState();
  if (!base) return null;
  const now = currentDoc();
  const g = useGraphStore.getState();
  const view = { ...now, shapeMap: g.shapeMap, shapeErrors: g.shapeErrors };
  const before = inferShapesClient(base.nodes, base.edges).errors;
  const newProblems = shapeProblems(view).filter((p) => p.kind === "cause" && !before[p.layer]);
  return {
    model: graphDiff(base, now),
    readable: friendlyDiff(base, now),
    training: baseTraining ? configDiff(baseTraining, useTrainingStore.getState().config) : [],
    newProblems,
  };
}

export function applyChanges() {
  if (!hasPendingChanges()) return;
  pauseHistory(false);
  // The graph before the changes is still the history's baseline, so this records them as one step
  useHistoryStore.getState().pushSnapshot();
  useChangeSetStore.setState({ base: null, baseTraining: null, lastDecision: "applied", version: useChangeSetStore.getState().version + 1 });
}

export function discardChanges() {
  const { base, baseTraining } = useChangeSetStore.getState();
  if (!base) return;
  const { shapes, errors } = inferShapesClient(base.nodes, base.edges);
  useGraphStore.setState({ nodes: base.nodes, edges: base.edges, groups: base.groups, shapeMap: shapes, shapeErrors: errors, selectedNodeId: null });
  if (baseTraining) useTrainingStore.getState().updateConfig(baseTraining);
  pauseHistory(false);
  useChangeSetStore.setState({ base: null, baseTraining: null, lastDecision: "discarded", version: useChangeSetStore.getState().version + 1 });
}

/**
 * Called before something replaces the whole graph (opening a project, a template, a new project):
 * the pending changes are kept as an undo step, so nothing is lost and undo can still reach them.
 */
export function settlePendingChanges() {
  if (hasPendingChanges()) applyChanges();
  useChangeSetStore.setState({ lastDecision: null });
}

/** Takes the user's last decision, to tell the assistant once */
export function takeDecision(): Decision | null {
  const d = useChangeSetStore.getState().lastDecision;
  if (d) useChangeSetStore.setState({ lastDecision: null });
  return d;
}
