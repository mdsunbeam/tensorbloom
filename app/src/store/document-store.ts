/**
 * The open project's file path and whether it has unsaved changes.
 * The project counts as saved while its graph, name, training settings and runs match what was last
 * saved, opened, or started fresh.
 */

import { create } from "zustand";
import { useGraphStore } from "./graph-store";
import { useTrainingStore } from "./training-store";
import { useRunStore } from "./run-store";
import { graphKey } from "./history-store";

interface DocumentState {
  /** Where Save writes without asking; null until the project is saved or opened from a file */
  filePath: string | null;
  dirty: boolean;
  /** Records the current project as saved. Pass a path to change filePath; leave it out to keep it. */
  markSaved: (filePath?: string | null) => void;
  markUnsaved: () => void;
}

let savedKey = "";

function documentKey(): string {
  const { projectName } = useGraphStore.getState();
  return JSON.stringify([graphKey(), projectName, useTrainingStore.getState().config, useRunStore.getState().runs.map((r) => r.id)]);
}

export const useDocumentStore = create<DocumentState>((set) => ({
  filePath: null,
  dirty: false,
  markSaved: (filePath) => {
    savedKey = documentKey();
    set((s) => ({ dirty: false, filePath: filePath === undefined ? s.filePath : filePath }));
  },
  markUnsaved: () => {
    savedKey = "";
    set({ dirty: true });
  },
}));

/** Keeps `dirty` current as the project changes. Returns a function that stops it. */
export function startDirtyTracking(): () => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const check = () => {
    timer = null;
    const dirty = documentKey() !== savedKey;
    if (dirty !== useDocumentStore.getState().dirty) useDocumentStore.setState({ dirty });
  };
  const schedule = () => { if (!timer) timer = setTimeout(check, 250); };
  const unsubGraph = useGraphStore.subscribe((s, p) => {
    if (s.nodes !== p.nodes || s.edges !== p.edges || s.groups !== p.groups || s.projectName !== p.projectName) schedule();
  });
  const unsubTraining = useTrainingStore.subscribe((s, p) => { if (s.config !== p.config) schedule(); });
  const unsubRuns = useRunStore.subscribe((s, p) => { if (s.runs !== p.runs) schedule(); });
  return () => {
    unsubGraph();
    unsubTraining();
    unsubRuns();
    if (timer) clearTimeout(timer);
  };
}
