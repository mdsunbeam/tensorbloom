import { create } from "zustand";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "./graph-store";
import type { TrainingConfig, TrainingMetrics, TrainingResults } from "./training-store";

/** One epoch's numbers, as the last report of that epoch gave them */
export interface EpochPoint {
  epoch: number;
  trainLoss: number;
  valLoss?: number;
  trainAcc?: number;
  valAcc?: number;
  lr: number;
}

/** A training run, kept with the project so runs can be compared and their model brought back */
export interface RunRecord {
  id: string;
  /** 1, 2, 3, ... in the order the project's runs started */
  number: number;
  /** The project's name when the run started, usually the model's */
  modelName: string;
  startedAt: string;
  durationMs: number;
  status: "finished" | "stopped" | "failed";
  error?: string;
  config: TrainingConfig;
  results?: TrainingResults;
  epochs: EpochPoint[];
  model: { params: number; layers: number; /** Same key ⇔ same layers, settings and connections */ key: string };
  /** The graph the run trained, so it can be opened again */
  graph: { nodes: Node<NodeData>[]; edges: Edge[] };
}

/** Projects keep this many runs; older ones are dropped */
export const MAX_RUNS = 30;

interface RunState {
  runs: RunRecord[];
  /** Runs picked for comparison, in the order they were picked */
  compareIds: string[];
  addRun: (run: RunRecord) => void;
  removeRun: (id: string) => void;
  setRuns: (runs: RunRecord[]) => void;
  toggleCompare: (id: string) => void;
  setCompare: (ids: string[]) => void;
}

export const useRunStore = create<RunState>((set, get) => ({
  runs: [],
  compareIds: [],
  // A new run is compared with the one before it, which is what people usually want to see
  addRun: (run) => {
    const runs = [...get().runs, run].slice(-MAX_RUNS);
    const prev = runs[runs.length - 2];
    set({ runs, compareIds: prev ? [prev.id, run.id] : [run.id] });
  },
  removeRun: (id) => set({ runs: get().runs.filter((r) => r.id !== id), compareIds: get().compareIds.filter((c) => c !== id) }),
  setRuns: (runs) => set({ runs: runs.slice(-MAX_RUNS), compareIds: runs.length ? [runs[runs.length - 1].id] : [] }),
  toggleCompare: (id) => {
    const ids = get().compareIds;
    set({ compareIds: ids.includes(id) ? ids.filter((c) => c !== id) : [...ids, id] });
  },
  setCompare: (compareIds) => set({ compareIds }),
}));

/** Reduces step-level reports to one point per epoch, taking each epoch's last values */
export function epochSummary(metrics: TrainingMetrics[]): EpochPoint[] {
  const byEpoch = new Map<number, EpochPoint>();
  for (const m of metrics) {
    const e = byEpoch.get(m.epoch) ?? { epoch: m.epoch, trainLoss: m.trainLoss, lr: m.lr };
    e.trainLoss = m.trainLoss;
    e.lr = m.lr;
    if (m.valLoss != null) e.valLoss = m.valLoss;
    if (m.trainAcc != null) e.trainAcc = m.trainAcc;
    if (m.valAcc != null) e.valAcc = m.valAcc;
    byEpoch.set(m.epoch, e);
  }
  return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
}

/** Identifies a model by its layers, their settings and connections, not where they sit on the canvas */
export function architectureKey(nodes: Node<NodeData>[], edges: Edge[]): string {
  const ns = [...nodes]
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((n) => [n.id, n.data.moduleType, n.data.params, n.data.dataOutputs ?? null, n.data.codeBlock ?? null]);
  const es = edges
    .map((e) => `${e.source}:${e.sourceHandle ?? ""}>${e.target}:${e.targetHandle ?? ""}`)
    .sort();
  return JSON.stringify([ns, es]);
}

/** A copy of the graph without the canvas's selection and measurement state */
export function graphSnapshot(nodes: Node<NodeData>[], edges: Edge[]): RunRecord["graph"] {
  return {
    nodes: nodes.map((n) => ({ id: n.id, type: n.type, position: n.position, data: n.data, ...(n.parentId ? { parentId: n.parentId } : {}) })),
    edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle, type: e.type })),
  };
}

/** The headline number for a run: best validation accuracy, else best validation loss, else final loss */
export function runHeadline(run: RunRecord): { label: string; value: string } | null {
  const r = run.results;
  if (!r) return null;
  if (r.bestValAcc != null) return { label: "val acc", value: `${(r.bestValAcc * 100).toFixed(1)}%` };
  if (r.bestValLoss != null) return { label: "val loss", value: formatNumber(r.bestValLoss) };
  return { label: "loss", value: formatNumber(r.finalTrainLoss) };
}

function formatNumber(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(3) : v.toPrecision(3);
}
