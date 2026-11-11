/**
 * What changed between two setups of a model: its layers, their connections, and the training
 * settings. Used to compare runs and to keep a "change one thing" experiment to one change.
 */
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { TrainingConfig } from "../store/training-store";

export interface ParamChange {
  name: string;
  from: string;
  to: string;
}

export interface LayerChange {
  id: string;
  label: string;
  moduleType: string;
  kind: "added" | "removed" | "changed";
  /** For a changed layer: the settings that differ, a new type, or a new name */
  params?: ParamChange[];
  replaced?: { from: string; to: string };
  renamed?: { from: string; to: string };
}

export interface ConnectionChange {
  kind: "added" | "removed";
  from: string;
  to: string;
  /** Node ids at either end, so both can be marked on the canvas */
  source: string;
  target: string;
}

export interface SettingChange {
  label: string;
  from: string;
  to: string;
}

export interface SetupDiff {
  layers: LayerChange[];
  connections: ConnectionChange[];
  settings: SettingChange[];
}

type Graph = { nodes: Node<NodeData>[]; edges: Edge[] };

/** Shows a parameter value the way the Properties panel would */
export function formatValue(v: unknown): string {
  if (v === undefined || v === null || v === "") return "none";
  if (typeof v === "boolean") return v ? "on" : "off";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

const same = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** Layers matched by id: added, removed, or changed in type, name, settings, or code */
export function graphDiff(a: Graph, b: Graph): Pick<SetupDiff, "layers" | "connections"> {
  const aById = new Map(a.nodes.map((n) => [n.id, n]));
  const bById = new Map(b.nodes.map((n) => [n.id, n]));
  const layers: LayerChange[] = [];

  for (const nb of b.nodes) {
    const na = aById.get(nb.id);
    if (!na) {
      layers.push({ id: nb.id, label: nb.data.label, moduleType: nb.data.moduleType, kind: "added" });
      continue;
    }
    const change: LayerChange = { id: nb.id, label: nb.data.label, moduleType: nb.data.moduleType, kind: "changed" };
    let changed = false;
    if (na.data.moduleType !== nb.data.moduleType) {
      change.replaced = { from: na.data.moduleType, to: nb.data.moduleType };
      changed = true;
    } else {
      const names = [...new Set([...Object.keys(na.data.params ?? {}), ...Object.keys(nb.data.params ?? {})])];
      const params = names
        .filter((k) => !same(na.data.params?.[k], nb.data.params?.[k]))
        .map((k) => ({ name: k, from: formatValue(na.data.params?.[k]), to: formatValue(nb.data.params?.[k]) }));
      if (!same(na.data.customCode, nb.data.customCode) || !same(na.data.codeBlock, nb.data.codeBlock)) {
        params.push({ name: "code", from: "before", to: "edited" });
      }
      if (!same(na.data.dataOutputs, nb.data.dataOutputs)) {
        params.push({ name: "outputs", from: "before", to: "edited" });
      }
      if (params.length) {
        change.params = params;
        changed = true;
      }
    }
    if (na.data.label !== nb.data.label) {
      change.renamed = { from: na.data.label, to: nb.data.label };
      changed = true;
    }
    if (changed) layers.push(change);
  }
  for (const na of a.nodes) {
    if (!bById.has(na.id)) layers.push({ id: na.id, label: na.data.label, moduleType: na.data.moduleType, kind: "removed" });
  }

  // Connections between layers in both graphs only: a new or removed layer brings its own edges
  const key = (e: Edge) => `${e.source}:${e.sourceHandle ?? ""}>${e.target}:${e.targetHandle ?? ""}`;
  const aKeys = new Set(a.edges.map(key));
  const bKeys = new Set(b.edges.map(key));
  const inBoth = (id: string) => aById.has(id) && bById.has(id);
  const label = (id: string) => (bById.get(id) ?? aById.get(id))?.data.label ?? id;
  const connections: ConnectionChange[] = [];
  for (const e of b.edges) {
    if (!aKeys.has(key(e)) && inBoth(e.source) && inBoth(e.target)) {
      connections.push({ kind: "added", from: label(e.source), to: label(e.target), source: e.source, target: e.target });
    }
  }
  for (const e of a.edges) {
    if (!bKeys.has(key(e)) && inBoth(e.source) && inBoth(e.target)) {
      connections.push({ kind: "removed", from: label(e.source), to: label(e.target), source: e.source, target: e.target });
    }
  }
  return { layers, connections };
}

/** Training settings that differ. The device and resume checkpoint belong to the computer, not the experiment. */
export function settingsDiff(a: TrainingConfig, b: TrainingConfig): SettingChange[] {
  const rows: [string, (c: TrainingConfig) => unknown][] = [
    ["Optimizer", (c) => c.optimizer.type],
    ["Learning rate", (c) => c.optimizer.lr],
    ["Weight decay", (c) => c.optimizer.weightDecay],
    ["Momentum", (c) => (c.optimizer.type === "SGD" || c.optimizer.type === "RMSprop" ? c.optimizer.momentum ?? 0 : null)],
    ["LR scheduler", (c) => c.scheduler?.type ?? "None"],
    ["Epochs", (c) => c.epochs],
    ["Batch size", (c) => c.batchSize],
    ["Loss function", (c) => c.loss],
    ["Weight initialization", (c) => c.weightInit],
    ["Random seed", (c) => c.seed ?? "random"],
    ["Mixed precision", (c) => c.mixedPrecision],
    ["Gradient clipping", (c) => (c.gradientClipping.enabled ? `max norm ${c.gradientClipping.maxNorm}` : false)],
    ["Early stopping", (c) => (c.earlyStopping.enabled ? `patience ${c.earlyStopping.patience}` : false)],
    ["Flip", (c) => c.augmentation.horizontalFlip],
    ["Vertical flip", (c) => c.augmentation.verticalFlip],
    ["Random crop", (c) => c.augmentation.randomCrop],
    ["Color jitter", (c) => c.augmentation.colorJitter],
    ["Rotation", (c) => c.augmentation.rotation],
  ];
  return rows
    .filter(([, get]) => !same(get(a), get(b)))
    .map(([label, get]) => ({ label, from: formatValue(get(a)), to: formatValue(get(b)) }));
}

export function setupDiff(a: Graph & { config: TrainingConfig }, b: Graph & { config: TrainingConfig }): SetupDiff {
  return { ...graphDiff(a, b), settings: settingsDiff(a.config, b.config) };
}

/** How many separate things changed: each layer setting, layer, connection, and training setting counts once */
export function changeCount(d: SetupDiff): number {
  const layerChanges = d.layers.reduce((n, l) => n + (l.kind === "changed" ? (l.params?.length ?? 0) + (l.replaced ? 1 : 0) : 1), 0);
  return layerChanges + d.connections.length + d.settings.length;
}

/** Changes worth counting for an experiment: renaming a layer changes nothing about the model */
export function isEmpty(d: SetupDiff): boolean {
  return changeCount(d) === 0;
}

/** Each change in a few words, for lists and run names */
export function describeChanges(d: SetupDiff): string[] {
  const out: string[] = [];
  for (const s of d.settings) out.push(`${s.label} ${s.from} → ${s.to}`);
  for (const l of d.layers) {
    if (l.kind === "added") out.push(`Added ${l.label}`);
    else if (l.kind === "removed") out.push(`Removed ${l.label}`);
    else {
      if (l.replaced) out.push(`${l.label}: ${l.replaced.from} → ${l.replaced.to}`);
      for (const p of l.params ?? []) out.push(`${l.label} ${p.name} ${p.from} → ${p.to}`);
    }
  }
  for (const c of d.connections) out.push(`${c.kind === "added" ? "Connected" : "Disconnected"} ${c.from} → ${c.to}`);
  return out;
}

/** One line naming what changed: the change itself when there's one, else a count */
export function changeSummary(d: SetupDiff): string {
  const parts = describeChanges(d);
  if (parts.length === 0) return "No changes";
  if (parts.length === 1) return parts[0];
  return `${parts.length} changes: ${parts.slice(0, 2).join(", ")}${parts.length > 2 ? ", …" : ""}`;
}

/** How each node on the canvas differs, for marking it */
export function nodeMarks(d: SetupDiff): Map<string, "added" | "changed"> {
  const marks = new Map<string, "added" | "changed">();
  for (const l of d.layers) {
    if (l.kind === "added") marks.set(l.id, "added");
    else if (l.kind === "changed" && (l.params?.length || l.replaced)) marks.set(l.id, "changed");
  }
  for (const c of d.connections) {
    if (!marks.has(c.target)) marks.set(c.target, "changed");
  }
  return marks;
}
