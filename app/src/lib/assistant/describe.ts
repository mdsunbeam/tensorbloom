/**
 * Compact, model-readable descriptions of the user's work: the graph, its shape problems, the training
 * settings and runs. Pure functions of store state, so they're easy to test and to reuse for previews.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../../store/graph-store";
import type { TrainingConfig } from "../../store/training-store";
import type { RunRecord } from "../../store/run-store";
import { runHeadline } from "../../store/run-store";
import type { ShapeMap, ShapeError } from "../shape-inference";
import { getOutputShape } from "../shape-inference";
import { shapeErrorFix } from "../shape-fixes";
import type { LayerGroup } from "../groups";
import { computeModelSummary } from "../model-summary";

export interface GraphView {
  nodes: Node<NodeData>[];
  edges: Edge[];
  groups: LayerGroup[];
  shapeMap: ShapeMap;
  shapeErrors: Record<string, ShapeError>;
}

/** The layer's settings, without the app's cached scan results (keys starting with "_") */
export function visibleParams(params: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    if (k.startsWith("_") || v === undefined) continue;
    out[k] = v;
  }
  return out;
}

function formatValue(v: unknown): string {
  if (typeof v === "string") return /^[[{(\d-]/.test(v) || v === "" ? v || '""' : v;
  return JSON.stringify(v);
}

/** Layers in the order data flows through them, so the description reads top to bottom */
export function flowOrder(nodes: Node<NodeData>[], edges: Edge[]): Node<NodeData>[] {
  const indeg = new Map(nodes.map((n) => [n.id, 0]));
  const out = new Map<string, string[]>();
  for (const e of edges) {
    if (!indeg.has(e.source) || !indeg.has(e.target)) continue;
    indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1);
    out.set(e.source, [...(out.get(e.source) ?? []), e.target]);
  }
  // Ties go to the layer higher on the canvas, which is how people read it
  const byPos = (a: Node<NodeData>, b: Node<NodeData>) => a.position.y - b.position.y || a.position.x - b.position.x;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const ready = nodes.filter((n) => indeg.get(n.id) === 0).sort(byPos);
  const order: Node<NodeData>[] = [];
  while (ready.length) {
    const n = ready.shift()!;
    order.push(n);
    for (const t of out.get(n.id) ?? []) {
      indeg.set(t, (indeg.get(t) ?? 0) - 1);
      if (indeg.get(t) === 0) { ready.push(byId.get(t)!); ready.sort(byPos); }
    }
  }
  // Layers in a cycle are left over; list them anyway
  if (order.length < nodes.length) {
    const seen = new Set(order.map((n) => n.id));
    order.push(...nodes.filter((n) => !seen.has(n.id)).sort(byPos));
  }
  return order;
}

function inputsOf(nodeId: string, edges: Edge[]): string[] {
  return edges
    .filter((e) => e.target === nodeId)
    .map((e) => {
      const from = e.sourceHandle && e.sourceHandle !== "out" ? `${e.source}.${e.sourceHandle}` : e.source;
      return e.targetHandle && e.targetHandle !== "in" ? `${from}→${e.targetHandle}` : from;
    });
}

/**
 * One line per layer, e.g.
 * `node_3 Conv2d "Conv2d" (in_channels=3, out_channels=16, kernel_size=3) ← node_2 → [1, 16, 30, 30]`
 */
export function describeGraph(view: GraphView): string {
  const { nodes, edges, groups, shapeMap, shapeErrors } = view;
  if (nodes.length === 0) return "The canvas is empty.";
  const groupOf = new Map<string, string>();
  for (const g of groups) for (const id of g.nodeIds) groupOf.set(id, g.name);
  const summary = computeModelSummary(nodes);
  const lines = [`${nodes.length} layers, ${edges.length} connections, ${summary.totalParams.toLocaleString("en-US")} parameters. Shapes use a batch size of 1.`];
  for (const n of flowOrder(nodes, edges)) {
    const d = n.data;
    const params = Object.entries(visibleParams(d.params)).map(([k, v]) => `${k}=${formatValue(v)}`).join(", ");
    const parts = [`${n.id} ${d.moduleType}`];
    if (d.label && d.label !== d.moduleType) parts.push(JSON.stringify(d.label));
    if (params) parts.push(`(${params})`);
    if (d.codeBlock) parts.push("[custom code]");
    if (d.customCode) parts.push(`[constructor: ${d.customCode}]`);
    const ins = inputsOf(n.id, edges);
    if (ins.length) parts.push(`← ${ins.join(", ")}`);
    const shape = getOutputShape(shapeMap, n.id);
    if (shape) parts.push(`→ [${shape.shape.join(", ")}]`);
    const g = groupOf.get(n.id);
    if (g) parts.push(`{group: ${g}}`);
    if (shapeErrors[n.id]?.type === "root") parts.push("⚠ shape error");
    lines.push(parts.join(" "));
  }
  return lines.join("\n");
}

/** The graph as structured data, for the get_graph tool */
export function graphJson(view: GraphView) {
  const { nodes, edges, groups, shapeMap } = view;
  return {
    layers: flowOrder(nodes, edges).map((n) => ({
      id: n.id,
      type: n.data.moduleType,
      label: n.data.label,
      params: visibleParams(n.data.params),
      ...(n.data.codeBlock ? { code: { init: n.data.codeBlock.initBody, forward: n.data.codeBlock.forwardBody, inputs: n.data.codeBlock.inputHandles.map((h) => h.id), outputs: n.data.codeBlock.outputHandles.map((h) => h.id) } } : {}),
      ...(n.data.customCode ? { constructor: n.data.customCode } : {}),
      ...(n.data.dataOutputs ? { dataOutputs: n.data.dataOutputs.map((o) => ({ id: o.id, column: o.column, dtype: o.dtype })) } : {}),
      outputShape: getOutputShape(shapeMap, n.id)?.shape ?? null,
    })),
    connections: edges.map((e) => ({ from: e.source, fromHandle: e.sourceHandle ?? "out", to: e.target, toHandle: e.targetHandle ?? "in" })),
    groups: groups.map((g) => ({ name: g.name, layers: g.nodeIds, collapsed: g.collapsed })),
  };
}

/** Shape problems, root causes first, each with its one-click fix when there is one */
export function shapeProblems(view: GraphView) {
  const byId = new Map(view.nodes.map((n) => [n.id, n]));
  return Object.entries(view.shapeErrors)
    .sort(([, a], [, b]) => (a.type === b.type ? 0 : a.type === "root" ? -1 : 1))
    .map(([nodeId, e]) => {
      const node = byId.get(nodeId);
      const fix = node ? shapeErrorFix(e, node.data.params) : null;
      return {
        layer: nodeId,
        layerType: node?.data.moduleType ?? null,
        kind: e.type === "root" ? "cause" : "caused by an earlier error",
        message: e.message,
        ...(e.type === "cascaded" ? { causedBy: e.sourceNodeId } : {}),
        ...(e.receivedShape ? { receives: e.receivedShape } : {}),
        ...(e.expectedShape ? { expects: e.expectedShape } : {}),
        ...(fix ? { suggestedFix: { param: fix.param, from: fix.from, to: fix.to } } : {}),
      };
    });
}

export function describeTraining(c: TrainingConfig): string {
  const parts = [
    `optimizer ${c.optimizer.type} (lr ${c.optimizer.lr}, weight decay ${c.optimizer.weightDecay}${c.optimizer.momentum != null ? `, momentum ${c.optimizer.momentum}` : ""})`,
    c.scheduler ? `scheduler ${c.scheduler.type} ${JSON.stringify(c.scheduler.params)}` : "no scheduler",
    `loss ${c.loss}`,
    `${c.epochs} epochs`,
    `batch size ${c.batchSize}`,
    `validation split ${c.valSplit}`,
    `device ${c.device}`,
  ];
  if (c.mixedPrecision) parts.push("mixed precision");
  if (c.gradientClipping.enabled) parts.push(`gradient clipping ${c.gradientClipping.maxNorm}`);
  if (c.earlyStopping.enabled) parts.push(`early stopping (patience ${c.earlyStopping.patience})`);
  if (c.weightInit !== "default") parts.push(`init ${c.weightInit}`);
  if (c.seed != null) parts.push(`seed ${c.seed}`);
  const aug = Object.entries(c.augmentation).filter(([, v]) => v).map(([k, v]) => (typeof v === "number" ? `${k} ${v}` : k));
  if (aug.length) parts.push(`augmentation: ${aug.join(", ")}`);
  return parts.join("; ");
}

/** One line per run, newest last */
export function describeRun(run: RunRecord): string {
  const head = runHeadline(run);
  const secs = Math.round(run.durationMs / 1000);
  const r = run.results;
  const bits = [
    `Run ${run.number}`,
    run.status,
    `${run.epochs.length} epochs`,
    `${secs}s`,
    `${run.model.params.toLocaleString("en-US")} params`,
    head ? `best ${head.label} ${head.value}` : null,
    r?.finalTrainLoss != null ? `final train loss ${round(r.finalTrainLoss)}` : null,
    r?.finalValLoss != null ? `final val loss ${round(r.finalValLoss)}` : null,
    r?.earlyStopped ? "stopped early" : null,
    run.error ? `error: ${run.error.slice(0, 160)}` : null,
  ];
  return bits.filter(Boolean).join(", ");
}

export function round(v: number): number {
  return Math.abs(v) >= 100 ? Math.round(v) : Number(v.toPrecision(4));
}

/** The settings two runs differ in, as "path: a → b" */
export function configDiff(a: TrainingConfig, b: TrainingConfig): string[] {
  const out: string[] = [];
  const walk = (x: unknown, y: unknown, path: string) => {
    if (x && y && typeof x === "object" && typeof y === "object" && !Array.isArray(x)) {
      const keys = new Set([...Object.keys(x as object), ...Object.keys(y as object)]);
      for (const k of keys) walk((x as Record<string, unknown>)[k], (y as Record<string, unknown>)[k], path ? `${path}.${k}` : k);
      return;
    }
    if (JSON.stringify(x) !== JSON.stringify(y)) out.push(`${path}: ${JSON.stringify(x)} → ${JSON.stringify(y)}`);
  };
  walk(a, b, "");
  return out;
}

/** What changed in the model between two graphs, by layer id */
export function graphDiff(a: { nodes: Node<NodeData>[]; edges: Edge[] }, b: { nodes: Node<NodeData>[]; edges: Edge[] }): string[] {
  const out: string[] = [];
  const am = new Map(a.nodes.map((n) => [n.id, n]));
  const bm = new Map(b.nodes.map((n) => [n.id, n]));
  for (const [id, n] of bm) {
    const o = am.get(id);
    if (!o) { out.push(`added ${id} ${n.data.moduleType}`); continue; }
    if (o.data.moduleType !== n.data.moduleType) { out.push(`${id}: ${o.data.moduleType} → ${n.data.moduleType}`); continue; }
    const op = visibleParams(o.data.params), np = visibleParams(n.data.params);
    for (const k of new Set([...Object.keys(op), ...Object.keys(np)])) {
      if (JSON.stringify(op[k]) !== JSON.stringify(np[k])) out.push(`${id} ${n.data.moduleType}.${k}: ${formatValue(op[k])} → ${formatValue(np[k])}`);
    }
    if (JSON.stringify(o.data.codeBlock ?? null) !== JSON.stringify(n.data.codeBlock ?? null)) out.push(`${id}: custom code changed`);
  }
  for (const [id, n] of am) if (!bm.has(id)) out.push(`removed ${id} ${n.data.moduleType}`);
  const key = (e: Edge) => `${e.source}:${e.sourceHandle ?? "out"}>${e.target}:${e.targetHandle ?? "in"}`;
  const ae = new Set(a.edges.map(key)), be = new Set(b.edges.map(key));
  for (const k of be) if (!ae.has(k)) out.push(`connected ${k}`);
  for (const k of ae) if (!be.has(k)) out.push(`disconnected ${k}`);
  return out;
}

/** What changed, worded for the person reviewing it: layers by their names, connections as a count */
export function friendlyDiff(a: { nodes: Node<NodeData>[]; edges: Edge[] }, b: { nodes: Node<NodeData>[]; edges: Edge[] }): string[] {
  const out: string[] = [];
  const am = new Map(a.nodes.map((n) => [n.id, n]));
  const bm = new Map(b.nodes.map((n) => [n.id, n]));
  const name = (n: Node<NodeData>) => (n.data.label && n.data.label !== n.data.moduleType ? `${n.data.label} (${n.data.moduleType})` : n.data.moduleType);
  for (const [id, n] of bm) {
    const o = am.get(id);
    if (!o) {
      const into = b.edges.find((e) => e.target === id);
      const from = into ? bm.get(into.source) : undefined;
      out.push(`Add ${name(n)}${from ? ` after ${from.data.label}` : ""}`);
      continue;
    }
    if (o.data.label !== n.data.label) out.push(`Rename ${o.data.label} to ${n.data.label}`);
    const op = visibleParams(o.data.params), np = visibleParams(n.data.params);
    const changed = [...new Set([...Object.keys(op), ...Object.keys(np)])].filter((k) => JSON.stringify(op[k]) !== JSON.stringify(np[k]));
    if (changed.length) out.push(`${n.data.label}: ${changed.map((k) => `${k} ${formatValue(op[k])} → ${formatValue(np[k])}`).join(", ")}`);
    if (JSON.stringify(o.data.codeBlock ?? null) !== JSON.stringify(n.data.codeBlock ?? null)) out.push(`${n.data.label}: new code`);
  }
  for (const [id, n] of am) if (!bm.has(id)) out.push(`Remove ${name(n)}`);
  const key = (e: Edge) => `${e.source}:${e.sourceHandle ?? "out"}>${e.target}:${e.targetHandle ?? "in"}`;
  const ae = new Set(a.edges.map(key)), be = new Set(b.edges.map(key));
  const rewired = [...be].filter((k) => !ae.has(k)).length + [...ae].filter((k) => !be.has(k)).length;
  // Connections that come with added or removed layers go without saying; say only when others changed
  const implied = (k: string) => { const [s, t] = k.split(">").map((x) => x.split(":")[0]); return !am.has(s) || !am.has(t) || !bm.has(s) || !bm.has(t); };
  // A connection replaced by a layer inserted into it goes without saying too
  const ends = (k: string) => k.split(">").map((x) => x.split(":")[0]);
  const bridged = (k: string) => {
    const [s, t] = ends(k);
    return [...bm.keys()].some((x) => !am.has(x) && [...be].some((e) => ends(e)[0] === s && ends(e)[1] === x) && [...be].some((e) => ends(e)[0] === x && ends(e)[1] === t));
  };
  const other = [...be].filter((k) => !ae.has(k) && !implied(k)).length + [...ae].filter((k) => !be.has(k) && !implied(k) && !bridged(k)).length;
  if (rewired && other) out.push(`${other} ${other === 1 ? "connection" : "connections"} changed`);
  return out;
}
