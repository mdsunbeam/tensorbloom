/** Applies lesson edits to a graph and training settings, without touching the stores */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../../store/graph-store";
import type { TrainingConfig } from "../../store/training-store";
import type { NodeCategory } from "../types";
import { DEFAULT_DATA_OUTPUTS } from "../types";
import { MODULE_REGISTRY } from "../registry";
import type { LayerSelector, LessonEdit, TrainingField } from "./types";

export interface LessonGraph {
  nodes: Node<NodeData>[];
  edges: Edge[];
}

export function matches(node: Node<NodeData>, select: LayerSelector): boolean {
  if (select.label != null && node.data.label !== select.label) return false;
  if (select.moduleType != null && node.data.moduleType !== select.moduleType) return false;
  return true;
}

export function selectorText(select: LayerSelector): string {
  if (select.label && select.moduleType) return `${select.label} (${select.moduleType})`;
  return select.label ?? select.moduleType ?? "any layer";
}

/** Reads a setting by its dotted path, e.g. "optimizer.lr" */
export function getTrainingField(config: TrainingConfig, field: TrainingField): unknown {
  if (field === "scheduler") return config.scheduler?.type ?? "None";
  let v: unknown = config;
  for (const key of field.split(".")) v = (v as Record<string, unknown> | undefined)?.[key];
  return v;
}

function setTrainingField(config: TrainingConfig, field: TrainingField, value: unknown): TrainingConfig {
  if (field === "scheduler") return { ...config, scheduler: value === "None" ? null : { type: String(value), params: {} } };
  const [head, tail] = field.split(".");
  if (!tail) return { ...config, [head]: value };
  const inner = (config as unknown as Record<string, Record<string, unknown>>)[head] ?? {};
  return { ...config, [head]: { ...inner, [tail]: value } };
}

function newNode(id: string, moduleType: string, label?: string, params?: Record<string, unknown>): Node<NodeData> {
  const def = MODULE_REGISTRY[moduleType];
  if (!def) throw new Error(`Unknown module "${moduleType}"`);
  const defaults: Record<string, unknown> = {};
  for (const p of def.params) defaults[p.name] = p.default;
  return {
    id,
    type: "tensorLabNode",
    position: { x: 0, y: 0 },
    data: {
      moduleType,
      label: label ?? def.label,
      category: def.category as NodeCategory,
      params: { ...defaults, ...params },
      ...(moduleType === "Data" ? { dataOutputs: [...DEFAULT_DATA_OUTPUTS] } : {}),
    },
  };
}

function freshId(nodes: Node<NodeData>[], prefix = "lesson"): string {
  const taken = new Set(nodes.map((n) => n.id));
  let i = nodes.length + 1;
  while (taken.has(`${prefix}_${i}`)) i++;
  return `${prefix}_${i}`;
}

function edgeId(source: string, sourceHandle: string, target: string, targetHandle: string) {
  return `e_${source}_${sourceHandle}_${target}_${targetHandle}`;
}

function selected(graph: LessonGraph, target: LayerSelector, edit: LessonEdit): Node<NodeData>[] {
  const hits = graph.nodes.filter((n) => matches(n, target));
  if (hits.length === 0) throw new Error(`${edit.op}: no layer matches ${selectorText(target)}`);
  return hits;
}

/**
 * Applies edits in order. Throws when an edit names a layer that isn't there, so a lesson
 * that drifts from its template fails its tests instead of starting broken.
 */
export function applyEdits(
  graph: LessonGraph,
  config: TrainingConfig,
  edits: LessonEdit[],
): { graph: LessonGraph; config: TrainingConfig } {
  let nodes = graph.nodes.map((n) => ({ ...n, data: { ...n.data, params: { ...n.data.params } } }));
  let edges = [...graph.edges];
  let cfg = config;

  for (const edit of edits) {
    const g = { nodes, edges };
    switch (edit.op) {
      case "set_param": {
        const ids = new Set(selected(g, edit.target, edit).map((n) => n.id));
        nodes = nodes.map((n) => (ids.has(n.id) ? { ...n, data: { ...n.data, params: { ...n.data.params, [edit.param]: edit.value } } } : n));
        break;
      }
      case "remove_layer": {
        for (const node of selected(g, edit.target, edit)) {
          const inbound = edges.filter((e) => e.target === node.id);
          const outbound = edges.filter((e) => e.source === node.id);
          edges = edges.filter((e) => e.source !== node.id && e.target !== node.id);
          // Bridge the gap: the layer's main input feeds whatever the layer fed
          const main = inbound.find((e) => (e.targetHandle ?? "in") === "in") ?? inbound[0];
          if (main) {
            for (const out of outbound) {
              const sh = main.sourceHandle ?? "out";
              const th = out.targetHandle ?? "in";
              edges.push({ id: edgeId(main.source, sh, out.target, th), source: main.source, target: out.target, sourceHandle: sh, targetHandle: th, type: "smoothstep" });
            }
          }
          nodes = nodes.filter((n) => n.id !== node.id);
        }
        break;
      }
      case "replace_layer": {
        const ids = new Set(selected(g, edit.target, edit).map((n) => n.id));
        nodes = nodes.map((n) => {
          if (!ids.has(n.id)) return n;
          const fresh = newNode(n.id, edit.module, edit.label, edit.params);
          return { ...fresh, position: n.position };
        });
        break;
      }
      case "insert_after": {
        const hits = selected(g, edit.target, edit);
        if (hits.length > 1) throw new Error(`insert_after: ${hits.length} layers match ${selectorText(edit.target)}; name one`);
        const anchor = hits[0];
        const added = newNode(freshId(nodes), edit.module, edit.label, edit.params);
        const anchorHandle = anchor.data.dataOutputs ? "x" : "out";
        edges = edges.map((e) =>
          e.source === anchor.id && (e.sourceHandle ?? "out") === anchorHandle
            ? { ...e, id: edgeId(added.id, "out", e.target, e.targetHandle ?? "in"), source: added.id, sourceHandle: "out" }
            : e,
        );
        edges.push({ id: edgeId(anchor.id, anchorHandle, added.id, "in"), source: anchor.id, target: added.id, sourceHandle: anchorHandle, targetHandle: "in", type: "smoothstep" });
        nodes = [...nodes, { ...added, position: { x: anchor.position.x, y: anchor.position.y + 40 } }];
        break;
      }
      case "set_training":
        cfg = setTrainingField(cfg, edit.field, edit.value);
        break;
    }
  }
  return { graph: { nodes, edges }, config: cfg };
}
