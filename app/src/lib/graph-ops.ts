/**
 * Plain graph operations: each takes the graph, explicit layer ids and checked settings, and returns
 * a new graph plus the ids it created. No canvas positions from the mouse, no selection.
 *
 * The graph store's mouse-driven actions use these, and so does the assistant, so a change made either
 * way follows the same rules. Invalid requests throw a GraphOpError whose message says what's wrong.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { ParamDef } from "./types";
import { DEFAULT_DATA_OUTPUTS, resolveHandlesWithRegistry } from "./types";
import { MODULE_REGISTRY } from "./registry";
import { isDatasetNode, DATASET_METADATA, resolveDataNodeShape } from "./dataset-metadata";
import { addGroup, nextGroupName, pruneGroups, type LayerGroup } from "./groups";

export type TLNode = Node<NodeData>;

export interface GraphDoc {
  nodes: TLNode[];
  edges: Edge[];
  groups: LayerGroup[];
}

export class GraphOpError extends Error {}

// ---------------------------------------------------------------------------
// Layer ids
// ---------------------------------------------------------------------------

let nodeIdCounter = 0;

export function nextNodeId(): string {
  return `node_${++nodeIdCounter}`;
}

/** Keeps new ids clear of the ones a loaded graph already uses */
export function syncNodeIds(nodes: { id: string }[]) {
  for (const n of nodes) {
    const m = n.id.match(/^node_(\d+)$/);
    if (m) nodeIdCounter = Math.max(nodeIdCounter, parseInt(m[1], 10));
  }
}

export function resetNodeIds() {
  nodeIdCounter = 0;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

function describeType(p: ParamDef): string {
  switch (p.type) {
    case "int": return "a whole number";
    case "float": return "a number";
    case "bool": return "true or false";
    case "select": return `one of ${(p.options ?? []).map((o) => JSON.stringify(o)).join(", ")}`;
    case "tuple_int": return "a list of whole numbers, e.g. [3, 3]";
    case "optional_int": return "a whole number or null";
    default: return "text";
  }
}

/** Checks one setting against its definition. Returns the value to store, or throws. */
export function checkParam(moduleType: string, def: ParamDef, value: unknown): unknown {
  const fail = () => {
    throw new GraphOpError(`${moduleType}.${def.name} must be ${describeType(def)}; got ${JSON.stringify(value)}.`);
  };
  switch (def.type) {
    case "int":
      if (!Number.isInteger(value)) fail();
      break;
    case "float":
      if (typeof value !== "number" || !Number.isFinite(value)) fail();
      break;
    case "bool":
      if (typeof value !== "boolean") fail();
      break;
    case "select":
      if (!(def.options ?? []).includes(value as string)) fail();
      break;
    case "tuple_int":
      if (Number.isInteger(value)) return value;
      if (!Array.isArray(value) || !value.every((v) => Number.isInteger(v))) fail();
      break;
    case "optional_int":
      if (value !== null && !Number.isInteger(value)) fail();
      break;
    case "string":
      // Shapes and similar settings are stored as text, e.g. "[1, 3, 32, 32]"
      if (Array.isArray(value)) return JSON.stringify(value);
      if (typeof value !== "string") fail();
      break;
  }
  if (typeof value === "number") {
    if (def.min != null && value < def.min) throw new GraphOpError(`${moduleType}.${def.name} must be at least ${def.min}.`);
    if (def.max != null && value > def.max) throw new GraphOpError(`${moduleType}.${def.name} must be at most ${def.max}.`);
  }
  return value;
}

/**
 * Checks settings for a layer type and returns them ready to store. Unknown names are rejected with
 * the names the layer does have, so nothing invents a setting PyTorch wouldn't accept.
 */
export function checkParams(moduleType: string, params: Record<string, unknown>): Record<string, unknown> {
  const def = MODULE_REGISTRY[moduleType];
  if (!def) throw new GraphOpError(`"${moduleType}" isn't a layer type.`);
  const out: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(params)) {
    const p = def.params.find((x) => x.name === name);
    if (!p) {
      // The Data layer has settings that depend on its source, which the registry doesn't list
      if (moduleType === "Data") { out[name] = value; continue; }
      const known = def.params.map((x) => x.name);
      throw new GraphOpError(`${moduleType} has no setting "${name}". ${known.length ? `Its settings: ${known.join(", ")}.` : "It has no settings."}`);
    }
    out[name] = checkParam(moduleType, p, value);
  }
  return out;
}

export function defaultParams(moduleType: string): Record<string, unknown> {
  const def = MODULE_REGISTRY[moduleType];
  const out: Record<string, unknown> = {};
  for (const p of def?.params ?? []) out[p.name] = p.default;
  return out;
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

export interface NewLayer {
  moduleType: string;
  params?: Record<string, unknown>;
  label?: string;
  position: { x: number; y: number };
  id?: string;
}

/** A new layer with default settings, overridden by any given */
export function createNode({ moduleType, params = {}, label, position, id }: NewLayer): TLNode {
  const def = MODULE_REGISTRY[moduleType];
  if (!def) throw new GraphOpError(`"${moduleType}" isn't a layer type.`);
  return {
    id: id ?? nextNodeId(),
    type: "tensorLabNode",
    position,
    data: {
      moduleType: def.type,
      label: label?.trim() || def.label,
      category: def.category,
      params: { ...defaultParams(moduleType), ...checkParams(moduleType, params) },
      // Data layers get explicit output handles
      ...(moduleType === "Data" ? { dataOutputs: [...DEFAULT_DATA_OUTPUTS] } : {}),
    },
  };
}

export function addLayer(doc: GraphDoc, layer: NewLayer): { doc: GraphDoc; id: string } {
  const node = createNode(layer);
  if (doc.nodes.some((n) => n.id === node.id)) throw new GraphOpError(`There's already a layer ${node.id}.`);
  return { doc: { ...doc, nodes: [...doc.nodes, node] }, id: node.id };
}

export function requireNode(doc: GraphDoc, id: string): TLNode {
  const n = doc.nodes.find((x) => x.id === id);
  if (!n) throw new GraphOpError(`There's no layer ${id}.`);
  return n;
}

/** Removes layers and their connections. With `bridge`, a removed layer's single input is connected to whatever it fed. */
export function removeLayers(doc: GraphDoc, ids: string[], { bridge = false } = {}): GraphDoc {
  const gone = new Set(ids);
  for (const id of ids) requireNode(doc, id);
  let edges = doc.edges;
  if (bridge) {
    for (const id of ids) {
      const ins = edges.filter((e) => e.target === id && !gone.has(e.source));
      const outs = edges.filter((e) => e.source === id);
      if (ins.length !== 1) continue;
      const src = ins[0];
      for (const o of outs) {
        edges = [...edges, makeEdge(src.source, src.sourceHandle ?? "out", o.target, o.targetHandle ?? "in")];
      }
      edges = edges.filter((e) => e.source !== id && e.target !== id);
    }
  }
  const nodes = doc.nodes.filter((n) => !gone.has(n.id));
  return {
    nodes,
    edges: edges.filter((e) => !gone.has(e.source) && !gone.has(e.target)),
    groups: pruneGroups(doc.groups, nodes),
  };
}

export function setParams(doc: GraphDoc, id: string, params: Record<string, unknown>): GraphDoc {
  const node = requireNode(doc, id);
  if (node.data.codeBlock) throw new GraphOpError(`${id} runs custom code; its settings are in that code.`);
  const checked = checkParams(node.data.moduleType, params);
  return {
    ...doc,
    nodes: doc.nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, params: { ...n.data.params, ...checked } } } : n)),
  };
}

export function renameLayer(doc: GraphDoc, id: string, label: string): GraphDoc {
  requireNode(doc, id);
  const trimmed = label.trim();
  if (!trimmed) throw new GraphOpError("A layer's name can't be empty.");
  return { ...doc, nodes: doc.nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, label: trimmed } } : n)) };
}

// ---------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------

export function makeEdge(source: string, sourceHandle: string, target: string, targetHandle: string): Edge {
  return {
    id: `e_${source}_${sourceHandle}_${target}_${targetHandle}`,
    source,
    sourceHandle,
    target,
    targetHandle,
    type: "smoothstep",
  };
}

export function handlesOf(node: TLNode) {
  return resolveHandlesWithRegistry(node.data, MODULE_REGISTRY);
}

export interface Connect {
  from: string;
  to: string;
  fromHandle?: string;
  toHandle?: string;
  /** Replace whatever is connected to the target handle now, instead of failing */
  replace?: boolean;
}

/**
 * Connects two layers. Handles default to the first output and the first free input. A Data layer
 * connected to an Input sets the Input's shape, as dragging the connection does.
 */
export function connect(doc: GraphDoc, c: Connect): GraphDoc {
  const src = requireNode(doc, c.from);
  const tgt = requireNode(doc, c.to);
  if (src.id === tgt.id) throw new GraphOpError("A layer can't connect to itself.");
  const outs = handlesOf(src).outputHandles;
  const ins = handlesOf(tgt).inputHandles;
  if (!outs.length) throw new GraphOpError(`${src.id} (${src.data.moduleType}) has no outputs.`);
  if (!ins.length) throw new GraphOpError(`${tgt.id} (${tgt.data.moduleType}) has no inputs.`);
  const fromHandle = c.fromHandle ?? outs[0].id;
  if (!outs.some((h) => h.id === fromHandle)) {
    throw new GraphOpError(`${src.id} has no output "${fromHandle}". Its outputs: ${outs.map((h) => h.id).join(", ")}.`);
  }
  const taken = new Set(doc.edges.filter((e) => e.target === tgt.id).map((e) => e.targetHandle ?? "in"));
  const toHandle = c.toHandle ?? (ins.find((h) => !taken.has(h.id)) ?? ins[0]).id;
  if (!ins.some((h) => h.id === toHandle)) {
    throw new GraphOpError(`${tgt.id} has no input "${toHandle}". Its inputs: ${ins.map((h) => h.id).join(", ")}.`);
  }
  let edges = doc.edges;
  const occupied = edges.find((e) => e.target === tgt.id && (e.targetHandle ?? "in") === toHandle);
  if (occupied) {
    if (occupied.source === src.id && (occupied.sourceHandle ?? "out") === fromHandle) return doc;
    if (!c.replace) {
      throw new GraphOpError(`${tgt.id}'s input "${toHandle}" is already connected to ${occupied.source}. Disconnect it first, or ask to replace it.`);
    }
    edges = edges.filter((e) => e !== occupied);
  }
  if (createsCycle(edges, src.id, tgt.id)) throw new GraphOpError(`Connecting ${src.id} to ${tgt.id} would make a loop.`);
  edges = [...edges, makeEdge(src.id, fromHandle, tgt.id, toHandle)];

  let nodes = doc.nodes;
  if (isDatasetNode(src.data.moduleType) && tgt.data.moduleType === "Input") {
    const shape = src.data.moduleType === "Data"
      ? resolveDataNodeShape(src.data.params)?.inputShape
      : DATASET_METADATA[src.data.moduleType]?.inputShape;
    if (shape) nodes = nodes.map((n) => (n.id === tgt.id ? { ...n, data: { ...n.data, params: { ...n.data.params, shape: JSON.stringify(shape) } } } : n));
  }
  return { ...doc, nodes, edges };
}

function createsCycle(edges: Edge[], from: string, to: string): boolean {
  // A loop forms if `from` is already reachable from `to`
  const next = new Map<string, string[]>();
  for (const e of edges) next.set(e.source, [...(next.get(e.source) ?? []), e.target]);
  const stack = [to];
  const seen = new Set<string>();
  while (stack.length) {
    const id = stack.pop()!;
    if (id === from) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    stack.push(...(next.get(id) ?? []));
  }
  return false;
}

/** Removes the connections from one layer to another (or just the one into `toHandle`) */
export function disconnect(doc: GraphDoc, { from, to, toHandle }: { from: string; to: string; toHandle?: string }): GraphDoc {
  const edges = doc.edges.filter((e) => !(e.source === from && e.target === to && (toHandle == null || (e.targetHandle ?? "in") === toHandle)));
  if (edges.length === doc.edges.length) throw new GraphOpError(`${from} isn't connected to ${to}${toHandle ? `'s input "${toHandle}"` : ""}.`);
  return { ...doc, edges };
}

// ---------------------------------------------------------------------------
// Placing new layers
// ---------------------------------------------------------------------------

/** Vertical room one inserted layer takes */
export const ROW = 120;

/**
 * Inserts a new layer right after an existing one: it takes over that layer's outgoing connections
 * from the same output, and the layers below move down to make room.
 */
export function insertAfter(doc: GraphDoc, afterId: string, layer: Omit<NewLayer, "position">, fromHandle?: string): { doc: GraphDoc; id: string } {
  const after = requireNode(doc, afterId);
  const handle = fromHandle ?? handlesOf(after).outputHandles[0]?.id;
  if (!handle) throw new GraphOpError(`${afterId} has no outputs to insert after.`);
  const position = { x: after.position.x, y: after.position.y + ROW };
  // Everything at or below where the new layer goes moves down a row
  const shifted: GraphDoc = {
    ...doc,
    nodes: doc.nodes.map((n) => (n.position.y >= position.y - 1 ? { ...n, position: { x: n.position.x, y: n.position.y + ROW } } : n)),
  };
  const added = addLayer(shifted, { ...layer, position });
  const node = requireNode(added.doc, added.id);
  const firstIn = handlesOf(node).inputHandles[0]?.id;
  const firstOut = handlesOf(node).outputHandles[0]?.id;
  if (!firstIn) throw new GraphOpError(`A ${layer.moduleType} has no input, so it can't go after another layer.`);
  const moving = added.doc.edges.filter((e) => e.source === afterId && (e.sourceHandle ?? "out") === handle);
  let edges = added.doc.edges.filter((e) => !moving.includes(e));
  edges = [...edges, makeEdge(afterId, handle, added.id, firstIn)];
  if (firstOut) for (const e of moving) edges.push(makeEdge(added.id, firstOut, e.target, e.targetHandle ?? "in"));
  return { doc: { ...added.doc, edges }, id: added.id };
}

/** A free spot to the right of everything, for layers that aren't inserted after another */
export function freePosition(doc: GraphDoc, index = 0): { x: number; y: number } {
  if (!doc.nodes.length) return { x: 300, y: 60 + index * ROW };
  const maxX = Math.max(...doc.nodes.map((n) => n.position.x));
  const minY = Math.min(...doc.nodes.map((n) => n.position.y));
  return { x: maxX + 300, y: minY + index * ROW };
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

export function groupLayers(doc: GraphDoc, ids: string[], name?: string, groupId = `group_${Date.now().toString(36)}`): { doc: GraphDoc; id: string } {
  if (ids.length < 2) throw new GraphOpError("A group needs at least two layers.");
  for (const id of ids) requireNode(doc, id);
  return { doc: { ...doc, groups: addGroup(doc.groups, ids, name?.trim() || nextGroupName(doc.groups), groupId) }, id: groupId };
}
