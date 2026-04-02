import { create } from "zustand";
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from "@xyflow/react";
import { MODULE_REGISTRY } from "../lib/registry";
import type { NodeCategory, CodeBlockDef, HandleDef, DataOutputDef } from "../lib/types";
import { safePythonId, DEFAULT_DATA_OUTPUTS } from "../lib/types";
import type { ShapeMap, ShapeError } from "../lib/shape-inference";
import { createCompositeBlock, expandCompositeBlock, type CompositeBlock } from "../lib/composite-nodes";
import { isDatasetNode, DATASET_METADATA, LEGACY_DATASET_TYPES, resolveDataNodeShape, DATA_PRESETS } from "../lib/dataset-metadata";
import { formatCtorArgs } from "../lib/decompiler";

export interface NodeData extends Record<string, unknown> {
  moduleType: string;
  label: string;
  category: NodeCategory;
  params: Record<string, unknown>;
  customCode?: string;
  codeBlock?: CodeBlockDef;
  dataOutputs?: DataOutputDef[];  // Data node: explicit output handles with column mapping
}

type TLNode = Node<NodeData>;
type TLEdge = Edge;

interface Clipboard {
  nodes: TLNode[];
  edges: TLEdge[];
}

interface GraphState {
  nodes: TLNode[];
  edges: TLEdge[];
  selectedNodeId: string | null;
  projectName: string;
  shapeMap: ShapeMap;
  shapeErrors: Record<string, ShapeError>;
  shapeDisplayBatch: boolean;
  clipboard: Clipboard | null;
  compositeBlocks: Record<string, CompositeBlock>;
  onNodesChange: OnNodesChange<TLNode>;
  onEdgesChange: OnEdgesChange<TLEdge>;
  onConnect: OnConnect;
  addNode: (moduleType: string, position: { x: number; y: number }) => void;
  updateNodeParam: (nodeId: string, paramName: string, value: unknown) => void;
  setSelectedNodeId: (id: string | null) => void;
  deleteSelected: () => void;
  deleteMultiple: (nodeIds: string[]) => void;
  duplicateSelected: () => void;
  renameNode: (nodeId: string, label: string) => void;
  copySelected: () => void;
  paste: () => void;
  selectAll: () => void;
  setShapeMap: (map: ShapeMap) => void;
  setShapeErrors: (errors: Record<string, ShapeError>) => void;
  toggleShapeDisplayBatch: () => void;
  addNodeConnected: (moduleType: string, sourceNodeId: string, sourceHandleId: string) => void;
  groupSelected: (name: string) => void;
  ungroupNode: (nodeId: string) => void;
  setCustomCode: (nodeId: string, code: string | undefined) => void;
  convertToCodeBlock: (nodeId: string) => void;
  revertToStandard: (nodeId: string) => void;
  updateCodeBlock: (nodeId: string, patch: Partial<CodeBlockDef>) => void;
  addCodeBlockHandle: (nodeId: string, side: "input" | "output", handle: HandleDef) => void;
  removeCodeBlockHandle: (nodeId: string, side: "input" | "output", handleId: string) => void;
  setDataOutputs: (nodeId: string, outputs: DataOutputDef[]) => void;
  addDataOutput: (nodeId: string, output: DataOutputDef) => void;
  removeDataOutput: (nodeId: string, handleId: string) => void;
  updateDataOutput: (nodeId: string, handleId: string, patch: Partial<DataOutputDef>) => void;
  clearGraph: () => void;
  loadGraph: (nodes: TLNode[], edges: TLEdge[]) => void;
  setProjectName: (name: string) => void;
}

let nodeIdCounter = 0;

function nextNodeId(): string {
  return `node_${++nodeIdCounter}`;
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  projectName: "Untitled",
  shapeMap: {},
  shapeErrors: {},
  shapeDisplayBatch: false,
  clipboard: null,
  compositeBlocks: {},

  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },

  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  onConnect: (connection) => {
    const newEdges = addEdge({ ...connection, type: "smoothstep" }, get().edges);

    // Auto-set Input node shape when a Dataset node connects to it
    const nodes = get().nodes;
    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);
    if (sourceNode && targetNode && isDatasetNode(sourceNode.data.moduleType) && targetNode.data.moduleType === "Input") {
      // Universal Data node: resolve shape from params
      let inputShape: number[] | null = null;
      if (sourceNode.data.moduleType === "Data") {
        const resolved = resolveDataNodeShape(sourceNode.data.params);
        if (resolved) inputShape = resolved.inputShape;
      } else {
        // Legacy node: use DATASET_METADATA
        const meta = DATASET_METADATA[sourceNode.data.moduleType];
        if (meta) inputShape = meta.inputShape;
      }
      if (inputShape) {
        set({
          edges: newEdges,
          nodes: nodes.map((n) =>
            n.id === targetNode.id
              ? { ...n, data: { ...n.data, params: { ...n.data.params, shape: JSON.stringify(inputShape) } } }
              : n,
          ),
        });
        return;
      }
    }
    set({ edges: newEdges });
  },

  addNode: (moduleType, position) => {
    const def = MODULE_REGISTRY[moduleType];
    if (!def) return;

    const defaults: Record<string, unknown> = {};
    for (const p of def.params) {
      defaults[p.name] = p.default;
    }

    const newNode: TLNode = {
      id: nextNodeId(),
      type: "tensorLabNode",
      position,
      data: {
        moduleType: def.type,
        label: def.label,
        category: def.category,
        params: defaults,
        // Data nodes get explicit output handles
        ...(moduleType === "Data" ? { dataOutputs: [...DEFAULT_DATA_OUTPUTS] } : {}),
      },
    };

    set({ nodes: [...get().nodes, newNode] });
  },

  updateNodeParam: (nodeId, paramName, value) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [paramName]: value } } }
          : n,
      ),
    });
  },

  setCustomCode: (nodeId, code) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, customCode: code } }
          : n,
      ),
    });
  },

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  deleteSelected: () => {
    const { nodes, edges, selectedNodeId } = get();
    // Delete all selected nodes (React Flow multi-select)
    const selectedIds = new Set<string>();
    if (selectedNodeId) selectedIds.add(selectedNodeId);
    for (const n of nodes) {
      if (n.selected) selectedIds.add(n.id);
    }
    if (selectedIds.size === 0) return;
    set({
      nodes: nodes.filter((n) => !selectedIds.has(n.id)),
      edges: edges.filter((e) => !selectedIds.has(e.source) && !selectedIds.has(e.target)),
      selectedNodeId: null,
    });
  },

  deleteMultiple: (nodeIds) => {
    const idSet = new Set(nodeIds);
    const { nodes, edges } = get();
    set({
      nodes: nodes.filter((n) => !idSet.has(n.id)),
      edges: edges.filter((e) => !idSet.has(e.source) && !idSet.has(e.target)),
      selectedNodeId: null,
    });
  },

  duplicateSelected: () => {
    const { nodes, edges } = get();
    // Duplicate all selected nodes
    const selected = nodes.filter((n) => n.selected || n.id === get().selectedNodeId);
    if (selected.length === 0) return;

    const idMap = new Map<string, string>();
    const newNodes: TLNode[] = [];

    for (const orig of selected) {
      const newId = nextNodeId();
      idMap.set(orig.id, newId);
      newNodes.push({
        ...orig,
        id: newId,
        position: { x: orig.position.x + 40, y: orig.position.y + 40 },
        data: { ...orig.data, params: { ...orig.data.params } },
        selected: true,
      });
    }

    // Duplicate edges between selected nodes
    const selectedIds = new Set(selected.map((n) => n.id));
    const newEdges: TLEdge[] = [];
    for (const e of edges) {
      if (selectedIds.has(e.source) && selectedIds.has(e.target)) {
        newEdges.push({
          ...e,
          id: `e_${idMap.get(e.source)}_${idMap.get(e.target)}`,
          source: idMap.get(e.source)!,
          target: idMap.get(e.target)!,
        });
      }
    }

    // Deselect old, select new
    const deselected = nodes.map((n) => ({ ...n, selected: false }));
    set({
      nodes: [...deselected, ...newNodes],
      edges: [...edges, ...newEdges],
      selectedNodeId: newNodes.length === 1 ? newNodes[0].id : null,
    });
  },

  renameNode: (nodeId, label) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, label } } : n,
      ),
    });
  },

  copySelected: () => {
    const { nodes, edges, selectedNodeId } = get();
    const selected = nodes.filter((n) => n.selected || n.id === selectedNodeId);
    if (selected.length === 0) return;

    const selectedIds = new Set(selected.map((n) => n.id));
    const relevantEdges = edges.filter(
      (e) => selectedIds.has(e.source) && selectedIds.has(e.target),
    );

    set({ clipboard: { nodes: structuredClone(selected), edges: structuredClone(relevantEdges) } });
  },

  paste: () => {
    const { clipboard, nodes, edges } = get();
    if (!clipboard || clipboard.nodes.length === 0) return;

    const idMap = new Map<string, string>();
    const newNodes: TLNode[] = [];

    for (const orig of clipboard.nodes) {
      const newId = nextNodeId();
      idMap.set(orig.id, newId);
      newNodes.push({
        ...orig,
        id: newId,
        position: { x: orig.position.x + 50, y: orig.position.y + 50 },
        data: { ...orig.data, params: { ...orig.data.params } },
        selected: true,
      });
    }

    const newEdges: TLEdge[] = clipboard.edges.map((e) => ({
      ...e,
      id: `e_${idMap.get(e.source)}_${idMap.get(e.target)}`,
      source: idMap.get(e.source)!,
      target: idMap.get(e.target)!,
    }));

    const deselected = nodes.map((n) => ({ ...n, selected: false }));
    set({
      nodes: [...deselected, ...newNodes],
      edges: [...edges, ...newEdges],
    });
  },

  selectAll: () => {
    set({ nodes: get().nodes.map((n) => ({ ...n, selected: true })) });
  },

  setShapeMap: (shapeMap) => set({ shapeMap }),
  setShapeErrors: (shapeErrors) => set({ shapeErrors }),
  toggleShapeDisplayBatch: () => set((s) => ({ shapeDisplayBatch: !s.shapeDisplayBatch })),

  addNodeConnected: (moduleType, sourceNodeId, sourceHandleId) => {
    const def = MODULE_REGISTRY[moduleType];
    if (!def) return;

    const sourceNode = get().nodes.find((n) => n.id === sourceNodeId);
    if (!sourceNode) return;

    const defaults: Record<string, unknown> = {};
    for (const p of def.params) {
      defaults[p.name] = p.default;
    }

    const newId = nextNodeId();
    const newNode: TLNode = {
      id: newId,
      type: "tensorLabNode",
      position: {
        x: sourceNode.position.x,
        y: sourceNode.position.y + 120,
      },
      data: {
        moduleType: def.type,
        label: def.label,
        category: def.category,
        params: defaults,
      },
    };

    const targetHandle = def.inputHandles?.[0]?.id ?? "in";
    const newEdge: TLEdge = {
      id: `e_${sourceNodeId}_${newId}`,
      source: sourceNodeId,
      sourceHandle: sourceHandleId,
      target: newId,
      targetHandle,
      type: "smoothstep",
    };

    set({
      nodes: [...get().nodes, newNode],
      edges: [...get().edges, newEdge],
      selectedNodeId: newId,
    });
  },

  groupSelected: (name) => {
    const { nodes, edges, compositeBlocks } = get();
    const selectedIds = new Set(nodes.filter((n) => n.selected).map((n) => n.id));
    const { selectedNodeId } = get();
    if (selectedNodeId) selectedIds.add(selectedNodeId);
    if (selectedIds.size < 2) return;

    const { nodes: newNodes, edges: newEdges, block } = createCompositeBlock(nodes, edges, selectedIds, name);
    set({
      nodes: newNodes,
      edges: newEdges,
      compositeBlocks: { ...compositeBlocks, [block.id]: block },
      selectedNodeId: block.id,
    });
  },

  ungroupNode: (nodeId) => {
    const { nodes, edges, compositeBlocks } = get();
    const block = compositeBlocks[nodeId];
    if (!block) return;

    const { nodes: newNodes, edges: newEdges } = expandCompositeBlock(nodes, edges, nodeId, block);
    const { [nodeId]: _, ...rest } = compositeBlocks;
    set({ nodes: newNodes, edges: newEdges, compositeBlocks: rest, selectedNodeId: null });
  },

  convertToCodeBlock: (nodeId) => {
    const { nodes, edges } = get();
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const mt = node.data.moduleType;
    // Blocked node types
    if (["Input", "Output", "Data"].includes(mt) || isDatasetNode(mt)) return;
    if (node.data.codeBlock) return; // already converted

    const def = MODULE_REGISTRY[mt];
    const origInputHandles = def?.inputHandles ?? [{ id: "in", label: "in", dataType: "tensor" as const }];
    const origOutputHandles = def?.outputHandles ?? [{ id: "out", label: "out", dataType: "tensor" as const }];

    // Map handle IDs to Python-safe names
    const inputHandles: HandleDef[] = origInputHandles.map((h) => ({
      ...h,
      id: safePythonId(h.id),
      label: safePythonId(h.id),
    }));
    const outputHandles: HandleDef[] = origOutputHandles.map((h) => ({
      ...h,
      id: safePythonId(h.id),
      label: safePythonId(h.id),
    }));

    // Generate default initBody
    const FUNCTIONAL_OPS = new Set(["Add", "Concatenate", "MatMul"]);
    let initBody: string;
    if (FUNCTIONAL_OPS.has(mt)) {
      initBody = "pass";
    } else if (node.data.customCode) {
      initBody = `self.layer = ${node.data.customCode}`;
    } else {
      const args = formatCtorArgs(mt, node.data.params);
      initBody = `self.layer = nn.${mt}(${args})`;
    }

    // Generate default forwardBody
    let forwardBody: string;
    const inputNames = inputHandles.map((h) => h.id);
    if (mt === "Add") {
      const vars = inputNames.length >= 2 ? inputNames : ["x", "y"];
      forwardBody = `return ${vars[0]} + ${vars[1]}`;
    } else if (mt === "Concatenate") {
      const dim = node.data.params.dim ?? 1;
      forwardBody = `return torch.cat([${inputNames.join(", ")}], dim=${dim})`;
    } else if (mt === "MatMul") {
      const vars = inputNames.length >= 2 ? inputNames : ["x", "y"];
      forwardBody = `return torch.matmul(${vars[0]}, ${vars[1]})`;
    } else {
      forwardBody = `return self.layer(${inputNames.join(", ")})`;
    }

    const codeBlock: CodeBlockDef = {
      inputHandles,
      outputHandles,
      initBody,
      forwardBody,
      originalModuleType: mt,
      originalParams: { ...node.data.params },
      originalCustomCode: node.data.customCode,
    };

    // Remap edges from old handle IDs to new Python-safe IDs
    const oldToNewInput = new Map<string, string>();
    origInputHandles.forEach((h, i) => {
      if (inputHandles[i]) oldToNewInput.set(h.id, inputHandles[i].id);
    });
    const oldToNewOutput = new Map<string, string>();
    origOutputHandles.forEach((h, i) => {
      if (outputHandles[i]) oldToNewOutput.set(h.id, outputHandles[i].id);
    });

    const newEdges = edges.map((e) => {
      if (e.target === nodeId && oldToNewInput.has(e.targetHandle ?? "in")) {
        return { ...e, targetHandle: oldToNewInput.get(e.targetHandle ?? "in")! };
      }
      if (e.source === nodeId && oldToNewOutput.has(e.sourceHandle ?? "out")) {
        return { ...e, sourceHandle: oldToNewOutput.get(e.sourceHandle ?? "out")! };
      }
      return e;
    });

    set({
      nodes: nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, codeBlock } } : n,
      ),
      edges: newEdges,
    });
  },

  revertToStandard: (nodeId) => {
    const { nodes, edges } = get();
    const node = nodes.find((n) => n.id === nodeId);
    if (!node || !node.data.codeBlock) return;

    const cb = node.data.codeBlock;
    const def = MODULE_REGISTRY[cb.originalModuleType];
    const stdInputHandles = def?.inputHandles ?? [{ id: "in", label: "in", dataType: "tensor" as const }];
    const stdOutputHandles = def?.outputHandles ?? [{ id: "out", label: "out", dataType: "tensor" as const }];

    // Build position-based remap from current code block handles to standard handles
    const inputRemap = new Map<string, string>();
    cb.inputHandles.forEach((h, i) => {
      if (stdInputHandles[i]) inputRemap.set(h.id, stdInputHandles[i].id);
    });
    const outputRemap = new Map<string, string>();
    cb.outputHandles.forEach((h, i) => {
      if (stdOutputHandles[i]) outputRemap.set(h.id, stdOutputHandles[i].id);
    });

    // Remap edges back, remove edges connected to handles that don't exist in standard
    const newEdges = edges
      .map((e) => {
        if (e.target === nodeId) {
          const remapped = inputRemap.get(e.targetHandle ?? "in");
          if (!remapped) return null; // handle doesn't exist in standard
          return { ...e, targetHandle: remapped };
        }
        if (e.source === nodeId) {
          const remapped = outputRemap.get(e.sourceHandle ?? "out");
          if (!remapped) return null;
          return { ...e, sourceHandle: remapped };
        }
        return e;
      })
      .filter((e): e is TLEdge => e !== null);

    const { codeBlock: _, ...restData } = node.data;

    set({
      nodes: nodes.map((n) =>
        n.id === nodeId
          ? {
              ...n,
              data: {
                ...restData,
                params: { ...cb.originalParams },
                customCode: cb.originalCustomCode,
              },
            }
          : n,
      ),
      edges: newEdges,
    });
  },

  updateCodeBlock: (nodeId, patch) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== nodeId || !n.data.codeBlock) return n;
        return {
          ...n,
          data: {
            ...n.data,
            codeBlock: { ...n.data.codeBlock, ...patch },
          },
        };
      }),
    });
  },

  addCodeBlockHandle: (nodeId, side, handle) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== nodeId || !n.data.codeBlock) return n;
        const cb = n.data.codeBlock;
        const key = side === "input" ? "inputHandles" : "outputHandles";
        return {
          ...n,
          data: {
            ...n.data,
            codeBlock: { ...cb, [key]: [...cb[key], handle] },
          },
        };
      }),
    });
  },

  removeCodeBlockHandle: (nodeId, side, handleId) => {
    const { nodes, edges } = get();
    const node = nodes.find((n) => n.id === nodeId);
    if (!node || !node.data.codeBlock) return;

    const cb = node.data.codeBlock;
    const key = side === "input" ? "inputHandles" : "outputHandles";

    // Remove the handle
    const newHandles = cb[key].filter((h) => h.id !== handleId);

    // Disconnect edges connected to the removed handle
    const newEdges = edges.filter((e) => {
      if (side === "input" && e.target === nodeId && e.targetHandle === handleId) return false;
      if (side === "output" && e.source === nodeId && e.sourceHandle === handleId) return false;
      return true;
    });

    set({
      nodes: nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, codeBlock: { ...cb, [key]: newHandles } } }
          : n,
      ),
      edges: newEdges,
    });
  },

  setDataOutputs: (nodeId, outputs) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, dataOutputs: outputs } } : n,
      ),
    });
  },

  addDataOutput: (nodeId, output) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== nodeId) return n;
        const current = n.data.dataOutputs ?? DEFAULT_DATA_OUTPUTS;
        return { ...n, data: { ...n.data, dataOutputs: [...current, output] } };
      }),
    });
  },

  removeDataOutput: (nodeId, handleId) => {
    const { nodes, edges } = get();
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const current = node.data.dataOutputs ?? DEFAULT_DATA_OUTPUTS;
    const newOutputs = current.filter((o) => o.id !== handleId);
    // Disconnect edges from the removed handle
    const newEdges = edges.filter((e) => !(e.source === nodeId && e.sourceHandle === handleId));
    set({
      nodes: nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, dataOutputs: newOutputs } } : n,
      ),
      edges: newEdges,
    });
  },

  updateDataOutput: (nodeId, handleId, patch) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== nodeId) return n;
        const current = n.data.dataOutputs ?? DEFAULT_DATA_OUTPUTS;
        return {
          ...n,
          data: {
            ...n.data,
            dataOutputs: current.map((o) => (o.id === handleId ? { ...o, ...patch } : o)),
          },
        };
      }),
    });
  },

  clearGraph: () => {
    nodeIdCounter = 0;
    set({ nodes: [], edges: [], selectedNodeId: null, shapeMap: {}, shapeErrors: {} });
  },

  loadGraph: (nodes, edges) => {
    // Migrate legacy dataset nodes to universal Data node
    const migrated = nodes.map((n) => {
      if (LEGACY_DATASET_TYPES.has(n.data.moduleType)) {
        const oldType = n.data.moduleType;
        const oldParams = n.data.params;
        const preset = DATA_PRESETS[oldType];
        const domain = preset?.domain ?? "auto";

        // Build new params: source + domain + merged old params
        const newParams: Record<string, unknown> = {
          source: oldType === "CustomCSV" ? "CSV" : oldType,
          domain,
          val_split: oldParams.val_split ?? 0.2,
        };

        // Map old "root" to appropriate field
        if (oldType === "HuggingFace") {
          newParams.dataset_name = oldParams.root ?? "mnist";
        } else if (["ImageFolder", "CustomCSV", "TensorFile"].includes(oldType)) {
          newParams.path = oldParams.root ?? "./data";
        } else {
          newParams.path = oldParams.root ?? preset?.defaults.path ?? "./data";
        }

        // Carry over all other params
        for (const [k, v] of Object.entries(oldParams)) {
          if (k !== "root" && k !== "val_split" && v !== undefined && v !== null) {
            newParams[k] = v;
          }
        }

        return {
          ...n,
          data: {
            ...n.data,
            moduleType: "Data",
            label: preset?.label ?? n.data.label,
            category: "data" as NodeCategory,
            params: newParams,
          },
        };
      }
      return n;
    });

    const maxId = migrated.reduce((max, n) => {
      const match = n.id.match(/^node_(\d+)$/);
      return match ? Math.max(max, parseInt(match[1], 10)) : max;
    }, 0);
    nodeIdCounter = Math.max(nodeIdCounter, maxId);
    set({ nodes: migrated, edges, selectedNodeId: null });
  },

  setProjectName: (name) => set({ projectName: name }),
}));
