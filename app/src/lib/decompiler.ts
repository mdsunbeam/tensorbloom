/**
 * Client-side decompiler: converts the React Flow graph into PyTorch code.
 * Runs entirely in the browser — no Python sidecar needed.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { ShapeMap } from "./shape-inference";
import { isDatasetNode } from "./dataset-metadata";

interface DecompileResult {
  code: string;
  errors: string[];
}

/** Functional ops that don't need __init__ */
const FUNCTIONAL_OPS = new Set(["Add", "Concatenate", "MatMul"]);

/** Edge with handle info for code block awareness */
interface IncomingEdge {
  sourceNodeId: string;
  sourceHandle: string;
  targetHandle: string;
}

/**
 * Compute variable names for all nodes in the graph.
 * Returns nodeId -> varName (e.g., "node_3" -> "conv1").
 * Uses topological sort + label sanitization + uniqueness logic.
 */
export function computeVariableNames(
  nodes: Node<NodeData>[],
  edges: Edge[],
): Map<string, string> {
  const varNames = new Map<string, string>();

  if (nodes.length === 0) return varNames;

  const sorted = topologicalSort(nodes, edges);
  if (!sorted) return varNames;

  // Build incoming map
  const incomingMap = new Map<string, string[]>();
  for (const e of edges) {
    const list = incomingMap.get(e.target) ?? [];
    list.push(e.source);
    incomingMap.set(e.target, list);
  }

  // Identify input nodes
  const isEntryNode = (n: Node<NodeData>) => n.data.moduleType === "Input";
  const inputNodes = nodes.filter(isEntryNode).length > 0
    ? nodes.filter(isEntryNode)
    : nodes.filter((n) => !incomingMap.has(n.id) && n.data.moduleType !== "Output" && !isDatasetNode(n.data.moduleType));

  // Assign input var names
  if (inputNodes.length === 1) {
    varNames.set(inputNodes[0].id, "x");
  } else {
    for (const n of inputNodes) {
      varNames.set(n.id, n.data.label || `x${inputNodes.indexOf(n) + 1}`);
    }
  }

  // Build unique names for module nodes
  const usedNames = new Set<string>();
  for (const n of sorted) {
    if (n.data.moduleType === "Input" || n.data.moduleType === "Output" || isDatasetNode(n.data.moduleType)) continue;
    let base = sanitizeId(n.data.label || n.id);
    if (usedNames.has(base)) {
      let i = 2;
      while (usedNames.has(`${base}_${i}`)) i++;
      base = `${base}_${i}`;
    }
    usedNames.add(base);
    varNames.set(n.id, base);
  }

  return varNames;
}

/**
 * Generate a per-node code snippet showing __init__ and forward() lines with shape annotations.
 */
export function generateNodeSnippet(
  node: Node<NodeData>,
  nodes: Node<NodeData>[],
  edges: Edge[],
  shapeMap: ShapeMap,
): { initCode: string; forwardCode: string } {
  const mt = node.data.moduleType;
  const varNames = computeVariableNames(nodes, edges);
  const myVar = varNames.get(node.id) ?? sanitizeId(node.data.label || node.id);

  // Code block nodes: show class definition + forward
  if (node.data.codeBlock) {
    const cb = node.data.codeBlock;
    const sig = `def forward(self, ${cb.inputHandles.map((h) => h.id).join(", ")}):`;
    const initCode = `# __init__\n${cb.initBody}`;
    const forwardCode = `# ${sig}\n${cb.forwardBody}`;
    return { initCode, forwardCode };
  }

  // Find incoming sources for this node
  const sources: string[] = [];
  for (const e of edges) {
    if (e.target === node.id) sources.push(e.source);
  }
  const inputVars = sources.map((s) => varNames.get(s) ?? "x");

  // Shape info
  const outputShape = shapeMap[node.id]?.shape;
  const inputShapes = sources.map((s) => shapeMap[s]?.shape).filter(Boolean) as number[][];

  // Generate init line
  let initCode = "";
  if (!FUNCTIONAL_OPS.has(mt)) {
    if (node.data.customCode) {
      initCode = `self.${myVar} = ${node.data.customCode}`;
    } else {
      const args = formatCtorArgs(mt, node.data.params);
      initCode = `self.${myVar} = nn.${mt}(${args})`;
    }
  }

  // Generate forward expression
  let forwardExpr = "";
  if (mt === "Add") {
    const vars = inputVars.length >= 2 ? inputVars : ["x", "y"];
    forwardExpr = `${myVar} = ${vars[0]} + ${vars[1]}`;
  } else if (mt === "Concatenate") {
    const dim = node.data.params.dim ?? 1;
    const vars = inputVars.length > 0 ? inputVars : ["x"];
    forwardExpr = `${myVar} = torch.cat([${vars.join(", ")}], dim=${dim})`;
  } else if (mt === "MatMul") {
    const vars = inputVars.length >= 2 ? inputVars : ["x", "y"];
    forwardExpr = `${myVar} = torch.matmul(${vars[0]}, ${vars[1]})`;
  } else {
    const inputStr = inputVars.length > 0 ? inputVars.join(", ") : "x";
    forwardExpr = `${myVar} = self.${myVar}(${inputStr})`;
  }

  // Build shape annotation
  let shapeAnnotation = "";
  if (outputShape) {
    const outStr = `[${outputShape.join(", ")}]`;
    if (inputShapes.length === 1) {
      const inStr = `[${inputShapes[0].join(", ")}]`;
      if (JSON.stringify(inputShapes[0]) === JSON.stringify(outputShape)) {
        shapeAnnotation = outStr;
      } else {
        shapeAnnotation = `${inStr} \u2192 ${outStr}`;
      }
    } else if (inputShapes.length > 1) {
      if (mt === "Concatenate") {
        const inParts = inputShapes.map((s) => `[${s.join(", ")}]`);
        shapeAnnotation = `${inParts.join(" cat ")} \u2192 ${outStr}`;
      } else {
        // For Add/MatMul with multiple inputs
        if (JSON.stringify(inputShapes[0]) === JSON.stringify(outputShape)) {
          shapeAnnotation = outStr;
        } else {
          shapeAnnotation = `[${inputShapes[0].join(", ")}] \u2192 ${outStr}`;
        }
      }
    } else {
      // No input shapes available, just show output
      shapeAnnotation = outStr;
    }
  }

  // Compose forward code line with shape comment
  const shapeSuffix = shapeAnnotation ? `              ${shapeAnnotation}` : "";
  const forwardCode = `# forward${shapeSuffix}\n${forwardExpr}`;

  return { initCode, forwardCode };
}

export function decompileGraph(
  nodes: Node<NodeData>[],
  edges: Edge[],
): DecompileResult {
  const errors: string[] = [];

  if (nodes.length === 0) {
    return { code: "", errors: [] };
  }

  // Build handle-aware adjacency: targetId -> IncomingEdge[]
  const incomingMap = new Map<string, IncomingEdge[]>();
  const incomingSimple = new Map<string, string[]>(); // for legacy compatibility
  for (const e of edges) {
    const incoming = incomingMap.get(e.target) ?? [];
    incoming.push({
      sourceNodeId: e.source,
      sourceHandle: e.sourceHandle ?? "out",
      targetHandle: e.targetHandle ?? "in",
    });
    incomingMap.set(e.target, incoming);

    const simple = incomingSimple.get(e.target) ?? [];
    simple.push(e.source);
    incomingSimple.set(e.target, simple);
  }

  // Classify nodes
  const isEntryNode = (n: Node<NodeData>) => n.data.moduleType === "Input";
  const inputNodes = nodes.filter(isEntryNode).length > 0
    ? nodes.filter(isEntryNode)
    : nodes.filter((n) => !incomingSimple.has(n.id) && n.data.moduleType !== "Output" && !isDatasetNode(n.data.moduleType));

  const outputNodes = nodes.filter((n) => n.data.moduleType === "Output");
  const sorted = topologicalSort(nodes, edges);
  if (!sorted) {
    errors.push("Graph contains a cycle");
    return { code: "# Error: graph contains a cycle\n", errors };
  }

  const varNames = computeVariableNames(nodes, edges);

  // Track per-handle output variable names for multi-output code blocks
  // nodeId -> handleId -> varName
  const outputVarMap = new Map<string, Map<string, string>>();

  // Collect code block helper class definitions
  const helperClasses: string[] = [];
  const codeBlockNodes = sorted.filter((n) => n.data.codeBlock);

  for (const n of codeBlockNodes) {
    const cb = n.data.codeBlock!;
    const className = `_${sanitizeId(varNames.get(n.id) ?? n.id)}_Block`;
    const initBodyLines = cb.initBody.split("\n").map((l) => `        ${l}`).join("\n");
    const forwardBodyLines = cb.forwardBody.split("\n").map((l) => `        ${l}`).join("\n");
    const sig = cb.inputHandles.map((h) => h.id).join(", ");
    helperClasses.push(
      `class ${className}(nn.Module):`,
      `    def __init__(self):`,
      `        super().__init__()`,
      initBodyLines,
      ``,
      `    def forward(self, ${sig}):`,
      forwardBodyLines,
      ``,
    );
  }

  // Build __init__ lines
  const initLines: string[] = [];
  for (const n of sorted) {
    if (n.data.moduleType === "Input" || n.data.moduleType === "Output" || isDatasetNode(n.data.moduleType)) continue;

    const name = varNames.get(n.id) ?? sanitizeId(n.id);

    if (n.data.codeBlock) {
      const className = `_${sanitizeId(name)}_Block`;
      initLines.push(`        self.${name} = ${className}()`);
      continue;
    }

    if (FUNCTIONAL_OPS.has(n.data.moduleType)) continue;

    if (n.data.customCode) {
      initLines.push(`        self.${name} = ${n.data.customCode}`);
    } else {
      const args = formatCtorArgs(n.data.moduleType, n.data.params);
      initLines.push(`        self.${name} = nn.${n.data.moduleType}(${args})`);
    }
  }

  // Resolve the upstream variable for a given incoming edge
  const resolveInputVar = (edge: IncomingEdge): string => {
    // Check multi-output map first
    const handleMap = outputVarMap.get(edge.sourceNodeId);
    if (handleMap) {
      const v = handleMap.get(edge.sourceHandle);
      if (v) return v;
    }
    return varNames.get(edge.sourceNodeId) ?? "x";
  };

  // Build forward lines
  const forwardLines: string[] = [];
  for (const n of sorted) {
    if (n.data.moduleType === "Input" || isDatasetNode(n.data.moduleType)) continue;
    if (n.data.moduleType === "Output") continue;

    const myVar = varNames.get(n.id) ?? sanitizeId(n.id);
    const incomingEdges = incomingMap.get(n.id) ?? [];

    if (n.data.codeBlock) {
      const cb = n.data.codeBlock;
      // Map each input handle to the upstream variable connected to it
      const argVars: string[] = [];
      for (const h of cb.inputHandles) {
        const edge = incomingEdges.find((e) => e.targetHandle === h.id);
        argVars.push(edge ? resolveInputVar(edge) : h.id);
      }

      if (cb.outputHandles.length > 1) {
        // Multi-output: unpack
        const outVars = cb.outputHandles.map((h) => `${myVar}_${h.id}`);
        forwardLines.push(`        ${outVars.join(", ")} = self.${myVar}(${argVars.join(", ")})`);
        const handleMap = new Map<string, string>();
        cb.outputHandles.forEach((h, i) => handleMap.set(h.id, outVars[i]));
        outputVarMap.set(n.id, handleMap);
      } else {
        forwardLines.push(`        ${myVar} = self.${myVar}(${argVars.join(", ")})`);
      }
      continue;
    }

    // Standard node handling
    const inputVars = incomingEdges.map(resolveInputVar);

    const mt = n.data.moduleType;
    if (mt === "Add") {
      if (inputVars.length >= 2) {
        forwardLines.push(`        ${myVar} = ${inputVars[0]} + ${inputVars[1]}`);
      }
    } else if (mt === "Concatenate") {
      const dim = n.data.params.dim ?? 1;
      forwardLines.push(`        ${myVar} = torch.cat([${inputVars.join(", ")}], dim=${dim})`);
    } else if (mt === "MatMul") {
      if (inputVars.length >= 2) {
        forwardLines.push(`        ${myVar} = torch.matmul(${inputVars[0]}, ${inputVars[1]})`);
      }
    } else {
      const inputStr = inputVars.length > 0 ? inputVars.join(", ") : "x";
      forwardLines.push(`        ${myVar} = self.${myVar}(${inputStr})`);
    }
  }

  // Return value
  let returnVar = "x";
  for (const n of outputNodes) {
    const incomingEdges = incomingMap.get(n.id) ?? [];
    if (incomingEdges.length > 0) {
      returnVar = resolveInputVar(incomingEdges[0]);
    }
  }
  if (outputNodes.length === 0 && forwardLines.length > 0) {
    const lastLine = forwardLines[forwardLines.length - 1];
    const match = lastLine.match(/^\s+(\w+)\s*=/);
    if (match) returnVar = match[1];
  }

  // Assemble
  const lines = [
    "import torch",
    "import torch.nn as nn",
    "",
  ];

  // Helper classes before Model
  if (helperClasses.length > 0) {
    lines.push("");
    lines.push(...helperClasses);
  }

  lines.push("");
  lines.push("class Model(nn.Module):");
  lines.push("    def __init__(self):");
  lines.push("        super().__init__()");

  if (initLines.length > 0) {
    lines.push(...initLines);
  } else {
    lines.push("        pass");
  }

  lines.push("");
  const forwardParams = inputNodes.length <= 1
    ? "x"
    : inputNodes.map((n) => varNames.get(n.id) ?? "x").join(", ");
  lines.push(`    def forward(self, ${forwardParams}):`);

  if (forwardLines.length > 0) {
    lines.push(...forwardLines);
    lines.push(`        return ${returnVar}`);
  } else {
    lines.push("        return x");
  }

  lines.push("");

  return { code: lines.join("\n"), errors };
}

function topologicalSort(
  nodes: Node<NodeData>[],
  edges: Edge[],
): Node<NodeData>[] | null {
  const inDegree = new Map<string, number>();
  const adjList = new Map<string, string[]>();

  for (const n of nodes) {
    inDegree.set(n.id, 0);
    adjList.set(n.id, []);
  }

  for (const e of edges) {
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
    adjList.get(e.source)?.push(e.target);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const result: Node<NodeData>[] = [];
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  while (queue.length > 0) {
    const id = queue.shift()!;
    const node = nodeMap.get(id);
    if (node) result.push(node);

    for (const neighbor of adjList.get(id) ?? []) {
      const newDeg = (inDegree.get(neighbor) ?? 1) - 1;
      inDegree.set(neighbor, newDeg);
      if (newDeg === 0) queue.push(neighbor);
    }
  }

  if (result.length !== nodes.length) return null;
  return result;
}

function sanitizeId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_]/g, "_");
}

export function formatCtorArgs(moduleType: string, params: Record<string, unknown>): string {
  // Module-specific defaults we can skip for cleaner output
  const DEFAULTS: Record<string, Record<string, unknown>> = {
    Linear: { bias: true },
    Conv1d: { stride: 1, padding: 0, dilation: 1, groups: 1, bias: true },
    Conv2d: { stride: 1, padding: 0, dilation: 1, groups: 1, bias: true },
    Conv3d: { stride: 1, padding: 0, dilation: 1, groups: 1, bias: true },
    ConvTranspose2d: { stride: 1, padding: 0, output_padding: 0, bias: true },
    BatchNorm1d: { eps: 1e-5, momentum: 0.1, affine: true },
    BatchNorm2d: { eps: 1e-5, momentum: 0.1, affine: true },
    LayerNorm: { eps: 1e-5, elementwise_affine: true },
    GroupNorm: { eps: 1e-5, affine: true },
    RMSNorm: { eps: 1e-5 },
    MaxPool2d: { padding: 0 },
    AvgPool2d: { padding: 0 },
    ReLU: { inplace: false },
    LeakyReLU: { negative_slope: 0.01, inplace: false },
    GELU: { approximate: "none" },
    SiLU: { inplace: false },
    Mish: { inplace: false },
    ELU: { alpha: 1.0, inplace: false },
    PReLU: { num_parameters: 1, init: 0.25 },
    Dropout: { inplace: false },
    Dropout2d: { inplace: false },
    LSTM: { num_layers: 1, bias: true, batch_first: true, dropout: 0, bidirectional: false },
    GRU: { num_layers: 1, bias: true, batch_first: true, dropout: 0, bidirectional: false },
    MultiheadAttention: { dropout: 0.0, bias: true, batch_first: true },
    TransformerEncoderLayer: { dim_feedforward: 2048, dropout: 0.1, activation: "relu", batch_first: true, norm_first: false },
    TransformerDecoderLayer: { dim_feedforward: 2048, dropout: 0.1, activation: "relu", batch_first: true, norm_first: false },
    CrossEntropyLoss: { reduction: "mean", label_smoothing: 0.0 },
    MSELoss: { reduction: "mean" },
    BCEWithLogitsLoss: { reduction: "mean" },
    Flatten: { start_dim: 1, end_dim: -1 },
    Softmax: {},
    Upsample: {},
  };

  const defaults = DEFAULTS[moduleType] ?? {};
  const parts: string[] = [];

  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined) continue;
    // Skip if same as default
    if (k in defaults && defaults[k] === v) continue;
    // Skip stride=null on pooling (means "same as kernel")
    if (v === null) continue;

    if (typeof v === "boolean") {
      parts.push(`${k}=${v ? "True" : "False"}`);
    } else if (typeof v === "string") {
      // Could be a JSON array like "[512]"
      if (v.startsWith("[")) {
        parts.push(`${k}=${v}`);
      } else {
        parts.push(`${k}='${v}'`);
      }
    } else if (Array.isArray(v)) {
      parts.push(`${k}=(${v.join(", ")})`);
    } else {
      parts.push(`${k}=${v}`);
    }
  }

  return parts.join(", ");
}
