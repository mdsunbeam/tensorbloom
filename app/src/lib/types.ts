export type NodeCategory =
  | "io"
  | "data"
  | "linear"
  | "conv"
  | "pool"
  | "norm"
  | "activation"
  | "recurrent"
  | "transformer"
  | "dropout"
  | "embedding"
  | "loss"
  | "reshape"
  | "math"
  | "vision";

export interface ParamDef {
  name: string;
  type: "int" | "float" | "bool" | "string" | "select" | "tuple_int" | "optional_int";
  default: unknown;
  options?: string[]; // for select type
  label?: string;
  min?: number;
  max?: number;
}

export interface ModuleDef {
  type: string; // e.g. "Conv2d"
  label: string;
  category: NodeCategory;
  params: ParamDef[];
  description: string;
  inputHandles: HandleDef[];
  outputHandles: HandleDef[];
}

export interface HandleDef {
  id: string;
  label: string;
  dataType: "tensor" | "tuple" | "any";
}

// ---------------------------------------------------------------------------
// Data Output configuration — explicit column-to-handle mapping
// ---------------------------------------------------------------------------

export interface DataOutputDef {
  id: string;         // handle name (Python var name, e.g., "x", "y", "mask")
  column: string;     // column/key from data source ("" = auto)
  dtype: string;      // "float32", "long", etc.
  shape?: number[];   // per-sample shape (excluding batch dim), from scan
}

// Result from data.scan RPC — tensor metadata without loading full data
export interface ScanTensor {
  name: string;
  shape: number[];
  dtype: string;
  num_samples: number;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
  num_classes?: number;
  class_values?: number[];
}

export interface ScanResult {
  tensors: ScanTensor[];
  total_samples: number;
  format: string;
}

export const DEFAULT_DATA_OUTPUTS: DataOutputDef[] = [
  { id: "x", column: "", dtype: "float32" },
  { id: "y", column: "", dtype: "long" },
];

// ---------------------------------------------------------------------------
// Code Block support — custom forward() + dynamic input/output handles
// ---------------------------------------------------------------------------

export interface CodeBlockDef {
  inputHandles: HandleDef[];
  outputHandles: HandleDef[];
  initBody: string;          // __init__ body (e.g., "self.conv = nn.Conv2d(3, 16, 3)")
  forwardBody: string;       // forward() body (e.g., "return self.conv(x)")
  originalModuleType: string;
  originalParams: Record<string, unknown>;
  originalCustomCode?: string;
}

const PYTHON_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await",
  "break", "class", "continue", "def", "del", "elif", "else", "except",
  "finally", "for", "from", "global", "if", "import", "in", "is",
  "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
  "while", "with", "yield",
]);

export function safePythonId(id: string): string {
  const cleaned = id.replace(/[^a-zA-Z0-9_]/g, "_");
  const base = /^[a-zA-Z_]/.test(cleaned) ? cleaned : `_${cleaned}`;
  return PYTHON_KEYWORDS.has(base) ? `${base}_` : base;
}

/**
 * Resolve the effective input/output handles for a node,
 * taking code block overrides into account.
 */
export function resolveHandles(data: {
  moduleType: string;
  codeBlock?: CodeBlockDef;
}): { inputHandles: HandleDef[]; outputHandles: HandleDef[] } {
  if (data.codeBlock) {
    return {
      inputHandles: data.codeBlock.inputHandles,
      outputHandles: data.codeBlock.outputHandles,
    };
  }
  // Lazy import avoided — caller is expected to have MODULE_REGISTRY access.
  // This is a pure-data utility, so we fall back to sensible defaults.
  // The actual registry lookup happens at each call site.
  return {
    inputHandles: [{ id: "in", label: "in", dataType: "tensor" }],
    outputHandles: [{ id: "out", label: "out", dataType: "tensor" }],
  };
}

/**
 * Resolve handles using MODULE_REGISTRY when available.
 * This is the preferred call-site helper.
 */
export function resolveHandlesWithRegistry(
  data: { moduleType: string; codeBlock?: CodeBlockDef; dataOutputs?: DataOutputDef[] },
  registry: Record<string, ModuleDef>,
): { inputHandles: HandleDef[]; outputHandles: HandleDef[] } {
  if (data.codeBlock) {
    return {
      inputHandles: data.codeBlock.inputHandles,
      outputHandles: data.codeBlock.outputHandles,
    };
  }
  // Data nodes with explicit output handles
  if (data.dataOutputs && data.dataOutputs.length > 0) {
    return {
      inputHandles: [],
      outputHandles: data.dataOutputs.map((d) => ({
        id: d.id,
        label: d.column || d.id,
        dataType: "tensor" as const,
      })),
    };
  }
  const def = registry[data.moduleType];
  return {
    inputHandles: def?.inputHandles ?? [{ id: "in", label: "in", dataType: "tensor" }],
    outputHandles: def?.outputHandles ?? [{ id: "out", label: "out", dataType: "tensor" }],
  };
}

export interface TensorLabNode {
  id: string;
  type: string; // matches ModuleDef.type key in registry
  position: { x: number; y: number };
  data: {
    moduleType: string;
    label: string;
    category: NodeCategory;
    params: Record<string, unknown>;
    shapeInfo?: { input?: number[][]; output?: number[][] };
  };
}

export interface TensorLabEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
}

export interface TrainingConfig {
  optimizer: { type: string; params: Record<string, unknown> };
  scheduler?: { type: string; params: Record<string, unknown> };
  loss: { type: string; params: Record<string, unknown> };
  epochs: number;
  batchSize: number;
  dataset: { type: string; params: Record<string, unknown> };
}

export interface ProjectFile {
  version: string;
  name: string;
  nodes: TensorLabNode[];
  edges: TensorLabEdge[];
  training?: TrainingConfig;
}

// JSON-RPC types for Python sidecar communication
export interface JsonRpcRequest {
  jsonrpc: "2.0";
  method: string;
  params?: unknown;
  id: number;
}

export interface JsonRpcResponse {
  jsonrpc: "2.0";
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
  id: number;
}
