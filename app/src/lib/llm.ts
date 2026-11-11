/**
 * Large language models: the architecture read from a Hugging Face model or a local file (a "spec"),
 * the graph drawn from it, and estimates of the memory it needs to run or fine-tune.
 *
 * The sidecar's llm.inspect produces the spec; python/tensorbloom_llm.py documents each field.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { NodeCategory } from "./types";
import { MODULE_REGISTRY } from "./registry";
import { autoLayout } from "./auto-layout";
import { layersBounds, CARD_HEIGHT, type LayerGroup } from "./groups";

export interface LLMSpec {
  family: string;
  name: string;
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  head_dim: number;
  ffn_dim: number;
  mlp: "swiglu" | "geglu" | "gelu";
  norm: "rmsnorm" | "layernorm";
  norm_eps: number;
  rope_theta: number;
  learned_positions: boolean;
  max_context: number;
  tie_embeddings: boolean;
  bias: "none" | "qkv" | "all";
  n_experts: number;
  experts_per_token: number;
  /** Parameter count: exact when read from the weight files' headers, otherwise computed from the config */
  params: number;
  params_exact?: boolean;
  /** How the weights are stored, e.g. "bf16" or "q4_k_m" */
  stored_dtype?: string | null;
  /** Total size of the weight files, when known */
  stored_bytes?: number;
  quantization?: { method: string; type?: string; bits?: number | null };
  torch_dtype?: string | null;
  warnings: string[];
  source?: { kind: "hf"; model_id: string; revision: string } | { kind: "path"; path: string };
  /** Whether PyTorch can load these weights (false for GGUF) */
  loadable?: boolean;
  local_path?: string;
}

/** What the sidecar's llm.device reports: the memory an LLM can use on this machine */
export interface LLMDevice {
  kind: "cuda" | "mps" | "cpu";
  name: string;
  /** Bytes the model can use: the largest GPU's memory, most of Apple silicon's shared memory, or RAM */
  memory: number | null;
  ram_total: number | null;
  gpu_count?: number;
  /** bitsandbytes 4- and 8-bit loading works here (CUDA only) */
  quantization_4bit: boolean;
  packages: Record<string, string | null>;
}

export function headDim(spec: LLMSpec): number {
  return spec.head_dim || Math.floor(spec.d_model / spec.n_heads);
}

/** Parameters of one decoder block (mirrors count_params in tensorbloom_llm.py) */
export function decoderBlockParams(p: {
  d_model: number; n_heads: number; n_kv_heads: number; head_dim?: number; ffn_dim: number;
  mlp?: string; norm?: string; bias?: string; n_experts?: number;
}): number {
  const d = p.d_model, H = p.n_heads, KV = p.n_kv_heads, F = p.ffn_dim;
  const hd = p.head_dim || Math.floor(d / H);
  const bias = p.bias ?? "none";
  const gated = (p.mlp ?? "swiglu") !== "gelu";
  const norm = (p.norm ?? "rmsnorm") === "rmsnorm" ? d : 2 * d;
  let attn = d * H * hd + 2 * d * KV * hd + H * hd * d;
  if (bias === "qkv" || bias === "all") attn += H * hd + 2 * KV * hd;
  if (bias === "all") attn += d;
  let mlp = (gated ? 3 : 2) * d * F + (bias === "all" ? (gated ? 2 * F : F) + d : 0);
  const E = p.n_experts ?? 0;
  if (E) mlp = E * mlp + d * E;
  return attn + mlp + 2 * norm;
}

export function countLLMParams(spec: LLMSpec): number {
  const d = spec.d_model, V = spec.vocab_size;
  const norm = spec.norm === "rmsnorm" ? d : 2 * d;
  let total = V * d + spec.n_layers * decoderBlockParams(spec) + norm;
  if (!spec.tie_embeddings) total += V * d;
  if (spec.learned_positions) total += spec.max_context * d;
  return total;
}

/** Parameters each token actually passes through: all of them, except in a mixture of experts */
export function activeParams(spec: LLMSpec): number {
  if (!spec.n_experts) return spec.params;
  const expert = decoderBlockParams({ ...spec, n_experts: 0 }) - decoderBlockParams({ ...spec, n_experts: 0, ffn_dim: 0 });
  return spec.params - spec.n_layers * (spec.n_experts - spec.experts_per_token) * expert;
}

// ─── Graph ──────────────────────────────────────────────────────────────────

/** Context length the imported graph is drawn with: the model's own, capped so shapes stay readable */
export function defaultGraphContext(spec: LLMSpec): number {
  return Math.min(spec.max_context || 2048, 4096);
}

/**
 * The graph for an LLM: tokens → embedding → N decoder blocks (in one collapsed group) → norm → lm_head.
 * Every node is a real layer, so the graph's code view and shapes work as for any other model.
 */
export function buildLLMGraph(spec: LLMSpec, context = defaultGraphContext(spec)): {
  nodes: Node<NodeData>[]; edges: Edge[]; groups: LayerGroup[]; notes: string[];
} {
  let i = 0;
  const node = (moduleType: string, label: string, params: Record<string, unknown> = {}): Node<NodeData> => {
    const def = MODULE_REGISTRY[moduleType];
    const defaults = Object.fromEntries((def?.params ?? []).map((p) => [p.name, p.default]));
    return {
      id: `llm_${++i}`,
      type: "tensorLabNode",
      position: { x: 0, y: 0 },
      data: { moduleType, label, category: (def?.category ?? "linear") as NodeCategory, params: { ...defaults, ...params } },
    };
  };
  const d = spec.d_model;
  const blocks = Array.from({ length: spec.n_layers }, (_, k) => node("DecoderBlock", `layer_${k}`, {
    d_model: d,
    n_heads: spec.n_heads,
    n_kv_heads: spec.n_kv_heads,
    head_dim: headDim(spec),
    ffn_dim: spec.ffn_dim,
    mlp: spec.mlp,
    norm: spec.norm,
    norm_eps: spec.norm_eps,
    rope_theta: spec.rope_theta,
    bias: spec.bias,
    n_experts: spec.n_experts,
    experts_per_token: spec.experts_per_token || 2,
  }));
  const nodes = [
    node("Input", "tokens", { shape: `[1, ${context}]`, dtype: "int64" }),
    node("Embedding", "embed_tokens", { num_embeddings: spec.vocab_size, embedding_dim: d }),
    ...blocks,
    spec.norm === "rmsnorm"
      ? node("RMSNorm", "norm", { normalized_shape: `[${d}]`, eps: spec.norm_eps })
      : node("LayerNorm", "norm", { normalized_shape: `[${d}]`, eps: spec.norm_eps }),
    node("Linear", spec.tie_embeddings ? "lm_head (tied)" : "lm_head", { in_features: d, out_features: spec.vocab_size, bias: false }),
    node("Output", "logits"),
  ];
  const edges: Edge[] = nodes.slice(1).map((n, k) => ({
    id: `e_${nodes[k].id}_${n.id}`,
    source: nodes[k].id,
    target: n.id,
    sourceHandle: "out",
    targetHandle: "in",
    type: "smoothstep",
  }));
  const notes: string[] = [];
  if (spec.learned_positions) notes.push("This model adds learned position embeddings to the tokens; the graph leaves them out.");
  if (spec.tie_embeddings) notes.push("lm_head shares its weights with embed_tokens in this model. The graph gives it its own, so its parameter count is higher than the model's.");
  const group: LayerGroup = { id: "llm_blocks", name: `${spec.n_layers} decoder blocks`, nodeIds: blocks.map((b) => b.id), collapsed: true };
  // Lay out with the blocks open, then fold their space away under the collapsed card (as collapsing a group does)
  const laidOut = autoLayout(nodes, edges);
  const bounds = layersBounds(group, laidOut);
  const inside = new Set(group.nodeIds);
  const fold = bounds ? Math.max(0, bounds.h - CARD_HEIGHT) : 0;
  const below = bounds ? bounds.y + bounds.h - 1 : Infinity;
  const folded = laidOut.map((n) => (!inside.has(n.id) && n.position.y >= below ? { ...n, position: { x: n.position.x, y: n.position.y - fold } } : n));
  return { nodes: folded, edges, groups: [group], notes };
}

// ─── Memory ─────────────────────────────────────────────────────────────────

export type Precision = "fp32" | "bf16" | "int8" | "nf4" | "stored";

export const PRECISION_LABELS: Record<Precision, string> = {
  fp32: "32-bit float",
  bf16: "16-bit (bf16)",
  int8: "8-bit",
  nf4: "4-bit (NF4)",
  stored: "As stored",
};

const GiB = 1024 ** 3;

/** Weights that bitsandbytes leaves at 16 bits when it quantizes the rest: the embedding and lm_head */
function unquantizedParams(spec: LLMSpec): number {
  return spec.vocab_size * spec.d_model * (spec.tie_embeddings ? 1 : 2);
}

/** Bytes for the weights alone */
export function weightBytes(spec: LLMSpec, precision: Precision): number {
  if (precision === "stored") return spec.stored_bytes ?? weightBytes(spec, "bf16");
  if (precision === "fp32") return spec.params * 4;
  if (precision === "bf16") return spec.params * 2;
  const keep = Math.min(unquantizedParams(spec), spec.params);
  // 8-bit: one byte per weight plus per-row scales; NF4: half a byte plus a scale per 64 weights
  const perWeight = precision === "int8" ? 1.0 : 0.516;
  return (spec.params - keep) * perWeight + keep * 2;
}

/** Bytes for the KV cache: keys and values for every layer and token in the context, at 16 bits */
export function kvCacheBytes(spec: LLMSpec, context: number, batch = 1): number {
  return 2 * spec.n_layers * spec.n_kv_heads * headDim(spec) * context * batch * 2;
}

/** Trainable parameters LoRA adds at this rank, on every linear layer of every block */
export function loraParams(spec: LLMSpec, rank = 16): number {
  const d = spec.d_model, hd = headDim(spec), F = spec.ffn_dim;
  const attn = rank * ((d + spec.n_heads * hd) * 2 + (d + spec.n_kv_heads * hd) * 2);
  const mlpMatrices = spec.mlp === "gelu" ? 2 : 3;
  const mlp = rank * mlpMatrices * (d + F) * Math.max(1, spec.n_experts);
  return spec.n_layers * (attn + mlp);
}

/** Activation memory for one training step, with gradient checkpointing (as fine-tuning uses) */
function trainingActivationBytes(spec: LLMSpec, context: number, batch: number): number {
  const tokens = context * batch;
  const checkpoints = spec.n_layers * tokens * spec.d_model * 2;
  const oneLayer = tokens * (10 * spec.d_model + 3 * spec.ffn_dim) * 2;
  // Logits and their gradient, in fp32 for the loss
  const logits = tokens * spec.vocab_size * 4 * 2;
  return checkpoints + oneLayer + logits;
}

/** Framework overhead: CUDA context, allocator slack, temporary buffers */
function overheadBytes(weights: number): number {
  return 0.5 * GiB + 0.05 * weights;
}

export type Task = "run" | "lora" | "full";

export const TASK_LABELS: Record<Task, string> = {
  run: "Run (chat)",
  lora: "LoRA fine-tune",
  full: "Full fine-tune",
};

export interface MemoryEstimate {
  total: number;
  parts: { label: string; bytes: number }[];
}

/**
 * Rough memory needed for a task at a precision. "lora" at nf4 is QLoRA.
 * Returns null where the combination doesn't make sense (full fine-tuning of quantized weights).
 */
export function estimateMemory(spec: LLMSpec, task: Task, precision: Precision, context: number, batch = 1, rank = 16): MemoryEstimate | null {
  const weights = weightBytes(spec, precision);
  if (task === "run") {
    const kv = kvCacheBytes(spec, context, batch);
    const parts = [
      { label: "Weights", bytes: weights },
      { label: `KV cache (${context.toLocaleString()} tokens)`, bytes: kv },
      { label: "Overhead", bytes: overheadBytes(weights) },
    ];
    return { total: parts.reduce((s, p) => s + p.bytes, 0), parts };
  }
  if (precision === "stored") return null;
  if (task === "full") {
    if (precision !== "bf16" && precision !== "fp32") return null;
    // Mixed precision with AdamW: fp32 master weights (4) + gradients (4) + two optimizer moments (8)
    const state = spec.params * 16;
    const parts = [
      { label: "Weights, gradients and optimizer state", bytes: state },
      { label: "Activations", bytes: trainingActivationBytes(spec, context, batch) },
      { label: "Overhead", bytes: overheadBytes(weights) },
    ];
    return { total: parts.reduce((s, p) => s + p.bytes, 0), parts };
  }
  const lora = loraParams(spec, rank);
  const parts = [
    { label: "Frozen weights", bytes: weights },
    { label: `LoRA adapters (rank ${rank}) and optimizer`, bytes: lora * 16 },
    { label: "Activations", bytes: trainingActivationBytes(spec, context, batch) },
    { label: "Overhead", bytes: overheadBytes(weights) },
  ];
  return { total: parts.reduce((s, p) => s + p.bytes, 0), parts };
}

export type Fit = "fits" | "tight" | "too-big" | "unknown";

export function fitFor(bytes: number, available: number | null | undefined): Fit {
  if (!available) return "unknown";
  if (bytes <= available * 0.85) return "fits";
  if (bytes <= available) return "tight";
  return "too-big";
}

/** Precisions worth showing for a model: what it's stored as, then the usual loading options */
export function precisionsFor(spec: LLMSpec): Precision[] {
  if (spec.quantization?.method === "gguf") return ["stored", "bf16"];
  return ["bf16", "int8", "nf4"];
}

export function formatBytes(bytes: number): string {
  if (bytes >= GiB) return `${(bytes / GiB).toFixed(bytes >= 100 * GiB ? 0 : 1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}
