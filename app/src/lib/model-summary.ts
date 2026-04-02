/**
 * Client-side parameter count and model summary.
 */

import type { Node } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";

export interface ModelSummary {
  totalParams: number;
  trainableParams: number;
  perNode: Record<string, number>;
  estimatedSizeMB: number;
  estimatedFlops: number;
  estimatedVramMB: number;
}

export function computeModelSummary(nodes: Node<NodeData>[]): ModelSummary {
  const perNode: Record<string, number> = {};
  let total = 0;
  let totalFlops = 0;

  for (const n of nodes) {
    const count = countParams(n.data.moduleType, n.data.params);
    perNode[n.id] = count;
    total += count;
    totalFlops += estimateFlops(n.data.moduleType, n.data.params);
  }

  const modelSizeMB = (total * 4) / (1024 * 1024);
  // VRAM estimate: model params + gradients + optimizer state (Adam: 2x) + activations (~2x model)
  const estimatedVramMB = modelSizeMB * 6;

  return {
    totalParams: total,
    trainableParams: total,
    perNode,
    estimatedSizeMB: modelSizeMB,
    estimatedFlops: totalFlops,
    estimatedVramMB,
  };
}

function estimateFlops(moduleType: string, p: Record<string, unknown>): number {
  switch (moduleType) {
    case "Linear": {
      const inF = Number(p.in_features ?? 0);
      const outF = Number(p.out_features ?? 0);
      return 2 * inF * outF; // multiply-accumulate
    }
    case "Conv2d": {
      const inC = Number(p.in_channels ?? 0);
      const outC = Number(p.out_channels ?? 0);
      const k = Number(p.kernel_size ?? 3);
      const groups = Number(p.groups ?? 1);
      // Assume 224x224 input scaled by stride — rough estimate
      const s = Number(p.stride ?? 1);
      const outH = 224 / s; // very rough
      return 2 * outC * (inC / groups) * k * k * outH * outH;
    }
    case "Conv1d": {
      const inC = Number(p.in_channels ?? 0);
      const outC = Number(p.out_channels ?? 0);
      const k = Number(p.kernel_size ?? 3);
      return 2 * outC * inC * k * 1000; // assume ~1000 length
    }
    case "MultiheadAttention": {
      const d = Number(p.embed_dim ?? 0);
      const seq = 128; // assume
      return 4 * seq * d * d + 2 * seq * seq * d; // QKV proj + attention
    }
    case "TransformerEncoderLayer":
    case "TransformerDecoderLayer": {
      const d = Number(p.d_model ?? 0);
      const ff = Number(p.dim_feedforward ?? 2048);
      const seq = 128;
      return 4 * seq * d * d + 2 * seq * seq * d + 4 * seq * d * ff;
    }
    case "LSTM": {
      const inp = Number(p.input_size ?? 0);
      const hid = Number(p.hidden_size ?? 0);
      const seq = 100;
      const bi = p.bidirectional ? 2 : 1;
      return bi * 8 * hid * (inp + hid) * seq;
    }
    default:
      return 0;
  }
}

function countParams(moduleType: string, p: Record<string, unknown>): number {
  switch (moduleType) {
    case "Linear": {
      const inF = Number(p.in_features ?? 0);
      const outF = Number(p.out_features ?? 0);
      const bias = p.bias !== false;
      return inF * outF + (bias ? outF : 0);
    }
    case "Conv1d": {
      const inC = Number(p.in_channels ?? 0);
      const outC = Number(p.out_channels ?? 0);
      const k = Number(p.kernel_size ?? 3);
      const groups = Number(p.groups ?? 1);
      const bias = p.bias !== false;
      return outC * (inC / groups) * k + (bias ? outC : 0);
    }
    case "Conv2d": {
      const inC = Number(p.in_channels ?? 0);
      const outC = Number(p.out_channels ?? 0);
      const k = Number(p.kernel_size ?? 3);
      const groups = Number(p.groups ?? 1);
      const bias = p.bias !== false;
      return outC * (inC / groups) * k * k + (bias ? outC : 0);
    }
    case "Conv3d": {
      const inC = Number(p.in_channels ?? 0);
      const outC = Number(p.out_channels ?? 0);
      const k = Number(p.kernel_size ?? 3);
      const groups = Number(p.groups ?? 1);
      const bias = p.bias !== false;
      return outC * (inC / groups) * k * k * k + (bias ? outC : 0);
    }
    case "ConvTranspose2d": {
      const inC = Number(p.in_channels ?? 0);
      const outC = Number(p.out_channels ?? 0);
      const k = Number(p.kernel_size ?? 3);
      const bias = p.bias !== false;
      return inC * outC * k * k + (bias ? outC : 0);
    }
    case "BatchNorm1d":
    case "BatchNorm2d":
    case "BatchNorm3d": {
      const nf = Number(p.num_features ?? 0);
      const affine = p.affine !== false;
      return affine ? 2 * nf : 0;
    }
    case "LayerNorm": {
      const shape = parseShape(p.normalized_shape);
      const prod = shape.reduce((a, b) => a * b, 1);
      const affine = p.elementwise_affine !== false;
      return affine ? 2 * prod : 0;
    }
    case "RMSNorm": {
      const shape = parseShape(p.normalized_shape);
      return shape.reduce((a, b) => a * b, 1);
    }
    case "GroupNorm": {
      const nc = Number(p.num_channels ?? 0);
      const affine = p.affine !== false;
      return affine ? 2 * nc : 0;
    }
    case "Embedding": {
      return Number(p.num_embeddings ?? 0) * Number(p.embedding_dim ?? 0);
    }
    case "LSTM": {
      const inp = Number(p.input_size ?? 0);
      const hid = Number(p.hidden_size ?? 0);
      const layers = Number(p.num_layers ?? 1);
      const bi = p.bidirectional ? 2 : 1;
      const bias = p.bias !== false ? 1 : 0;
      // First layer
      let total = bi * 4 * hid * (inp + hid + (bias ? 2 : 0));
      // Subsequent layers
      for (let i = 1; i < layers; i++) {
        total += bi * 4 * hid * (bi * hid + hid + (bias ? 2 : 0));
      }
      return total;
    }
    case "GRU": {
      const inp = Number(p.input_size ?? 0);
      const hid = Number(p.hidden_size ?? 0);
      const layers = Number(p.num_layers ?? 1);
      const bi = p.bidirectional ? 2 : 1;
      const bias = p.bias !== false ? 1 : 0;
      let total = bi * 3 * hid * (inp + hid + (bias ? 2 : 0));
      for (let i = 1; i < layers; i++) {
        total += bi * 3 * hid * (bi * hid + hid + (bias ? 2 : 0));
      }
      return total;
    }
    case "MultiheadAttention": {
      const d = Number(p.embed_dim ?? 0);
      const bias = p.bias !== false;
      // Q, K, V in-proj + out-proj
      return 3 * d * d + d * d + (bias ? 4 * d : 0);
    }
    case "TransformerEncoderLayer": {
      const d = Number(p.d_model ?? 0);
      const ff = Number(p.dim_feedforward ?? 2048);
      // MHA: 4*d^2 + 4*d, FFN: d*ff + ff + ff*d + d, 2x LayerNorm: 4*d
      return 4 * d * d + 4 * d + 2 * d * ff + ff + d + 4 * d;
    }
    case "TransformerDecoderLayer": {
      const d = Number(p.d_model ?? 0);
      const ff = Number(p.dim_feedforward ?? 2048);
      // Self-attn + cross-attn + FFN + 3x LayerNorm
      return 2 * (4 * d * d + 4 * d) + 2 * d * ff + ff + d + 6 * d;
    }
    case "PReLU": {
      return Number(p.num_parameters ?? 1);
    }
    default:
      return 0;
  }
}

function parseShape(val: unknown): number[] {
  if (Array.isArray(val)) return val.map(Number);
  if (typeof val === "string") {
    try {
      const parsed = JSON.parse(val.replace(/\(/g, "[").replace(/\)/g, "]"));
      if (Array.isArray(parsed)) return parsed.map(Number);
    } catch { /* ignore */ }
  }
  if (typeof val === "number") return [val];
  return [0];
}

export function formatParamCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function formatFlops(n: number): string {
  if (n >= 1e15) return `${(n / 1e15).toFixed(1)} PFLOPs`;
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)} TFLOPs`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GFLOPs`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MFLOPs`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KFLOPs`;
  return `${n} FLOPs`;
}
