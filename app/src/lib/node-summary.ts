import type { NodeData } from "../store/graph-store";
import { DATASET_METADATA, isDatasetNode, DATA_PRESETS, resolveDataNodeShape } from "./dataset-metadata";

// Inline param count — avoids importing the full summary module for every node
export function quickParamCount(mt: string, p: Record<string, unknown>): number {
  switch (mt) {
    case "Linear": return Number(p.in_features ?? 0) * Number(p.out_features ?? 0) + (p.bias !== false ? Number(p.out_features ?? 0) : 0);
    case "Conv1d": { const k = Number(p.kernel_size ?? 3); return Number(p.out_channels ?? 0) * (Number(p.in_channels ?? 0) / Number(p.groups ?? 1)) * k + (p.bias !== false ? Number(p.out_channels ?? 0) : 0); }
    case "Conv2d": { const k = Number(p.kernel_size ?? 3); return Number(p.out_channels ?? 0) * (Number(p.in_channels ?? 0) / Number(p.groups ?? 1)) * k * k + (p.bias !== false ? Number(p.out_channels ?? 0) : 0); }
    case "Conv3d": { const k = Number(p.kernel_size ?? 3); return Number(p.out_channels ?? 0) * (Number(p.in_channels ?? 0) / Number(p.groups ?? 1)) * k * k * k + (p.bias !== false ? Number(p.out_channels ?? 0) : 0); }
    case "ConvTranspose1d": { const k = Number(p.kernel_size ?? 3); return Number(p.in_channels ?? 0) * Number(p.out_channels ?? 0) * k + (p.bias !== false ? Number(p.out_channels ?? 0) : 0); }
    case "ConvTranspose2d": { const k = Number(p.kernel_size ?? 3); return Number(p.in_channels ?? 0) * Number(p.out_channels ?? 0) * k * k + (p.bias !== false ? Number(p.out_channels ?? 0) : 0); }
    case "Embedding": return Number(p.num_embeddings ?? 0) * Number(p.embedding_dim ?? 0);
    case "LSTM": { const h = Number(p.hidden_size ?? 0); const inp = Number(p.input_size ?? 0); const bi = p.bidirectional ? 2 : 1; return bi * 4 * h * (inp + h + 2); }
    case "MultiheadAttention": { const d = Number(p.embed_dim ?? 0); return 4 * d * d + 4 * d; }
    case "TransformerEncoderLayer": { const d = Number(p.d_model ?? 0); const ff = Number(p.dim_feedforward ?? 2048); return 4 * d * d + 4 * d + 2 * d * ff + ff + d + 4 * d; }
    default: return 0;
  }
}

/** The one-line parameter summary shown under a node's title, e.g. "784 → 512" */
export function getParamSummary(data: NodeData): string {
  const p = data.params;
  // Universal Data node
  if (data.moduleType === "Data") {
    const source = String(p.source ?? "");
    if (!source) return "Select a source";
    const preset = DATA_PRESETS[source];
    if (preset) {
      return `${preset.label}: ${JSON.stringify(preset.meta.inputShape)} \u2192 ${preset.meta.numClasses} cls`;
    }
    // Show the user-specified identifier for custom sources
    const identifier = source === "HuggingFace" ? String(p.dataset_name ?? "")
      : ["ImageFolder", "CSV", "TensorFile"].includes(source) ? String(p.path ?? "")
      : "";
    // TensorFile with scan result: just show sample count
    const scan = p._scan as { total_samples?: number } | undefined;
    if (scan && scan.total_samples) {
      return `${scan.total_samples.toLocaleString()} samples`;
    }
    const resolved = resolveDataNodeShape(p);
    if (resolved) {
      return `${identifier || source}: ${JSON.stringify(resolved.inputShape)}`;
    }
    return identifier || source;
  }
  // Legacy dataset nodes
  if (isDatasetNode(data.moduleType)) {
    const meta = DATASET_METADATA[data.moduleType];
    if (meta) return `${JSON.stringify(meta.inputShape)} \u2192 ${meta.numClasses} cls`;
    return String(p.root ?? "");
  }
  switch (data.moduleType) {
    case "Linear": return `${p.in_features} \u2192 ${p.out_features}`;
    case "Conv1d": case "Conv2d": case "Conv3d": return `${p.in_channels} \u2192 ${p.out_channels}, k=${p.kernel_size}`;
    case "ConvTranspose1d": case "ConvTranspose2d": return `${p.in_channels} \u2192 ${p.out_channels}, k=${p.kernel_size}`;
    case "MaxPool2d": case "AvgPool2d": return `k=${p.kernel_size}`;
    case "BatchNorm1d": case "BatchNorm2d": return `features=${p.num_features}`;
    case "LayerNorm": case "RMSNorm": return `shape=${p.normalized_shape}`;
    case "Dropout": case "Dropout2d": return `p=${p.p}`;
    case "Embedding": return `${p.num_embeddings} \u00d7 ${p.embedding_dim}`;
    case "LSTM": case "GRU": return `${p.input_size} \u2192 ${p.hidden_size}, layers=${p.num_layers}`;
    case "MultiheadAttention": return `d=${p.embed_dim}, heads=${p.num_heads}`;
    case "TransformerEncoderLayer": case "TransformerDecoderLayer": return `d=${p.d_model}, heads=${p.nhead}`;
    case "Flatten": return `dim ${p.start_dim} \u2192 ${p.end_dim}`;
    case "Softmax": return `dim=${p.dim}`;
    case "AdaptiveAvgPool2d": return `out=${JSON.stringify(p.output_size)}`;
    case "Upsample": return `\u00d7${p.scale_factor} ${p.mode}`;
    default: return "";
  }
}
