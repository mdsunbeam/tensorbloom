/**
 * Pre-built architecture templates — curated gold-standard models.
 * Returns nodes and edges that can be loaded into the graph.
 */

import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";
import type { NodeCategory } from "./types";
import { DEFAULT_DATA_OUTPUTS } from "./types";
import { MODULE_REGISTRY } from "./registry";

export type TemplateCategory = "getting-started" | "vision" | "language" | "audio" | "generative" | "other";

export interface TemplateTrainingDefaults {
  optimizer?: { type: string; lr: number; weightDecay: number; momentum?: number };
  loss?: string;
  epochs?: number;
  batchSize?: number;
  dataset?: { type: string; root: string };
}

export interface Template {
  name: string;
  description: string;
  category: TemplateCategory;
  year?: number;
  paper?: string;
  /** False for unsupervised (autoencoder, etc.) — no Data.y → Loss.target edge. Default true. */
  supervised?: boolean;
  trainingDefaults?: TemplateTrainingDefaults;
  build: () => { nodes: Node<NodeData>[]; edges: Edge[] };
}

let _id = 0;
function nid(): string {
  return `tpl_${++_id}`;
}

function makeNode(
  moduleType: string,
  overrides?: Partial<Record<string, unknown>>,
  label?: string,
): Node<NodeData> {
  const def = MODULE_REGISTRY[moduleType];
  const defaults: Record<string, unknown> = {};
  if (def) {
    for (const p of def.params) defaults[p.name] = p.default;
  }
  return {
    id: nid(),
    type: "tensorLabNode",
    position: { x: 0, y: 0 },
    data: {
      moduleType,
      label: label ?? def?.label ?? moduleType,
      category: (def?.category ?? "linear") as NodeCategory,
      params: { ...defaults, ...overrides },
      ...(moduleType === "Data" ? { dataOutputs: [...DEFAULT_DATA_OUTPUTS] } : {}),
    },
  };
}

const LOSS_TYPES = new Set([
  "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "BCELoss",
  "L1Loss", "HuberLoss", "NLLLoss", "KLDivLoss",
  "TripletMarginLoss", "CosineEmbeddingLoss", "CTCLoss", "SmoothL1Loss",
]);

function chain(nodes: Node<NodeData>[]): Edge[] {
  const edges: Edge[] = [];
  for (let i = 1; i < nodes.length; i++) {
    const tgtMt = nodes[i].data.moduleType;
    // Data node outputs use "x" handle; Loss node prediction input uses "input" handle
    const sourceHandle = nodes[i - 1].data.dataOutputs ? "x" : "out";
    const targetHandle = LOSS_TYPES.has(tgtMt) ? "input" : "in";
    edges.push({
      id: `e_${nodes[i - 1].id}_${nodes[i].id}`,
      source: nodes[i - 1].id,
      target: nodes[i].id,
      sourceHandle,
      targetHandle,
      type: "smoothstep",
    });
  }
  return edges;
}

/**
 * Prepend a Dataset node connected to root computation nodes (those with no incoming edges).
 * Called at template load time to inject the dataset block into the graph.
 */
export function prependDatasetNode(
  result: { nodes: Node<NodeData>[]; edges: Edge[] },
  datasetType: string,
  supervised: boolean = true,
): { nodes: Node<NodeData>[]; edges: Edge[] } {
  // Map sidecar dataset type strings to source values
  const sourceMap: Record<string, string> = { "Custom CSV": "CSV", CustomCSV: "CSV" };
  const source = sourceMap[datasetType] ?? datasetType;

  // Create a universal Data node with the appropriate source preset
  const dsNode = makeNode("Data", { source, domain: "auto" });

  // Find root computation nodes (no incoming edges, not loss nodes)
  const targetsWithEdges = new Set(result.edges.map(e => e.target));
  const rootNodes = result.nodes.filter(n =>
    !targetsWithEdges.has(n.id) && !LOSS_TYPES.has(n.data.moduleType)
  );
  const lossNodes = result.nodes.filter((n) => LOSS_TYPES.has(n.data.moduleType));

  const dsEdges: Edge[] = [];
  // Connect Data[x] → root computation nodes
  for (const rootNode of rootNodes) {
    dsEdges.push({
      id: `e_${dsNode.id}_${rootNode.id}`,
      source: dsNode.id,
      target: rootNode.id,
      sourceHandle: "x",
      targetHandle: "in",
      type: "smoothstep",
    });
  }
  // Connect Data → Loss.target nodes
  // Supervised: Data[y] (labels) → Loss.target
  // Reconstruction: Data[x] (input) → Loss.target (compare output vs input)
  for (const lossNode of lossNodes) {
    dsEdges.push({
      id: `e_${dsNode.id}_${lossNode.id}`,
      source: dsNode.id,
      target: lossNode.id,
      sourceHandle: supervised ? "y" : "x",
      targetHandle: "target",
      type: "smoothstep",
    });
  }

  return {
    nodes: [dsNode, ...result.nodes],
    edges: [...dsEdges, ...result.edges],
  };
}

// ─── Category display order & labels ────────────────────────────────────────

export const TEMPLATE_CATEGORY_ORDER: TemplateCategory[] = [
  "getting-started", "vision", "language", "audio", "generative", "other",
];

export const TEMPLATE_CATEGORY_LABELS: Record<TemplateCategory, string> = {
  "getting-started": "Getting Started",
  vision: "Vision",
  language: "Language",
  audio: "Audio",
  generative: "Generative",
  other: "Other",
};

// ─── Templates ──────────────────────────────────────────────────────────────

export const TEMPLATES: Template[] = [
  // ═══════════════════════════════════════════════════════════════════════════
  // GETTING STARTED
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Simple MLP",
    description: "Flatten → 3-layer feedforward network for MNIST digit classification",
    category: "getting-started",
    trainingDefaults: { optimizer: { type: "Adam", lr: 1e-3, weightDecay: 0 }, loss: "CrossEntropyLoss", epochs: 10, batchSize: 64, dataset: { type: "MNIST", root: "./data" } },
    build() {
      _id = 0;
      const nodes = [
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten"),
        makeNode("Linear", { in_features: 784, out_features: 512 }, "fc1"),
        makeNode("ReLU", {}, "relu1"),
        makeNode("Dropout", { p: 0.2 }),
        makeNode("Linear", { in_features: 512, out_features: 256 }, "fc2"),
        makeNode("ReLU", {}, "relu2"),
        makeNode("Dropout", { p: 0.2 }),
        makeNode("Linear", { in_features: 256, out_features: 10 }, "fc3"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // VISION
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "LeNet-5",
    description: "The original CNN for handwritten digit recognition (MNIST)",
    category: "vision",
    year: 1998,
    paper: "LeCun et al., 1998",
    trainingDefaults: { optimizer: { type: "SGD", lr: 0.01, weightDecay: 0, momentum: 0.9 }, loss: "CrossEntropyLoss", epochs: 15, batchSize: 64, dataset: { type: "MNIST", root: "./data" } },
    build() {
      _id = 0;
      const nodes = [
        makeNode("Conv2d", { in_channels: 1, out_channels: 6, kernel_size: 5, padding: 0, bias: true }, "conv1"),
        makeNode("Tanh", {}, "tanh1"),
        makeNode("AvgPool2d", { kernel_size: 2 }, "pool1"),
        makeNode("Conv2d", { in_channels: 6, out_channels: 16, kernel_size: 5, padding: 0, bias: true }, "conv2"),
        makeNode("Tanh", {}, "tanh2"),
        makeNode("AvgPool2d", { kernel_size: 2 }, "pool2"),
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten"),
        makeNode("Linear", { in_features: 256, out_features: 120 }, "fc1"),
        makeNode("Tanh", {}, "tanh3"),
        makeNode("Linear", { in_features: 120, out_features: 84 }, "fc2"),
        makeNode("Tanh", {}, "tanh4"),
        makeNode("Linear", { in_features: 84, out_features: 10 }, "fc3"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },
  {
    name: "ResNet-18",
    description: "Residual blocks with skip connections — stem + 2 blocks with downsample projection",
    category: "vision",
    year: 2015,
    paper: "He et al., 2015",
    trainingDefaults: { optimizer: { type: "SGD", lr: 0.1, weightDecay: 0, momentum: 0.9 }, loss: "CrossEntropyLoss", epochs: 100, batchSize: 128, dataset: { type: "CIFAR10", root: "./data" } },
    build() {
      _id = 0;
      // Stem
      const stem_conv = makeNode("Conv2d", { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3, bias: false }, "stem_conv");
      const stem_bn = makeNode("BatchNorm2d", { num_features: 64 }, "stem_bn");
      const stem_relu = makeNode("ReLU", {}, "stem_relu");
      const stem_pool = makeNode("MaxPool2d", { kernel_size: 3, stride: 2, padding: 1 }, "stem_pool");
      // Block 1 (64→64, no downsample)
      const b1_conv1 = makeNode("Conv2d", { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1, bias: false }, "block1_conv1");
      const b1_bn1 = makeNode("BatchNorm2d", { num_features: 64 }, "block1_bn1");
      const b1_relu1 = makeNode("ReLU", {}, "block1_relu1");
      const b1_conv2 = makeNode("Conv2d", { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1, bias: false }, "block1_conv2");
      const b1_bn2 = makeNode("BatchNorm2d", { num_features: 64 }, "block1_bn2");
      const b1_add = makeNode("Add", {}, "block1_skip");
      const b1_relu2 = makeNode("ReLU", {}, "block1_relu2");
      // Block 2 (64→128, with downsample projection)
      const b2_conv1 = makeNode("Conv2d", { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 2, padding: 1, bias: false }, "block2_conv1");
      const b2_bn1 = makeNode("BatchNorm2d", { num_features: 128 }, "block2_bn1");
      const b2_relu1 = makeNode("ReLU", {}, "block2_relu1");
      const b2_conv2 = makeNode("Conv2d", { in_channels: 128, out_channels: 128, kernel_size: 3, padding: 1, bias: false }, "block2_conv2");
      const b2_bn2 = makeNode("BatchNorm2d", { num_features: 128 }, "block2_bn2");
      // Downsample projection for block 2
      const ds_conv = makeNode("Conv2d", { in_channels: 64, out_channels: 128, kernel_size: 1, stride: 2, padding: 0, bias: false }, "downsample_conv");
      const ds_bn = makeNode("BatchNorm2d", { num_features: 128 }, "downsample_bn");
      const b2_add = makeNode("Add", {}, "block2_skip");
      const b2_relu2 = makeNode("ReLU", {}, "block2_relu2");
      // Classifier head
      const gap = makeNode("AdaptiveAvgPool2d", { output_size: [1, 1] }, "global_pool");
      const flat = makeNode("Flatten", { start_dim: 1, end_dim: -1 });
      const fc = makeNode("Linear", { in_features: 128, out_features: 10 }, "fc");
      const out = makeNode("CrossEntropyLoss");

      const nodes = [
        stem_conv, stem_bn, stem_relu, stem_pool,
        b1_conv1, b1_bn1, b1_relu1, b1_conv2, b1_bn2, b1_add, b1_relu2,
        b2_conv1, b2_bn1, b2_relu1, b2_conv2, b2_bn2,
        ds_conv, ds_bn, b2_add, b2_relu2,
        gap, flat, fc, out,
      ];
      const edges: Edge[] = [
        ...chain([stem_conv, stem_bn, stem_relu, stem_pool]),
        // Block 1: main path
        ...chain([stem_pool, b1_conv1, b1_bn1, b1_relu1, b1_conv2, b1_bn2]),
        // Block 1: skip
        { id: "skip1a", source: b1_bn2.id, target: b1_add.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "skip1b", source: stem_pool.id, target: b1_add.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        ...chain([b1_add, b1_relu2]),
        // Block 2: main path
        ...chain([b1_relu2, b2_conv1, b2_bn1, b2_relu1, b2_conv2, b2_bn2]),
        // Block 2: downsample projection
        ...chain([b1_relu2, ds_conv, ds_bn]),
        // Block 2: skip (with projection)
        { id: "skip2a", source: b2_bn2.id, target: b2_add.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "skip2b", source: ds_bn.id, target: b2_add.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        ...chain([b2_add, b2_relu2, gap, flat, fc, out]),
      ];
      return { nodes, edges };
    },
  },
  {
    name: "Vision Transformer (ViT)",
    description: "Patch embedding via Conv2d → Transformer encoder stack → classifier",
    category: "vision",
    year: 2020,
    paper: "Dosovitskiy et al., 2020",
    trainingDefaults: { optimizer: { type: "AdamW", lr: 1e-3, weightDecay: 0.01 }, loss: "CrossEntropyLoss", epochs: 50, batchSize: 128, dataset: { type: "CIFAR10", root: "./data" } },
    build() {
      _id = 0;
      const nodes = [
        makeNode("Conv2d", { in_channels: 3, out_channels: 384, kernel_size: 16, stride: 16, padding: 0 }, "patch_embed"),
        makeNode("Flatten", { start_dim: 2, end_dim: -1 }, "flatten"),
        makeNode("Permute", { dims: "[0, 2, 1]" }, "permute"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.1, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_1"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.1, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_2"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.1, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_3"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.1, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_4"),
        makeNode("LayerNorm", { normalized_shape: "[384]" }, "final_norm"),
        makeNode("Permute", { dims: "[0, 2, 1]" }, "to_pool"),
        makeNode("AdaptiveAvgPool1d", { output_size: 1 }, "pool"),
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten2"),
        makeNode("Linear", { in_features: 384, out_features: 10 }, "head"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },
  {
    name: "MobileNetV2 Block",
    description: "Inverted residual blocks — expand → depthwise → project + skip connection",
    category: "vision",
    year: 2018,
    paper: "Sandler et al., 2018",
    trainingDefaults: { optimizer: { type: "SGD", lr: 0.01, weightDecay: 0, momentum: 0.9 }, loss: "CrossEntropyLoss", epochs: 50, batchSize: 64, dataset: { type: "CIFAR10", root: "./data" } },
    build() {
      _id = 0;
      // Initial conv
      const init_conv = makeNode("Conv2d", { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 2, padding: 1, bias: false }, "init_conv");
      const init_bn = makeNode("BatchNorm2d", { num_features: 32 }, "init_bn");
      const init_relu = makeNode("ReLU6", {}, "init_relu6");
      // Block 1: expand(32→192) → depthwise(192, groups=192) → project(192→32) + skip
      const b1_expand = makeNode("Conv2d", { in_channels: 32, out_channels: 192, kernel_size: 1, bias: false }, "b1_expand");
      const b1_expand_bn = makeNode("BatchNorm2d", { num_features: 192 }, "b1_expand_bn");
      const b1_expand_relu = makeNode("ReLU6", {}, "b1_expand_relu6");
      const b1_dw = makeNode("Conv2d", { in_channels: 192, out_channels: 192, kernel_size: 3, padding: 1, groups: 192, bias: false }, "b1_depthwise");
      const b1_dw_bn = makeNode("BatchNorm2d", { num_features: 192 }, "b1_dw_bn");
      const b1_dw_relu = makeNode("ReLU6", {}, "b1_dw_relu6");
      const b1_proj = makeNode("Conv2d", { in_channels: 192, out_channels: 32, kernel_size: 1, bias: false }, "b1_project");
      const b1_proj_bn = makeNode("BatchNorm2d", { num_features: 32 }, "b1_proj_bn");
      const b1_add = makeNode("Add", {}, "b1_skip");
      // Block 2: expand(32→192) → depthwise(192, groups=192, stride=2) → project(192→64) (no skip, dim change)
      const b2_expand = makeNode("Conv2d", { in_channels: 32, out_channels: 192, kernel_size: 1, bias: false }, "b2_expand");
      const b2_expand_bn = makeNode("BatchNorm2d", { num_features: 192 }, "b2_expand_bn");
      const b2_expand_relu = makeNode("ReLU6", {}, "b2_expand_relu6");
      const b2_dw = makeNode("Conv2d", { in_channels: 192, out_channels: 192, kernel_size: 3, stride: 2, padding: 1, groups: 192, bias: false }, "b2_depthwise");
      const b2_dw_bn = makeNode("BatchNorm2d", { num_features: 192 }, "b2_dw_bn");
      const b2_dw_relu = makeNode("ReLU6", {}, "b2_dw_relu6");
      const b2_proj = makeNode("Conv2d", { in_channels: 192, out_channels: 64, kernel_size: 1, bias: false }, "b2_project");
      const b2_proj_bn = makeNode("BatchNorm2d", { num_features: 64 }, "b2_proj_bn");
      // Classifier
      const gap = makeNode("AdaptiveAvgPool2d", { output_size: [1, 1] }, "gap");
      const flat = makeNode("Flatten", { start_dim: 1, end_dim: -1 });
      const fc = makeNode("Linear", { in_features: 64, out_features: 10 }, "fc");
      const out = makeNode("CrossEntropyLoss");

      const nodes = [
        init_conv, init_bn, init_relu,
        b1_expand, b1_expand_bn, b1_expand_relu, b1_dw, b1_dw_bn, b1_dw_relu, b1_proj, b1_proj_bn, b1_add,
        b2_expand, b2_expand_bn, b2_expand_relu, b2_dw, b2_dw_bn, b2_dw_relu, b2_proj, b2_proj_bn,
        gap, flat, fc, out,
      ];
      const edges: Edge[] = [
        ...chain([init_conv, init_bn, init_relu]),
        // Block 1 main path
        ...chain([init_relu, b1_expand, b1_expand_bn, b1_expand_relu, b1_dw, b1_dw_bn, b1_dw_relu, b1_proj, b1_proj_bn]),
        // Block 1 skip (same dims → add)
        { id: "mob_skip1a", source: b1_proj_bn.id, target: b1_add.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "mob_skip1b", source: init_relu.id, target: b1_add.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        // Block 2 (no skip — dimension change)
        ...chain([b1_add, b2_expand, b2_expand_bn, b2_expand_relu, b2_dw, b2_dw_bn, b2_dw_relu, b2_proj, b2_proj_bn]),
        // Classifier
        ...chain([b2_proj_bn, gap, flat, fc, out]),
      ];
      return { nodes, edges };
    },
  },
  {
    name: "U-Net",
    description: "Encoder-decoder with skip connections for image segmentation",
    category: "vision",
    year: 2015,
    paper: "Ronneberger et al., 2015",
    trainingDefaults: { optimizer: { type: "Adam", lr: 1e-4, weightDecay: 0 }, loss: "BCEWithLogitsLoss", epochs: 50, batchSize: 8, dataset: { type: "ImageFolder", root: "./data" } },
    build() {
      _id = 0;
      // Encoder
      const e1_conv = makeNode("Conv2d", { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1 }, "enc1_conv");
      const e1_relu = makeNode("ReLU", {}, "enc1_relu");
      const e1_pool = makeNode("MaxPool2d", { kernel_size: 2 }, "enc1_pool");

      const e2_conv = makeNode("Conv2d", { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1 }, "enc2_conv");
      const e2_relu = makeNode("ReLU", {}, "enc2_relu");
      const e2_pool = makeNode("MaxPool2d", { kernel_size: 2 }, "enc2_pool");

      // Bridge
      const bridge = makeNode("Conv2d", { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1 }, "bridge");
      const bridge_relu = makeNode("ReLU", {}, "bridge_relu");

      // Decoder
      const up2 = makeNode("ConvTranspose2d", { in_channels: 256, out_channels: 128, kernel_size: 2, stride: 2 }, "up2");
      const cat2 = makeNode("Concatenate", { dim: 1 }, "skip2");
      const d2_conv = makeNode("Conv2d", { in_channels: 256, out_channels: 128, kernel_size: 3, padding: 1 }, "dec2_conv");
      const d2_relu = makeNode("ReLU", {}, "dec2_relu");

      const up1 = makeNode("ConvTranspose2d", { in_channels: 128, out_channels: 64, kernel_size: 2, stride: 2 }, "up1");
      const cat1 = makeNode("Concatenate", { dim: 1 }, "skip1");
      const d1_conv = makeNode("Conv2d", { in_channels: 128, out_channels: 64, kernel_size: 3, padding: 1 }, "dec1_conv");
      const d1_relu = makeNode("ReLU", {}, "dec1_relu");

      const out_conv = makeNode("Conv2d", { in_channels: 64, out_channels: 1, kernel_size: 1, padding: 0 }, "output_conv");
      const out = makeNode("BCEWithLogitsLoss");

      const nodes = [
        e1_conv, e1_relu, e1_pool,
        e2_conv, e2_relu, e2_pool,
        bridge, bridge_relu,
        up2, cat2, d2_conv, d2_relu,
        up1, cat1, d1_conv, d1_relu,
        out_conv, out,
      ];

      const edges: Edge[] = [
        // Encoder path
        ...chain([e1_conv, e1_relu]),
        ...chain([e1_relu, e1_pool, e2_conv, e2_relu]),
        ...chain([e2_relu, e2_pool, bridge, bridge_relu]),
        // Decoder path
        ...chain([bridge_relu, up2]),
        // Skip connection 2: e2_relu + up2 -> cat2
        { id: `e_skip2a`, source: up2.id, target: cat2.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: `e_skip2b`, source: e2_relu.id, target: cat2.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        ...chain([cat2, d2_conv, d2_relu, up1]),
        // Skip connection 1: e1_relu + up1 -> cat1
        { id: `e_skip1a`, source: up1.id, target: cat1.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: `e_skip1b`, source: e1_relu.id, target: cat1.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        ...chain([cat1, d1_conv, d1_relu, out_conv, out]),
      ];

      return { nodes, edges };
    },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // LANGUAGE
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Transformer",
    description: '"Attention Is All You Need" — encoder-decoder transformer with dual inputs',
    category: "language",
    year: 2017,
    paper: "Vaswani et al., 2017",
    trainingDefaults: { optimizer: { type: "AdamW", lr: 5e-4, weightDecay: 0.01 }, loss: "CrossEntropyLoss", epochs: 20, batchSize: 32, dataset: { type: "WikiText2", root: "./data" } },
    build() {
      _id = 0;
      const src_embed = makeNode("Embedding", { num_embeddings: 30000, embedding_dim: 512 }, "src_embed");
      const tgt_embed = makeNode("Embedding", { num_embeddings: 30000, embedding_dim: 512 }, "tgt_embed");
      const transformer = makeNode("Transformer", {
        d_model: 512, nhead: 8, num_encoder_layers: 6, num_decoder_layers: 6,
        dim_feedforward: 2048, dropout: 0.1, batch_first: true,
      }, "transformer");
      const head = makeNode("Linear", { in_features: 512, out_features: 30000 }, "lm_head");
      const out = makeNode("CrossEntropyLoss");

      const nodes = [src_embed, tgt_embed, transformer, head, out];
      const edges: Edge[] = [
        // src path
        { id: "e_src_tf", source: src_embed.id, target: transformer.id, sourceHandle: "out", targetHandle: "src", type: "smoothstep" },
        // tgt path
        { id: "e_tgt_tf", source: tgt_embed.id, target: transformer.id, sourceHandle: "out", targetHandle: "tgt", type: "smoothstep" },
        // output
        ...chain([transformer, head, out]),
      ];
      return { nodes, edges };
    },
  },
  {
    name: "nanoGPT",
    description: "Karpathy's char-level GPT — 6 decoder layers, 384 dim, 6 heads, vocab 65",
    category: "language",
    year: 2022,
    paper: "Karpathy, 2022",
    trainingDefaults: { optimizer: { type: "AdamW", lr: 3e-4, weightDecay: 0.01 }, loss: "CrossEntropyLoss", epochs: 20, batchSize: 64, dataset: { type: "TinyShakespeare", root: "./data" } },
    build() {
      _id = 0;
      const nodes = [
        makeNode("Embedding", { num_embeddings: 65, embedding_dim: 384 }, "token_embed"),
        makeNode("Dropout", { p: 0.2 }, "embed_drop"),
        makeNode("TransformerDecoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.2, batch_first: true, norm_first: true, activation: "gelu" }, "block_1"),
        makeNode("TransformerDecoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.2, batch_first: true, norm_first: true, activation: "gelu" }, "block_2"),
        makeNode("TransformerDecoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.2, batch_first: true, norm_first: true, activation: "gelu" }, "block_3"),
        makeNode("TransformerDecoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.2, batch_first: true, norm_first: true, activation: "gelu" }, "block_4"),
        makeNode("TransformerDecoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.2, batch_first: true, norm_first: true, activation: "gelu" }, "block_5"),
        makeNode("TransformerDecoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.2, batch_first: true, norm_first: true, activation: "gelu" }, "block_6"),
        makeNode("LayerNorm", { normalized_shape: "[384]" }, "final_ln"),
        makeNode("Linear", { in_features: 384, out_features: 65 }, "lm_head"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },
  {
    name: "BERT Base",
    description: "BERT encoder for text classification — 12 encoder layers, 768 dim, 12 heads",
    category: "language",
    year: 2018,
    paper: "Devlin et al., 2018",
    trainingDefaults: { optimizer: { type: "AdamW", lr: 2e-5, weightDecay: 0.01 }, loss: "CrossEntropyLoss", epochs: 3, batchSize: 16, dataset: { type: "AGNews", root: "./data" } },
    build() {
      _id = 0;
      // Encoder output [B,T,768] → pool to [B,768] → classify
      const nodes = [
        makeNode("Embedding", { num_embeddings: 30522, embedding_dim: 768 }, "token_embed"),
        makeNode("LayerNorm", { normalized_shape: "[768]" }, "embed_ln"),
        makeNode("Dropout", { p: 0.1 }, "embed_drop"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_1"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_2"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_3"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_4"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_5"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_6"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_7"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_8"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_9"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_10"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_11"),
        makeNode("TransformerEncoderLayer", { d_model: 768, nhead: 12, dim_feedforward: 3072, dropout: 0.1, batch_first: true, norm_first: false, activation: "gelu" }, "encoder_12"),
        makeNode("Permute", { dims: "[0, 2, 1]" }, "to_pool"),
        makeNode("AdaptiveAvgPool1d", { output_size: 1 }, "pool"),
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten"),
        makeNode("Linear", { in_features: 768, out_features: 4 }, "classifier"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },
  {
    name: "LSTM Text Classifier",
    description: "Embedding → bidirectional LSTM → pool → linear head for sentiment analysis",
    category: "language",
    paper: undefined,
    trainingDefaults: { optimizer: { type: "Adam", lr: 1e-3, weightDecay: 0 }, loss: "CrossEntropyLoss", epochs: 10, batchSize: 32, dataset: { type: "IMDB", root: "./data" } },
    build() {
      _id = 0;
      // LSTM output is [B, T, 512] — permute+pool to get [B, 512] for classification
      const nodes = [
        makeNode("Embedding", { num_embeddings: 25000, embedding_dim: 128 }, "embed"),
        makeNode("LSTM", { input_size: 128, hidden_size: 256, num_layers: 2, dropout: 0.3, batch_first: true, bidirectional: true }, "lstm"),
        makeNode("Permute", { dims: "[0, 2, 1]" }, "permute"),
        makeNode("AdaptiveAvgPool1d", { output_size: 1 }, "pool"),
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten"),
        makeNode("Dropout", { p: 0.5 }),
        makeNode("Linear", { in_features: 512, out_features: 2 }, "classifier"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // AUDIO
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Whisper Encoder",
    description: "Whisper-style Conv1d stem + transformer encoder + classifier for keyword spotting",
    category: "audio",
    year: 2022,
    paper: "Radford et al., 2022",
    trainingDefaults: { optimizer: { type: "AdamW", lr: 1e-3, weightDecay: 0.01 }, loss: "CrossEntropyLoss", epochs: 30, batchSize: 32, dataset: { type: "SpeechCommands", root: "./data" } },
    build() {
      _id = 0;
      // Input: mel spectrogram [B, 80, 100] from SpeechCommands
      const nodes = [
        makeNode("Conv1d", { in_channels: 80, out_channels: 384, kernel_size: 3, padding: 1, bias: true }, "conv1"),
        makeNode("GELU", {}, "gelu1"),
        makeNode("Conv1d", { in_channels: 384, out_channels: 384, kernel_size: 3, stride: 2, padding: 1, bias: true }, "conv2"),
        makeNode("GELU", {}, "gelu2"),
        makeNode("Permute", { dims: "[0, 2, 1]" }, "permute"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.0, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_1"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.0, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_2"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.0, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_3"),
        makeNode("TransformerEncoderLayer", { d_model: 384, nhead: 6, dim_feedforward: 1536, dropout: 0.0, batch_first: true, norm_first: true, activation: "gelu" }, "encoder_4"),
        makeNode("LayerNorm", { normalized_shape: "[384]" }, "final_ln"),
        makeNode("Permute", { dims: "[0, 2, 1]" }, "to_pool"),
        makeNode("AdaptiveAvgPool1d", { output_size: 1 }, "pool"),
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten"),
        makeNode("Linear", { in_features: 384, out_features: 35 }, "classifier"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },
  {
    name: "WaveNet Block",
    description: "Dilated convolutions (d=1,2,4) with residual connections for keyword spotting",
    category: "audio",
    year: 2016,
    paper: "van den Oord et al., 2016",
    trainingDefaults: { optimizer: { type: "Adam", lr: 1e-3, weightDecay: 0 }, loss: "CrossEntropyLoss", epochs: 30, batchSize: 32, dataset: { type: "SpeechCommands", root: "./data" } },
    build() {
      _id = 0;
      // Adapted for SpeechCommands mel spectrograms [B, 80, 100]
      // Initial conv
      const init_conv = makeNode("Conv1d", { in_channels: 80, out_channels: 64, kernel_size: 1, bias: true }, "init_conv");

      // Dilated block 1 (dilation=1)
      const d1_conv = makeNode("Conv1d", { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1, dilation: 1, bias: true }, "dilated_1");
      const d1_tanh = makeNode("Tanh", {}, "tanh_1");
      const d1_1x1 = makeNode("Conv1d", { in_channels: 64, out_channels: 64, kernel_size: 1, bias: true }, "1x1_1");
      const d1_add = makeNode("Add", {}, "residual_1");

      // Dilated block 2 (dilation=2)
      const d2_conv = makeNode("Conv1d", { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 2, dilation: 2, bias: true }, "dilated_2");
      const d2_tanh = makeNode("Tanh", {}, "tanh_2");
      const d2_1x1 = makeNode("Conv1d", { in_channels: 64, out_channels: 64, kernel_size: 1, bias: true }, "1x1_2");
      const d2_add = makeNode("Add", {}, "residual_2");

      // Dilated block 3 (dilation=4)
      const d3_conv = makeNode("Conv1d", { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 4, dilation: 4, bias: true }, "dilated_4");
      const d3_tanh = makeNode("Tanh", {}, "tanh_3");
      const d3_1x1 = makeNode("Conv1d", { in_channels: 64, out_channels: 64, kernel_size: 1, bias: true }, "1x1_3");
      const d3_add = makeNode("Add", {}, "residual_3");

      // Classifier head (Conv1d output is already [B, C, L], no permute needed)
      const relu_out = makeNode("ReLU", {}, "relu_out");
      const pool = makeNode("AdaptiveAvgPool1d", { output_size: 1 }, "pool");
      const flat = makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten");
      const fc = makeNode("Linear", { in_features: 64, out_features: 35 }, "classifier");
      const out = makeNode("CrossEntropyLoss");

      const nodes = [
        init_conv,
        d1_conv, d1_tanh, d1_1x1, d1_add,
        d2_conv, d2_tanh, d2_1x1, d2_add,
        d3_conv, d3_tanh, d3_1x1, d3_add,
        relu_out, pool, flat, fc, out,
      ];

      const edges: Edge[] = [
        // Block 1
        ...chain([init_conv, d1_conv, d1_tanh, d1_1x1]),
        { id: "wn_r1a", source: d1_1x1.id, target: d1_add.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "wn_r1b", source: init_conv.id, target: d1_add.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        // Block 2
        ...chain([d1_add, d2_conv, d2_tanh, d2_1x1]),
        { id: "wn_r2a", source: d2_1x1.id, target: d2_add.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "wn_r2b", source: d1_add.id, target: d2_add.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        // Block 3
        ...chain([d2_add, d3_conv, d3_tanh, d3_1x1]),
        { id: "wn_r3a", source: d3_1x1.id, target: d3_add.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "wn_r3b", source: d2_add.id, target: d3_add.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        // Classifier
        ...chain([d3_add, relu_out, pool, flat, fc, out]),
      ];

      return { nodes, edges };
    },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // GENERATIVE
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Autoencoder",
    description: "Symmetric encoder-decoder with bottleneck for representation learning",
    category: "generative",
    supervised: false,
    trainingDefaults: { optimizer: { type: "Adam", lr: 1e-3, weightDecay: 0 }, loss: "MSELoss", epochs: 20, batchSize: 128, dataset: { type: "MNIST", root: "./data" } },
    build() {
      _id = 0;
      const nodes = [
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten"),
        makeNode("Linear", { in_features: 784, out_features: 256 }, "enc1"),
        makeNode("ReLU", {}, "enc_relu1"),
        makeNode("Linear", { in_features: 256, out_features: 64 }, "enc2"),
        makeNode("ReLU", {}, "enc_relu2"),
        makeNode("Linear", { in_features: 64, out_features: 16 }, "bottleneck"),
        makeNode("ReLU", {}, "bn_relu"),
        makeNode("Linear", { in_features: 16, out_features: 64 }, "dec1"),
        makeNode("ReLU", {}, "dec_relu1"),
        makeNode("Linear", { in_features: 64, out_features: 256 }, "dec2"),
        makeNode("ReLU", {}, "dec_relu2"),
        makeNode("Linear", { in_features: 256, out_features: 784 }, "dec3"),
        makeNode("Sigmoid", {}, "output_act"),
        makeNode("MSELoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },
  {
    name: "Conv Autoencoder",
    description: "Convolutional encoder-decoder for unsupervised image reconstruction (MNIST)",
    category: "generative",
    supervised: false,
    trainingDefaults: { optimizer: { type: "Adam", lr: 1e-3, weightDecay: 0 }, loss: "MSELoss", epochs: 20, batchSize: 128, dataset: { type: "MNIST", root: "./data" } },
    build() {
      _id = 0;
      const nodes = [
        // Encoder
        makeNode("Conv2d", { in_channels: 1, out_channels: 16, kernel_size: 3, padding: 1 }, "enc_conv1"),
        makeNode("ReLU", {}, "enc_relu1"),
        makeNode("MaxPool2d", { kernel_size: 2, stride: 2 }, "enc_pool1"),
        makeNode("Conv2d", { in_channels: 16, out_channels: 32, kernel_size: 3, padding: 1 }, "enc_conv2"),
        makeNode("ReLU", {}, "enc_relu2"),
        makeNode("MaxPool2d", { kernel_size: 2, stride: 2 }, "enc_pool2"),
        // Decoder
        makeNode("ConvTranspose2d", { in_channels: 32, out_channels: 16, kernel_size: 2, stride: 2 }, "dec_up1"),
        makeNode("ReLU", {}, "dec_relu1"),
        makeNode("ConvTranspose2d", { in_channels: 16, out_channels: 1, kernel_size: 2, stride: 2 }, "dec_up2"),
        makeNode("Sigmoid", {}, "output_act"),
        makeNode("MSELoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },
  {
    name: "Embedding Classifier",
    description: "Conv feature extractor with learned embedding and classification head",
    category: "generative",
    trainingDefaults: { optimizer: { type: "Adam", lr: 1e-3, weightDecay: 0 }, loss: "CrossEntropyLoss", epochs: 20, batchSize: 64, dataset: { type: "MNIST", root: "./data" } },
    build() {
      _id = 0;
      const nodes = [
        makeNode("Conv2d", { in_channels: 1, out_channels: 32, kernel_size: 3, padding: 1 }, "conv1"),
        makeNode("ReLU", {}, "relu1"),
        makeNode("MaxPool2d", { kernel_size: 2, stride: 2 }, "pool1"),
        makeNode("Conv2d", { in_channels: 32, out_channels: 64, kernel_size: 3, padding: 1 }, "conv2"),
        makeNode("ReLU", {}, "relu2"),
        makeNode("MaxPool2d", { kernel_size: 2, stride: 2 }, "pool2"),
        makeNode("Flatten", { start_dim: 1, end_dim: -1 }, "flatten"),
        makeNode("Linear", { in_features: 64 * 7 * 7, out_features: 128 }, "embedding"),
        makeNode("ReLU", {}, "embed_act"),
        makeNode("Linear", { in_features: 128, out_features: 10 }, "classifier"),
        makeNode("CrossEntropyLoss"),
      ];
      return { nodes, edges: chain(nodes) };
    },
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // OTHER
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "DenseNet Block",
    description: "Dense connectivity — each layer receives all previous features via concatenation",
    category: "other",
    year: 2017,
    paper: "Huang et al., 2017",
    trainingDefaults: { optimizer: { type: "SGD", lr: 0.1, weightDecay: 0, momentum: 0.9 }, loss: "CrossEntropyLoss", epochs: 50, batchSize: 64, dataset: { type: "CIFAR10", root: "./data" } },
    build() {
      _id = 0;
      const init_conv = makeNode("Conv2d", { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1 }, "init_conv");
      const init_bn = makeNode("BatchNorm2d", { num_features: 64 }, "init_bn");
      const init_relu = makeNode("ReLU", {}, "init_relu");
      // Dense layer 1
      const d1 = makeNode("Conv2d", { in_channels: 64, out_channels: 32, kernel_size: 3, padding: 1 }, "dense1");
      const d1_relu = makeNode("ReLU", {}, "dense1_relu");
      const cat1 = makeNode("Concatenate", { dim: 1 }, "cat1"); // 64+32=96
      // Dense layer 2
      const d2 = makeNode("Conv2d", { in_channels: 96, out_channels: 32, kernel_size: 3, padding: 1 }, "dense2");
      const d2_relu = makeNode("ReLU", {}, "dense2_relu");
      const cat2 = makeNode("Concatenate", { dim: 1 }, "cat2"); // 96+32=128
      // Transition
      const trans_conv = makeNode("Conv2d", { in_channels: 128, out_channels: 64, kernel_size: 1 }, "transition");
      const trans_pool = makeNode("AvgPool2d", { kernel_size: 2 }, "transition_pool");
      // Head
      const gap = makeNode("AdaptiveAvgPool2d", { output_size: [1, 1] }, "gap");
      const flat = makeNode("Flatten", { start_dim: 1, end_dim: -1 });
      const fc = makeNode("Linear", { in_features: 64, out_features: 10 }, "fc");
      const out = makeNode("CrossEntropyLoss");

      const nodes = [init_conv, init_bn, init_relu,
        d1, d1_relu, cat1, d2, d2_relu, cat2,
        trans_conv, trans_pool, gap, flat, fc, out];

      const edges: Edge[] = [
        ...chain([init_conv, init_bn, init_relu]),
        ...chain([init_relu, d1, d1_relu]),
        // cat1 = init_relu + d1_relu
        { id: "dc1a", source: init_relu.id, target: cat1.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "dc1b", source: d1_relu.id, target: cat1.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        ...chain([cat1, d2, d2_relu]),
        // cat2 = cat1 + d2_relu
        { id: "dc2a", source: cat1.id, target: cat2.id, sourceHandle: "out", targetHandle: "a", type: "smoothstep" },
        { id: "dc2b", source: d2_relu.id, target: cat2.id, sourceHandle: "out", targetHandle: "b", type: "smoothstep" },
        ...chain([cat2, trans_conv, trans_pool, gap, flat, fc, out]),
      ];
      return { nodes, edges };
    },
  },
];
