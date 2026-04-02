/**
 * Dataset metadata — single source of truth for dataset properties,
 * compatibility checking, and domain inference.
 *
 * The universal "Data" node uses DATA_PRESETS for known datasets and
 * resolveDataNodeShape() for shape inference (including introspection cache).
 */

import type { Node } from "@xyflow/react";
import type { NodeData } from "../store/graph-store";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DataDomain = "vision" | "text" | "audio" | "tabular" | "auto";

export interface DataPreset {
  label: string;
  domain: DataDomain;
  defaults: Record<string, unknown>;
  meta: { sampleShape: number[]; inputShape: number[]; numClasses: number };
  description: string;
  sizeHint: string;
  recommendedLoss: string;
  requiresPip?: string;
}

export interface ProbeResult {
  sample_shape: number[];
  sample_dtype: string;
  num_classes: number;
  label_dtype: string;
}

export interface IntrospectionResult {
  columns: string[];
  column_types: Record<string, string>;
  label_values: string[];
  sample_shape: number[];
  num_classes: number;
  detected_domain: string;
  splits: string[];
  total_samples: number;
}

// ---------------------------------------------------------------------------
// DATA_PRESETS — replaces the old per-dataset registry entries
// ---------------------------------------------------------------------------

export const DATA_PRESETS: Record<string, DataPreset> = {
  CIFAR10: {
    label: "CIFAR-10",
    domain: "vision",
    defaults: { path: "./data", image_size: 32, channels: "rgb", horizontal_flip: true, random_crop: true },
    meta: { sampleShape: [3, 32, 32], inputShape: [1, 3, 32, 32], numClasses: 10 },
    description: "32x32 color images, 10 classes",
    sizeHint: "~170 MB",
    recommendedLoss: "CrossEntropyLoss",
  },
  CIFAR100: {
    label: "CIFAR-100",
    domain: "vision",
    defaults: { path: "./data", image_size: 32, channels: "rgb", horizontal_flip: true, random_crop: true },
    meta: { sampleShape: [3, 32, 32], inputShape: [1, 3, 32, 32], numClasses: 100 },
    description: "32x32 color images, 100 classes",
    sizeHint: "~170 MB",
    recommendedLoss: "CrossEntropyLoss",
  },
  MNIST: {
    label: "MNIST",
    domain: "vision",
    defaults: { path: "./data", image_size: 28, channels: "grayscale", random_crop: true },
    meta: { sampleShape: [1, 28, 28], inputShape: [1, 1, 28, 28], numClasses: 10 },
    description: "28x28 grayscale digits, 10 classes",
    sizeHint: "~12 MB",
    recommendedLoss: "CrossEntropyLoss",
  },
  FashionMNIST: {
    label: "Fashion-MNIST",
    domain: "vision",
    defaults: { path: "./data", image_size: 28, channels: "grayscale", random_crop: true },
    meta: { sampleShape: [1, 28, 28], inputShape: [1, 1, 28, 28], numClasses: 10 },
    description: "28x28 grayscale fashion items, 10 classes",
    sizeHint: "~30 MB",
    recommendedLoss: "CrossEntropyLoss",
  },
  TinyShakespeare: {
    label: "Tiny Shakespeare",
    domain: "text",
    defaults: { path: "./data", max_length: 256 },
    meta: { sampleShape: [256], inputShape: [1, 256], numClasses: 65 },
    description: "Char-level language modeling, vocab=65",
    sizeHint: "~1 MB",
    recommendedLoss: "CrossEntropyLoss",
  },
  WikiText2: {
    label: "WikiText-2",
    domain: "text",
    defaults: { path: "./data", max_length: 128, vocab_size: 30000 },
    meta: { sampleShape: [128], inputShape: [1, 128], numClasses: 30000 },
    description: "Word-level language modeling, vocab=30K",
    sizeHint: "~12 MB",
    recommendedLoss: "CrossEntropyLoss",
    requiresPip: "datasets",
  },
  IMDB: {
    label: "IMDB",
    domain: "text",
    defaults: { path: "./data", max_length: 100, vocab_size: 25000 },
    meta: { sampleShape: [100], inputShape: [1, 100], numClasses: 2 },
    description: "Binary sentiment classification",
    sizeHint: "~80 MB",
    recommendedLoss: "BCEWithLogitsLoss",
    requiresPip: "datasets",
  },
  AGNews: {
    label: "AG News",
    domain: "text",
    defaults: { path: "./data", max_length: 100, vocab_size: 25000 },
    meta: { sampleShape: [100], inputShape: [1, 100], numClasses: 4 },
    description: "4-class news topic classification",
    sizeHint: "~30 MB",
    recommendedLoss: "CrossEntropyLoss",
    requiresPip: "datasets",
  },
  SpeechCommands: {
    label: "Speech Commands",
    domain: "audio",
    defaults: { path: "./data" },
    meta: { sampleShape: [80, 100], inputShape: [1, 80, 100], numClasses: 35 },
    description: "35-class keyword spotting, mel spectrogram",
    sizeHint: "~2 GB",
    recommendedLoss: "CrossEntropyLoss",
    requiresPip: "torchaudio",
  },
};

// ---------------------------------------------------------------------------
// DATASET_METADATA — thin wrapper over DATA_PRESETS for backward compat
// ---------------------------------------------------------------------------

export interface DatasetMeta {
  sampleShape: number[];
  inputShape: number[];
  numClasses: number;
  domain: "vision" | "text" | "audio" | "tabular";
  description: string;
  sizeHint: string;
  recommendedLoss: string;
  requiresPip?: string;
}

export const DATASET_METADATA: Record<string, DatasetMeta> = Object.fromEntries(
  Object.entries(DATA_PRESETS).map(([k, p]) => [
    k,
    {
      sampleShape: p.meta.sampleShape,
      inputShape: p.meta.inputShape,
      numClasses: p.meta.numClasses,
      domain: p.domain as "vision" | "text" | "audio" | "tabular",
      description: p.description,
      sizeHint: p.sizeHint,
      recommendedLoss: p.recommendedLoss,
      requiresPip: p.requiresPip,
    },
  ]),
);

// Add non-preset entries that have dynamic shapes
for (const [key, meta] of Object.entries({
  ImageFolder: { sampleShape: [3, 224, 224], inputShape: [1, 3, 224, 224], numClasses: 0, domain: "vision" as const, description: "Custom image folder (train/class/img.jpg)", sizeHint: "varies", recommendedLoss: "CrossEntropyLoss" },
  CustomCSV: { sampleShape: [0], inputShape: [1, 0], numClasses: 0, domain: "tabular" as const, description: "Custom CSV file (configurable columns)", sizeHint: "varies", recommendedLoss: "CrossEntropyLoss" },
  HuggingFace: { sampleShape: [0], inputShape: [1, 0], numClasses: 0, domain: "tabular" as const, description: "Any HuggingFace dataset", sizeHint: "varies", recommendedLoss: "CrossEntropyLoss", requiresPip: "datasets" },
  TensorFile: { sampleShape: [0], inputShape: [1, 0], numClasses: 0, domain: "tabular" as const, description: "Pre-saved tensor file (.pt/.npy/.npz)", sizeHint: "varies", recommendedLoss: "CrossEntropyLoss" },
})) {
  DATASET_METADATA[key] = meta;
}

// ---------------------------------------------------------------------------
// Legacy dataset types — old project files use these as moduleType
// ---------------------------------------------------------------------------

export const LEGACY_DATASET_TYPES = new Set([
  "CIFAR10", "CIFAR100", "MNIST", "FashionMNIST",
  "TinyShakespeare", "WikiText2", "IMDB", "AGNews",
  "SpeechCommands", "ImageFolder", "CustomCSV", "HuggingFace", "TensorFile",
]);

// ---------------------------------------------------------------------------
// isDatasetNode — matches both "Data" and legacy types
// ---------------------------------------------------------------------------

/** All module types that are dataset nodes */
export const DATASET_MODULE_TYPES = new Set([
  "Data",
  ...LEGACY_DATASET_TYPES,
]);

export function isDatasetNode(moduleType: string): boolean {
  return DATASET_MODULE_TYPES.has(moduleType);
}

// ---------------------------------------------------------------------------
// resolveDataNodeShape — shape inference for universal Data nodes
// ---------------------------------------------------------------------------

/**
 * Resolve the output shape for a Data node based on its params.
 * For presets, uses known metadata. For introspected data, uses cached shape.
 * Returns null if shape cannot be determined.
 */
export function resolveDataNodeShape(params: Record<string, unknown>): {
  inputShape: number[];
  numClasses: number;
} | null {
  const source = String(params.source ?? "");

  // 1. Probed — most authoritative (actual pipeline output)
  const probed = params._probed as ProbeResult | undefined;
  if (probed && probed.sample_shape.length > 0) {
    return {
      inputShape: [1, ...probed.sample_shape],
      numClasses: probed.num_classes,
    };
  }

  // 1b. Scan result — for TensorFile, use first tensor's shape
  const scan = params._scan as { tensors?: { name: string; shape: number[] }[] } | undefined;
  if (scan && scan.tensors && scan.tensors.length > 0) {
    const firstTensor = scan.tensors[0];
    // shape includes batch dim from scan, so use as-is for inputShape
    // but convention is [1, ...sample_shape]
    const sampleShape = firstTensor.shape.slice(1);
    return {
      inputShape: [1, ...sampleShape],
      numClasses: 0,
    };
  }

  // 2. Check if it's a preset
  const preset = DATA_PRESETS[source];
  if (preset) {
    // For vision presets with overridable image_size
    if (preset.domain === "vision" && params.image_size != null) {
      const imgSize = Number(params.image_size);
      const channels = params.channels === "grayscale" ? 1 : preset.meta.sampleShape[0];
      if (imgSize !== preset.meta.sampleShape[1]) {
        return {
          inputShape: [1, channels, imgSize, imgSize],
          numClasses: preset.meta.numClasses,
        };
      }
    }
    // For text presets with overridable max_length
    if (preset.domain === "text" && params.max_length != null) {
      const maxLen = Number(params.max_length);
      if (maxLen !== preset.meta.sampleShape[0]) {
        return {
          inputShape: [1, maxLen],
          numClasses: preset.meta.numClasses,
        };
      }
    }
    return {
      inputShape: preset.meta.inputShape,
      numClasses: preset.meta.numClasses,
    };
  }

  // Check introspection cache
  const introspected = params._introspected as IntrospectionResult | undefined;
  if (introspected && introspected.sample_shape.length > 0) {
    return {
      inputShape: [1, ...introspected.sample_shape],
      numClasses: introspected.num_classes,
    };
  }

  // Custom sources with user-set params
  if (source === "ImageFolder") {
    const imgSize = Number(params.image_size ?? 224);
    const channels = params.channels === "grayscale" ? 1 : 3;
    return { inputShape: [1, channels, imgSize, imgSize], numClasses: 0 };
  }

  if (source === "HuggingFace" && params.domain === "vision") {
    const imgSize = Number(params.image_size ?? 32);
    return { inputShape: [1, 3, imgSize, imgSize], numClasses: 0 };
  }

  if (source === "HuggingFace" && params.domain === "text") {
    const maxLen = Number(params.max_length ?? 128);
    return { inputShape: [1, maxLen], numClasses: 0 };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Compatibility checking
// ---------------------------------------------------------------------------

export type CompatibilityStatus = "match" | "fixable" | "mismatch" | "unknown";

export interface CompatibilityResult {
  status: CompatibilityStatus;
  message: string;
  suggestedInputShape?: number[];
  suggestedOutputFeatures?: number;
}

export function checkDatasetCompatibility(
  datasetType: string,
  inputNodeShape: number[] | null,
  outputFeatures: number | null,
): CompatibilityResult {
  const meta = DATASET_METADATA[datasetType];
  if (!meta) {
    return { status: "unknown", message: "Custom dataset \u2014 shape verified at training time" };
  }

  if (!inputNodeShape || inputNodeShape.length === 0) {
    return {
      status: "fixable",
      message: `${datasetType} expects input ${JSON.stringify(meta.inputShape)}`,
      suggestedInputShape: meta.inputShape,
    };
  }

  const shapeMatch = JSON.stringify(inputNodeShape) === JSON.stringify(meta.inputShape);
  const flatExpected = meta.inputShape.slice(1).reduce((a, b) => a * b, 1);
  const flatCurrent = inputNodeShape.slice(1).reduce((a, b) => a * b, 1);
  const flatMatch = flatExpected === flatCurrent;

  if (!shapeMatch && !flatMatch) {
    return {
      status: "fixable",
      message: `${datasetType} expects ${JSON.stringify(meta.inputShape)} but Input is ${JSON.stringify(inputNodeShape)}`,
      suggestedInputShape: meta.inputShape,
    };
  }

  // Check output features vs num_classes
  if (meta.numClasses > 0 && outputFeatures != null && outputFeatures > 0 && outputFeatures !== meta.numClasses) {
    return {
      status: "mismatch",
      message: `${datasetType} has ${meta.numClasses} classes but final Linear outputs ${outputFeatures}`,
      suggestedOutputFeatures: meta.numClasses,
    };
  }

  const shapeStr = JSON.stringify(meta.inputShape);
  return {
    status: "match",
    message: `${datasetType} \u2192 ${shapeStr}, ${meta.numClasses} classes \u2014 matches your Input node`,
  };
}

// ---------------------------------------------------------------------------
// Model domain inference
// ---------------------------------------------------------------------------

export type ModelDomain = "vision" | "text" | "audio" | "general";

const VISION_MODULES = new Set(["Conv2d", "Conv3d", "ConvTranspose2d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "BatchNorm2d"]);
const TEXT_MODULES = new Set(["Embedding", "LSTM", "GRU", "TransformerEncoderLayer", "TransformerDecoderLayer", "MultiheadAttention"]);
const AUDIO_MODULES = new Set(["Conv1d", "ConvTranspose1d", "BatchNorm1d"]);

export function inferModelDomain(nodes: Node<NodeData>[]): ModelDomain {
  let vision = 0;
  let text = 0;
  let audio = 0;

  for (const node of nodes) {
    const mt = node.data.moduleType;
    if (VISION_MODULES.has(mt)) vision++;
    if (TEXT_MODULES.has(mt)) text++;
    if (AUDIO_MODULES.has(mt)) audio++;
  }

  if (vision === 0 && text === 0 && audio === 0) return "general";
  if (vision >= text && vision >= audio) return "vision";
  if (text >= vision && text >= audio) return "text";
  return "audio";
}

const DOMAIN_DATASETS: Record<ModelDomain, string[]> = {
  vision: ["CIFAR10", "MNIST", "ImageFolder"],
  text: ["IMDB", "TinyShakespeare", "WikiText2"],
  audio: ["SpeechCommands"],
  general: ["CIFAR10", "MNIST", "TensorFile"],
};

export function getRecommendedDatasets(domain: ModelDomain): string[] {
  return DOMAIN_DATASETS[domain];
}

// ---------------------------------------------------------------------------
// extractDatasetConfig — maps Data node params to sidecar format
// ---------------------------------------------------------------------------

/**
 * Extract training-compatible dataset config from a Dataset node's params.
 * Handles both the universal "Data" node and legacy per-dataset nodes.
 */
export function extractDatasetConfig(moduleType: string, params: Record<string, unknown>): {
  dataset: { type: string; root: string };
  valSplit: number;
  datasetExtra: Record<string, unknown>;
  augmentation: {
    horizontalFlip: boolean;
    verticalFlip: boolean;
    randomCrop: boolean;
    colorJitter: boolean;
    rotation: number;
  };
} {
  // Map source strings to what the sidecar expects
  const SIDECAR_TYPE_MAP: Record<string, string> = {
    CustomCSV: "Custom CSV",
    CSV: "Custom CSV",
  };

  const STRIP_KEYS = new Set([
    "source", "domain", "path", "dataset_name", "val_split",
    "horizontal_flip", "random_crop", "color_jitter", "rotation",
    "_introspected", "_probed", "_scan", "root",
  ]);

  // Universal Data node: derive type/root from source + params
  if (moduleType === "Data") {
    const source = String(params.source ?? "");
    const sidecarType = SIDECAR_TYPE_MAP[source] ?? source;

    // Determine root
    let root: string;
    if (source === "HuggingFace") {
      root = String(params.dataset_name ?? "mnist");
    } else if (["ImageFolder", "CSV", "TensorFile"].includes(source)) {
      root = String(params.path ?? "./data");
    } else {
      // Preset — use path or default
      root = String(params.path ?? DATA_PRESETS[source]?.defaults.path ?? "./data");
    }

    // Collect extra params (everything that's not structural or augmentation)
    const datasetExtra: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(params)) {
      if (!STRIP_KEYS.has(k) && v !== undefined && v !== null && v !== "") {
        datasetExtra[k] = v;
      }
    }

    return {
      dataset: { type: sidecarType, root },
      valSplit: Number(params.val_split ?? 0.2),
      datasetExtra,
      augmentation: {
        horizontalFlip: !!params.horizontal_flip,
        verticalFlip: false,
        randomCrop: !!params.random_crop,
        colorJitter: !!params.color_jitter,
        rotation: Number(params.rotation ?? 0),
      },
    };
  }

  // Legacy nodes: old behavior
  const typeMap: Record<string, string> = { CustomCSV: "Custom CSV" };

  const AUGMENTATION_KEYS = new Set(["root", "val_split", "horizontal_flip", "random_crop", "color_jitter", "rotation"]);
  const datasetExtra: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    if (!AUGMENTATION_KEYS.has(k) && v !== undefined && v !== null && v !== "") {
      datasetExtra[k] = v;
    }
  }

  return {
    dataset: {
      type: typeMap[moduleType] ?? moduleType,
      root: String(params.root ?? "./data"),
    },
    valSplit: Number(params.val_split ?? 0.2),
    datasetExtra,
    augmentation: {
      horizontalFlip: !!params.horizontal_flip,
      verticalFlip: false,
      randomCrop: !!params.random_crop,
      colorJitter: !!params.color_jitter,
      rotation: Number(params.rotation ?? 0),
    },
  };
}
