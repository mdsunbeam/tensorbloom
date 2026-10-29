import type { ModuleDef, NodeCategory } from "./types";

const tensorIn = { id: "in", label: "in", dataType: "tensor" as const };
const tensorOut = { id: "out", label: "out", dataType: "tensor" as const };

export const MODULE_REGISTRY: Record<string, ModuleDef> = {
  // --- I/O ---
  Input: {
    type: "Input",
    label: "Input",
    category: "io",
    description: "Network input tensor",
    params: [
      { name: "shape", type: "string", default: "[1, 3, 224, 224]", label: "Shape" },
      { name: "dtype", type: "select", default: "float32", options: ["float32", "float16", "float64", "int64", "int32"] },
    ],
    inputHandles: [{ id: "in", label: "data", dataType: "tensor" as const }],
    outputHandles: [tensorOut],
  },
  Output: {
    type: "Output",
    label: "Output",
    category: "io",
    description: "Network output",
    params: [],
    inputHandles: [tensorIn],
    outputHandles: [],
  },

  // --- Data (universal) ---
  Data: {
    type: "Data",
    label: "Data",
    category: "data",
    description: "Data source — built-in presets or custom tensor files",
    params: [
      { name: "source", type: "select", default: "", options: [
        "",
        "CIFAR10", "CIFAR100", "MNIST", "FashionMNIST",
        "TinyShakespeare", "WikiText2", "IMDB", "AGNews", "SpeechCommands",
        "TensorFile",
      ], label: "Source" },
      { name: "val_split", type: "float", default: 0.2, label: "Val Split", min: 0, max: 0.5 },
    ],
    inputHandles: [],
    outputHandles: [tensorOut],
  },

  // --- Linear ---
  Linear: {
    type: "Linear",
    label: "Linear",
    category: "linear",
    description: "Fully connected layer: y = xA^T + b",
    params: [
      { name: "in_features", type: "int", default: 512, min: 1 },
      { name: "out_features", type: "int", default: 256, min: 1 },
      { name: "bias", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Identity: {
    type: "Identity",
    label: "Identity",
    category: "linear",
    description: "Pass-through (identity mapping)",
    params: [],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Convolution ---
  Conv1d: {
    type: "Conv1d",
    label: "Conv1d",
    category: "conv",
    description: "1D convolution over temporal data",
    params: [
      { name: "in_channels", type: "int", default: 1, min: 1 },
      { name: "out_channels", type: "int", default: 32, min: 1 },
      { name: "kernel_size", type: "int", default: 3, min: 1 },
      { name: "stride", type: "int", default: 1, min: 1 },
      { name: "padding", type: "int", default: 0, min: 0 },
      { name: "dilation", type: "int", default: 1, min: 1 },
      { name: "groups", type: "int", default: 1, min: 1 },
      { name: "bias", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Conv2d: {
    type: "Conv2d",
    label: "Conv2d",
    category: "conv",
    description: "2D convolution over images",
    params: [
      { name: "in_channels", type: "int", default: 3, min: 1 },
      { name: "out_channels", type: "int", default: 64, min: 1 },
      { name: "kernel_size", type: "int", default: 3, min: 1 },
      { name: "stride", type: "int", default: 1, min: 1 },
      { name: "padding", type: "int", default: 0, min: 0 },
      { name: "dilation", type: "int", default: 1, min: 1 },
      { name: "groups", type: "int", default: 1, min: 1 },
      { name: "bias", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Conv3d: {
    type: "Conv3d",
    label: "Conv3d",
    category: "conv",
    description: "3D convolution over volumes",
    params: [
      { name: "in_channels", type: "int", default: 1, min: 1 },
      { name: "out_channels", type: "int", default: 32, min: 1 },
      { name: "kernel_size", type: "int", default: 3, min: 1 },
      { name: "stride", type: "int", default: 1, min: 1 },
      { name: "padding", type: "int", default: 0, min: 0 },
      { name: "dilation", type: "int", default: 1, min: 1 },
      { name: "groups", type: "int", default: 1, min: 1 },
      { name: "bias", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  ConvTranspose2d: {
    type: "ConvTranspose2d",
    label: "ConvTranspose2d",
    category: "conv",
    description: "2D transposed convolution (upsampling)",
    params: [
      { name: "in_channels", type: "int", default: 64, min: 1 },
      { name: "out_channels", type: "int", default: 32, min: 1 },
      { name: "kernel_size", type: "int", default: 3, min: 1 },
      { name: "stride", type: "int", default: 1, min: 1 },
      { name: "padding", type: "int", default: 0, min: 0 },
      { name: "output_padding", type: "int", default: 0, min: 0 },
      { name: "bias", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Pooling ---
  MaxPool2d: {
    type: "MaxPool2d",
    label: "MaxPool2d",
    category: "pool",
    description: "2D max pooling — downsample by taking maximum",
    params: [
      { name: "kernel_size", type: "int", default: 2, min: 1 },
      { name: "stride", type: "optional_int", default: null },
      { name: "padding", type: "int", default: 0, min: 0 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  AvgPool2d: {
    type: "AvgPool2d",
    label: "AvgPool2d",
    category: "pool",
    description: "2D average pooling",
    params: [
      { name: "kernel_size", type: "int", default: 2, min: 1 },
      { name: "stride", type: "optional_int", default: null },
      { name: "padding", type: "int", default: 0, min: 0 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  AdaptiveAvgPool2d: {
    type: "AdaptiveAvgPool2d",
    label: "AdaptiveAvgPool2d",
    category: "pool",
    description: "Adaptive average pooling to target output size",
    params: [
      { name: "output_size", type: "tuple_int", default: [1, 1] },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Normalization ---
  BatchNorm1d: {
    type: "BatchNorm1d",
    label: "BatchNorm1d",
    category: "norm",
    description: "1D batch normalization",
    params: [
      { name: "num_features", type: "int", default: 64, min: 1 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "momentum", type: "float", default: 0.1 },
      { name: "affine", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  BatchNorm2d: {
    type: "BatchNorm2d",
    label: "BatchNorm2d",
    category: "norm",
    description: "2D batch normalization (after Conv2d)",
    params: [
      { name: "num_features", type: "int", default: 64, min: 1 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "momentum", type: "float", default: 0.1 },
      { name: "affine", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  LayerNorm: {
    type: "LayerNorm",
    label: "LayerNorm",
    category: "norm",
    description: "Layer normalization (transformers)",
    params: [
      { name: "normalized_shape", type: "string", default: "[512]" },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "elementwise_affine", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  GroupNorm: {
    type: "GroupNorm",
    label: "GroupNorm",
    category: "norm",
    description: "Group normalization",
    params: [
      { name: "num_groups", type: "int", default: 32, min: 1 },
      { name: "num_channels", type: "int", default: 64, min: 1 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "affine", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  RMSNorm: {
    type: "RMSNorm",
    label: "RMSNorm",
    category: "norm",
    description: "Root mean square normalization",
    params: [
      { name: "normalized_shape", type: "string", default: "[512]" },
      { name: "eps", type: "float", default: 1e-5 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Activations ---
  ReLU: {
    type: "ReLU",
    label: "ReLU",
    category: "activation",
    description: "Rectified Linear Unit: max(0, x)",
    params: [{ name: "inplace", type: "bool", default: false }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  LeakyReLU: {
    type: "LeakyReLU",
    label: "LeakyReLU",
    category: "activation",
    description: "Leaky ReLU: max(neg_slope*x, x)",
    params: [
      { name: "negative_slope", type: "float", default: 0.01 },
      { name: "inplace", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  GELU: {
    type: "GELU",
    label: "GELU",
    category: "activation",
    description: "Gaussian Error Linear Unit",
    params: [
      { name: "approximate", type: "select", default: "none", options: ["none", "tanh"] },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  SiLU: {
    type: "SiLU",
    label: "SiLU",
    category: "activation",
    description: "Sigmoid Linear Unit (Swish): x * sigmoid(x)",
    params: [{ name: "inplace", type: "bool", default: false }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Mish: {
    type: "Mish",
    label: "Mish",
    category: "activation",
    description: "Mish: x * tanh(softplus(x))",
    params: [{ name: "inplace", type: "bool", default: false }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Sigmoid: {
    type: "Sigmoid",
    label: "Sigmoid",
    category: "activation",
    description: "Sigmoid: 1 / (1 + exp(-x))",
    params: [],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Tanh: {
    type: "Tanh",
    label: "Tanh",
    category: "activation",
    description: "Hyperbolic tangent",
    params: [],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Softmax: {
    type: "Softmax",
    label: "Softmax",
    category: "activation",
    description: "Softmax normalization along a dimension",
    params: [{ name: "dim", type: "int", default: -1 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  PReLU: {
    type: "PReLU",
    label: "PReLU",
    category: "activation",
    description: "Parametric ReLU with learnable slope",
    params: [
      { name: "num_parameters", type: "int", default: 1, min: 1 },
      { name: "init", type: "float", default: 0.25 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  ELU: {
    type: "ELU",
    label: "ELU",
    category: "activation",
    description: "Exponential Linear Unit",
    params: [
      { name: "alpha", type: "float", default: 1.0 },
      { name: "inplace", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Dropout ---
  Dropout: {
    type: "Dropout",
    label: "Dropout",
    category: "dropout",
    description: "Randomly zero elements during training",
    params: [
      { name: "p", type: "float", default: 0.5, min: 0, max: 1 },
      { name: "inplace", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Dropout2d: {
    type: "Dropout2d",
    label: "Dropout2d",
    category: "dropout",
    description: "Zero entire channels randomly",
    params: [
      { name: "p", type: "float", default: 0.5, min: 0, max: 1 },
      { name: "inplace", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Embedding ---
  Embedding: {
    type: "Embedding",
    label: "Embedding",
    category: "embedding",
    description: "Lookup table: index -> dense vector",
    params: [
      { name: "num_embeddings", type: "int", default: 10000, min: 1 },
      { name: "embedding_dim", type: "int", default: 512, min: 1 },
      { name: "padding_idx", type: "optional_int", default: null },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Recurrent ---
  LSTM: {
    type: "LSTM",
    label: "LSTM",
    category: "recurrent",
    description: "Long Short-Term Memory",
    params: [
      { name: "input_size", type: "int", default: 512, min: 1 },
      { name: "hidden_size", type: "int", default: 256, min: 1 },
      { name: "num_layers", type: "int", default: 1, min: 1 },
      { name: "bias", type: "bool", default: true },
      { name: "batch_first", type: "bool", default: true },
      { name: "dropout", type: "float", default: 0, min: 0, max: 1 },
      { name: "bidirectional", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [
      { id: "out", label: "output", dataType: "tensor" },
      { id: "hidden", label: "hidden", dataType: "tuple" },
    ],
  },
  GRU: {
    type: "GRU",
    label: "GRU",
    category: "recurrent",
    description: "Gated Recurrent Unit",
    params: [
      { name: "input_size", type: "int", default: 512, min: 1 },
      { name: "hidden_size", type: "int", default: 256, min: 1 },
      { name: "num_layers", type: "int", default: 1, min: 1 },
      { name: "bias", type: "bool", default: true },
      { name: "batch_first", type: "bool", default: true },
      { name: "dropout", type: "float", default: 0, min: 0, max: 1 },
      { name: "bidirectional", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [
      { id: "out", label: "output", dataType: "tensor" },
      { id: "hidden", label: "hidden", dataType: "tensor" },
    ],
  },

  // --- Transformer ---
  MultiheadAttention: {
    type: "MultiheadAttention",
    label: "MultiheadAttention",
    category: "transformer",
    description: "Multi-head scaled dot-product attention",
    params: [
      { name: "embed_dim", type: "int", default: 512, min: 1 },
      { name: "num_heads", type: "int", default: 8, min: 1 },
      { name: "dropout", type: "float", default: 0.0, min: 0, max: 1 },
      { name: "bias", type: "bool", default: true },
      { name: "batch_first", type: "bool", default: true },
    ],
    inputHandles: [
      { id: "query", label: "Q", dataType: "tensor" },
      { id: "key", label: "K", dataType: "tensor" },
      { id: "value", label: "V", dataType: "tensor" },
    ],
    outputHandles: [
      { id: "out", label: "out", dataType: "tensor" },
      { id: "weights", label: "weights", dataType: "tensor" },
    ],
  },
  TransformerEncoderLayer: {
    type: "TransformerEncoderLayer",
    label: "TransformerEncoderLayer",
    category: "transformer",
    description: "Self-attention + FFN + norms + residuals",
    params: [
      { name: "d_model", type: "int", default: 512, min: 1 },
      { name: "nhead", type: "int", default: 8, min: 1 },
      { name: "dim_feedforward", type: "int", default: 2048, min: 1 },
      { name: "dropout", type: "float", default: 0.1, min: 0, max: 1 },
      { name: "activation", type: "select", default: "relu", options: ["relu", "gelu"] },
      { name: "batch_first", type: "bool", default: true },
      { name: "norm_first", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  TransformerDecoderLayer: {
    type: "TransformerDecoderLayer",
    label: "TransformerDecoderLayer",
    category: "transformer",
    description: "Self-attn + cross-attn + FFN + norms",
    params: [
      { name: "d_model", type: "int", default: 512, min: 1 },
      { name: "nhead", type: "int", default: 8, min: 1 },
      { name: "dim_feedforward", type: "int", default: 2048, min: 1 },
      { name: "dropout", type: "float", default: 0.1, min: 0, max: 1 },
      { name: "activation", type: "select", default: "relu", options: ["relu", "gelu"] },
      { name: "batch_first", type: "bool", default: true },
      { name: "norm_first", type: "bool", default: false },
    ],
    inputHandles: [
      tensorIn,
      { id: "memory", label: "memory", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },

  // --- Loss Functions ---
  CrossEntropyLoss: {
    type: "CrossEntropyLoss",
    label: "CrossEntropyLoss",
    category: "loss",
    description: "Cross entropy for multi-class classification",
    params: [
      { name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum"] },
      { name: "label_smoothing", type: "float", default: 0.0, min: 0, max: 1 },
    ],
    inputHandles: [
      { id: "input", label: "logits", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  MSELoss: {
    type: "MSELoss",
    label: "MSELoss",
    category: "loss",
    description: "Mean Squared Error for regression",
    params: [
      { name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum"] },
    ],
    inputHandles: [
      { id: "input", label: "pred", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  BCEWithLogitsLoss: {
    type: "BCEWithLogitsLoss",
    label: "BCEWithLogitsLoss",
    category: "loss",
    description: "Binary cross entropy with sigmoid",
    params: [
      { name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum"] },
    ],
    inputHandles: [
      { id: "input", label: "logits", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },

  // --- Reshape ---
  Flatten: {
    type: "Flatten",
    label: "Flatten",
    category: "reshape",
    description: "Flatten dimensions into one",
    params: [
      { name: "start_dim", type: "int", default: 1 },
      { name: "end_dim", type: "int", default: -1 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Unflatten: {
    type: "Unflatten",
    label: "Unflatten",
    category: "reshape",
    description: "Expand a dimension into multiple",
    params: [
      { name: "dim", type: "int", default: 1 },
      { name: "unflattened_size", type: "string", default: "[8, 8]" },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Math (functional operations) ---
  Add: {
    type: "Add",
    label: "Add",
    category: "math",
    description: "Element-wise addition (residual connection)",
    params: [],
    inputHandles: [
      { id: "a", label: "A", dataType: "tensor" },
      { id: "b", label: "B", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },
  Concatenate: {
    type: "Concatenate",
    label: "Concat",
    category: "math",
    description: "Concatenate tensors along a dimension",
    params: [{ name: "dim", type: "int", default: 1 }],
    inputHandles: [
      { id: "a", label: "A", dataType: "tensor" },
      { id: "b", label: "B", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },
  MatMul: {
    type: "MatMul",
    label: "MatMul",
    category: "math",
    description: "Matrix multiplication",
    params: [],
    inputHandles: [
      { id: "a", label: "A", dataType: "tensor" },
      { id: "b", label: "B", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },

  // --- Vision ---
  Upsample: {
    type: "Upsample",
    label: "Upsample",
    category: "vision",
    description: "Upsample tensor spatially",
    params: [
      { name: "scale_factor", type: "float", default: 2.0, min: 0.1 },
      { name: "mode", type: "select", default: "nearest", options: ["nearest", "bilinear", "bicubic", "trilinear"] },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  PixelShuffle: {
    type: "PixelShuffle",
    label: "PixelShuffle",
    category: "vision",
    description: "Sub-pixel convolution upsampling",
    params: [{ name: "upscale_factor", type: "int", default: 2, min: 1 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  PixelUnshuffle: {
    type: "PixelUnshuffle",
    label: "PixelUnshuffle",
    category: "vision",
    description: "Inverse of PixelShuffle (downsampling)",
    params: [{ name: "downscale_factor", type: "int", default: 2, min: 1 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Convolutions ---
  ConvTranspose1d: {
    type: "ConvTranspose1d",
    label: "ConvTranspose1d",
    category: "conv",
    description: "1D transposed convolution (upsampling)",
    params: [
      { name: "in_channels", type: "int", default: 64, min: 1 },
      { name: "out_channels", type: "int", default: 32, min: 1 },
      { name: "kernel_size", type: "int", default: 3, min: 1 },
      { name: "stride", type: "int", default: 1, min: 1 },
      { name: "padding", type: "int", default: 0, min: 0 },
      { name: "bias", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Pooling ---
  MaxPool1d: {
    type: "MaxPool1d",
    label: "MaxPool1d",
    category: "pool",
    description: "1D maximum pooling",
    params: [
      { name: "kernel_size", type: "int", default: 2, min: 1 },
      { name: "stride", type: "optional_int", default: null },
      { name: "padding", type: "int", default: 0, min: 0 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  AvgPool1d: {
    type: "AvgPool1d",
    label: "AvgPool1d",
    category: "pool",
    description: "1D average pooling",
    params: [
      { name: "kernel_size", type: "int", default: 2, min: 1 },
      { name: "stride", type: "optional_int", default: null },
      { name: "padding", type: "int", default: 0, min: 0 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  MaxPool3d: {
    type: "MaxPool3d",
    label: "MaxPool3d",
    category: "pool",
    description: "3D maximum pooling",
    params: [
      { name: "kernel_size", type: "int", default: 2, min: 1 },
      { name: "stride", type: "optional_int", default: null },
      { name: "padding", type: "int", default: 0, min: 0 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  AdaptiveMaxPool2d: {
    type: "AdaptiveMaxPool2d",
    label: "AdaptiveMaxPool2d",
    category: "pool",
    description: "Adaptive max pooling to target output size",
    params: [{ name: "output_size", type: "tuple_int", default: [1, 1] }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  AdaptiveAvgPool1d: {
    type: "AdaptiveAvgPool1d",
    label: "AdaptiveAvgPool1d",
    category: "pool",
    description: "Adaptive average pooling for 1D data",
    params: [{ name: "output_size", type: "int", default: 1, min: 1 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Normalization ---
  InstanceNorm1d: {
    type: "InstanceNorm1d",
    label: "InstanceNorm1d",
    category: "norm",
    description: "1D instance normalization (style transfer)",
    params: [
      { name: "num_features", type: "int", default: 64, min: 1 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "affine", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  InstanceNorm2d: {
    type: "InstanceNorm2d",
    label: "InstanceNorm2d",
    category: "norm",
    description: "2D instance normalization (style transfer)",
    params: [
      { name: "num_features", type: "int", default: 64, min: 1 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "affine", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  BatchNorm3d: {
    type: "BatchNorm3d",
    label: "BatchNorm3d",
    category: "norm",
    description: "3D batch normalization",
    params: [
      { name: "num_features", type: "int", default: 64, min: 1 },
      { name: "eps", type: "float", default: 1e-5 },
      { name: "momentum", type: "float", default: 0.1 },
      { name: "affine", type: "bool", default: true },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Activations ---
  SELU: {
    type: "SELU",
    label: "SELU",
    category: "activation",
    description: "Scaled Exponential Linear Unit (self-normalizing)",
    params: [{ name: "inplace", type: "bool", default: false }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  CELU: {
    type: "CELU",
    label: "CELU",
    category: "activation",
    description: "Continuously Differentiable ELU",
    params: [
      { name: "alpha", type: "float", default: 1.0 },
      { name: "inplace", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Hardswish: {
    type: "Hardswish",
    label: "Hardswish",
    category: "activation",
    description: "Efficient piecewise-linear Swish approximation",
    params: [{ name: "inplace", type: "bool", default: false }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Hardsigmoid: {
    type: "Hardsigmoid",
    label: "Hardsigmoid",
    category: "activation",
    description: "Efficient piecewise-linear Sigmoid approximation",
    params: [{ name: "inplace", type: "bool", default: false }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  ReLU6: {
    type: "ReLU6",
    label: "ReLU6",
    category: "activation",
    description: "ReLU capped at 6 (mobile networks)",
    params: [{ name: "inplace", type: "bool", default: false }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  LogSoftmax: {
    type: "LogSoftmax",
    label: "LogSoftmax",
    category: "activation",
    description: "Log of softmax (use with NLLLoss)",
    params: [{ name: "dim", type: "int", default: -1 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Softplus: {
    type: "Softplus",
    label: "Softplus",
    category: "activation",
    description: "Smooth approximation of ReLU: log(1 + exp(x))",
    params: [
      { name: "beta", type: "float", default: 1.0 },
      { name: "threshold", type: "float", default: 20 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  GLU: {
    type: "GLU",
    label: "GLU",
    category: "activation",
    description: "Gated Linear Unit — splits and gates with sigmoid",
    params: [{ name: "dim", type: "int", default: -1 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Dropout ---
  Dropout3d: {
    type: "Dropout3d",
    label: "Dropout3d",
    category: "dropout",
    description: "Zero entire channels for volumetric data",
    params: [
      { name: "p", type: "float", default: 0.5, min: 0, max: 1 },
      { name: "inplace", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  AlphaDropout: {
    type: "AlphaDropout",
    label: "AlphaDropout",
    category: "dropout",
    description: "Dropout preserving self-normalizing property (for SELU)",
    params: [{ name: "p", type: "float", default: 0.5, min: 0, max: 1 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Losses ---
  L1Loss: {
    type: "L1Loss",
    label: "L1Loss",
    category: "loss",
    description: "Mean Absolute Error loss",
    params: [{ name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum"] }],
    inputHandles: [
      { id: "input", label: "pred", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  HuberLoss: {
    type: "HuberLoss",
    label: "HuberLoss",
    category: "loss",
    description: "Robust regression loss (smooth L1)",
    params: [
      { name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum"] },
      { name: "delta", type: "float", default: 1.0 },
    ],
    inputHandles: [
      { id: "input", label: "pred", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  NLLLoss: {
    type: "NLLLoss",
    label: "NLLLoss",
    category: "loss",
    description: "Negative log likelihood (after LogSoftmax)",
    params: [{ name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum"] }],
    inputHandles: [
      { id: "input", label: "log_probs", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  KLDivLoss: {
    type: "KLDivLoss",
    label: "KLDivLoss",
    category: "loss",
    description: "KL divergence (for VAEs, knowledge distillation)",
    params: [
      { name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum", "batchmean"] },
      { name: "log_target", type: "bool", default: false },
    ],
    inputHandles: [
      { id: "input", label: "pred", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  TripletMarginLoss: {
    type: "TripletMarginLoss",
    label: "TripletMarginLoss",
    category: "loss",
    description: "Metric learning with anchor/positive/negative",
    params: [
      { name: "margin", type: "float", default: 1.0 },
      { name: "p", type: "float", default: 2.0 },
    ],
    inputHandles: [
      { id: "anchor", label: "anchor", dataType: "tensor" },
      { id: "positive", label: "positive", dataType: "tensor" },
      { id: "negative", label: "negative", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  CosineEmbeddingLoss: {
    type: "CosineEmbeddingLoss",
    label: "CosineEmbeddingLoss",
    category: "loss",
    description: "Cosine distance-based loss for similarity",
    params: [{ name: "margin", type: "float", default: 0.0 }],
    inputHandles: [
      { id: "input1", label: "x1", dataType: "tensor" },
      { id: "input2", label: "x2", dataType: "tensor" },
      { id: "target", label: "target", dataType: "tensor" },
    ],
    outputHandles: [],
  },
  CTCLoss: {
    type: "CTCLoss",
    label: "CTCLoss",
    category: "loss",
    description: "Connectionist Temporal Classification (OCR, ASR)",
    params: [
      { name: "blank", type: "int", default: 0 },
      { name: "reduction", type: "select", default: "mean", options: ["none", "mean", "sum"] },
    ],
    inputHandles: [
      { id: "input", label: "log_probs", dataType: "tensor" },
      { id: "target", label: "targets", dataType: "tensor" },
    ],
    outputHandles: [],
  },

  // --- Distance ---
  CosineSimilarity: {
    type: "CosineSimilarity",
    label: "CosineSimilarity",
    category: "math",
    description: "Cosine similarity between two tensors",
    params: [{ name: "dim", type: "int", default: 1 }],
    inputHandles: [
      { id: "a", label: "x1", dataType: "tensor" },
      { id: "b", label: "x2", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },
  PairwiseDistance: {
    type: "PairwiseDistance",
    label: "PairwiseDistance",
    category: "math",
    description: "Pairwise distance between tensor rows",
    params: [{ name: "p", type: "float", default: 2.0 }],
    inputHandles: [
      { id: "a", label: "x1", dataType: "tensor" },
      { id: "b", label: "x2", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },

  // --- Padding ---
  ZeroPad2d: {
    type: "ZeroPad2d",
    label: "ZeroPad2d",
    category: "reshape",
    description: "Pad tensor with zeros around spatial dims",
    params: [{ name: "padding", type: "int", default: 1, min: 0 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  ReflectionPad2d: {
    type: "ReflectionPad2d",
    label: "ReflectionPad2d",
    category: "reshape",
    description: "Pad with reflection of input",
    params: [{ name: "padding", type: "int", default: 1, min: 0 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  ReplicationPad2d: {
    type: "ReplicationPad2d",
    label: "ReplicationPad2d",
    category: "reshape",
    description: "Pad by replicating border values",
    params: [{ name: "padding", type: "int", default: 1, min: 0 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Embedding ---
  EmbeddingBag: {
    type: "EmbeddingBag",
    label: "EmbeddingBag",
    category: "embedding",
    description: "Embedding with sum/mean/max pooling (efficient for sparse)",
    params: [
      { name: "num_embeddings", type: "int", default: 10000, min: 1 },
      { name: "embedding_dim", type: "int", default: 128, min: 1 },
      { name: "mode", type: "select", default: "mean", options: ["sum", "mean", "max"] },
    ],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Additional Recurrent ---
  RNN: {
    type: "RNN",
    label: "RNN",
    category: "recurrent",
    description: "Simple recurrent neural network",
    params: [
      { name: "input_size", type: "int", default: 128, min: 1 },
      { name: "hidden_size", type: "int", default: 128, min: 1 },
      { name: "num_layers", type: "int", default: 1, min: 1 },
      { name: "nonlinearity", type: "select", default: "tanh", options: ["tanh", "relu"] },
      { name: "bias", type: "bool", default: true },
      { name: "batch_first", type: "bool", default: true },
      { name: "bidirectional", type: "bool", default: false },
    ],
    inputHandles: [tensorIn],
    outputHandles: [
      { id: "out", label: "output", dataType: "tensor" },
      { id: "hidden", label: "hidden", dataType: "tensor" },
    ],
  },

  // --- Additional Math ---
  Multiply: {
    type: "Multiply",
    label: "Multiply",
    category: "math",
    description: "Element-wise multiplication",
    params: [],
    inputHandles: [
      { id: "a", label: "A", dataType: "tensor" },
      { id: "b", label: "B", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },
  Split: {
    type: "Split",
    label: "Split",
    category: "math",
    description: "Split tensor into chunks along a dimension",
    params: [
      { name: "split_size", type: "int", default: 1, min: 1 },
      { name: "dim", type: "int", default: 0 },
    ],
    inputHandles: [tensorIn],
    outputHandles: [
      { id: "out_0", label: "chunk_0", dataType: "tensor" },
      { id: "out_1", label: "chunk_1", dataType: "tensor" },
    ],
  },

  // --- Additional Reshape ---
  Permute: {
    type: "Permute",
    label: "Permute",
    category: "reshape",
    description: "Rearrange tensor dimensions",
    params: [{ name: "dims", type: "string", default: "[0, 2, 1]" }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Squeeze: {
    type: "Squeeze",
    label: "Squeeze",
    category: "reshape",
    description: "Remove dimensions of size 1",
    params: [{ name: "dim", type: "optional_int", default: null }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },
  Unsqueeze: {
    type: "Unsqueeze",
    label: "Unsqueeze",
    category: "reshape",
    description: "Add a dimension of size 1",
    params: [{ name: "dim", type: "int", default: 0 }],
    inputHandles: [tensorIn],
    outputHandles: [tensorOut],
  },

  // --- Transformer (full) ---
  Transformer: {
    type: "Transformer",
    label: "Transformer",
    category: "transformer",
    description: "Full encoder-decoder transformer",
    params: [
      { name: "d_model", type: "int", default: 512, min: 1 },
      { name: "nhead", type: "int", default: 8, min: 1 },
      { name: "num_encoder_layers", type: "int", default: 6, min: 1 },
      { name: "num_decoder_layers", type: "int", default: 6, min: 1 },
      { name: "dim_feedforward", type: "int", default: 2048, min: 1 },
      { name: "dropout", type: "float", default: 0.1, min: 0, max: 1 },
      { name: "batch_first", type: "bool", default: true },
    ],
    inputHandles: [
      { id: "src", label: "src", dataType: "tensor" },
      { id: "tgt", label: "tgt", dataType: "tensor" },
    ],
    outputHandles: [tensorOut],
  },
};

export const CATEGORY_LABELS: Record<NodeCategory, string> = {
  io: "Input / Output",
  data: "Data",
  linear: "Linear",
  conv: "Convolution",
  pool: "Pooling",
  norm: "Normalization",
  activation: "Activation",
  recurrent: "Recurrent",
  transformer: "Transformer",
  dropout: "Regularization",
  embedding: "Embedding",
  loss: "Loss Function",
  reshape: "Reshape",
  math: "Math",
  vision: "Vision",
};

export const CATEGORY_COLORS: Record<NodeCategory, string> = {
  io: "var(--node-io)",
  data: "var(--node-data)",
  linear: "var(--node-linear)",
  conv: "var(--node-conv)",
  pool: "var(--node-pool)",
  norm: "var(--node-norm)",
  activation: "var(--node-activation)",
  recurrent: "var(--node-recurrent)",
  transformer: "var(--node-transformer)",
  dropout: "var(--node-dropout)",
  embedding: "var(--node-embedding)",
  loss: "var(--node-loss)",
  reshape: "var(--node-reshape)",
  math: "var(--node-math)",
  vision: "var(--node-vision)",
};

export function getModulesByCategory(): Record<NodeCategory, ModuleDef[]> {
  const result: Record<string, ModuleDef[]> = {};
  for (const def of Object.values(MODULE_REGISTRY)) {
    if (!result[def.category]) result[def.category] = [];
    result[def.category].push(def);
  }
  return result as Record<NodeCategory, ModuleDef[]>;
}
