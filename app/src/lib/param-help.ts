/**
 * One-line explanations of layer parameters, shown next to each field in the Properties panel.
 * Keyed by parameter name; MODULE_HELP overrides a name where it means something else for that layer.
 */

const PARAM_HELP: Record<string, string> = {
  // Sizes
  in_features: "Size of each input sample: the last dimension of the tensor coming in.",
  out_features: "Size of each output sample: how many numbers this layer produces per input.",
  in_channels: "Channels in the input, e.g. 3 for an RGB image. Must match the previous layer's output channels.",
  out_channels: "Channels this layer produces, one per learned filter.",
  num_features: "Channels (or features) in the input, the C in [N, C, ...].",
  num_channels: "Channels in the input. Must be divisible by num_groups.",
  input_size: "Features in each step of the input sequence: its last dimension.",
  hidden_size: "Features in the hidden state, which is also the size of each output step.",
  num_layers: "How many recurrent layers are stacked, each feeding the next.",
  embed_dim: "Size of each token's vector. Must be divisible by num_heads.",
  d_model: "Size of each token's vector throughout the transformer. Must be divisible by nhead.",
  dim_feedforward: "Width of the feed-forward network inside each transformer layer, often 4 × d_model.",
  num_embeddings: "How many distinct tokens or ids there are (the vocabulary size).",
  embedding_dim: "Length of the vector each token is mapped to.",
  normalized_shape: "The trailing dimensions to normalize over, usually just the feature size.",
  num_groups: "How many groups the channels are split into, each normalized separately.",
  num_heads: "Attention heads that run in parallel. embed_dim is split evenly between them.",
  nhead: "Attention heads in each layer. d_model is split evenly between them.",
  num_encoder_layers: "How many encoder layers are stacked.",
  num_decoder_layers: "How many decoder layers are stacked.",
  num_parameters: "Learned slopes: 1 shared by all channels, or one per channel.",
  output_size: "The output's spatial size. The layer picks its window to produce exactly this size.",

  // Convolution and pooling
  kernel_size: "Size of the sliding window, e.g. 3 for 3×3.",
  stride: "How far the window moves each step. 2 halves the spatial size.",
  padding: "Zeros added around the input's edges, so the window can cover them. For odd kernel sizes, kernel_size // 2 keeps the size the same at stride 1.",
  dilation: "Spacing between the window's taps. Larger values see a wider area with the same number of weights.",
  groups: "Splits channels into groups that are convolved separately. Equal to in_channels for a depthwise convolution.",
  output_padding: "Extra size added to one side of the output, to pick between output sizes a stride allows.",
  upscale_factor: "How much to enlarge height and width, by moving channels into space. Channels must be divisible by its square.",
  downscale_factor: "How much to shrink height and width, by moving space into channels.",
  scale_factor: "How much to enlarge height and width, e.g. 2 doubles them.",

  // Regularization and normalization
  p: "Chance that each value is zeroed during training, e.g. 0.2. Nothing is dropped during evaluation.",
  dropout: "Dropout applied inside the layer during training, from 0 (none) to 1.",
  eps: "A small number added to the variance so it never divides by zero.",
  momentum: "How fast the running mean and variance follow new batches. Used at evaluation time.",
  affine: "Learn a scale and shift after normalizing.",
  elementwise_affine: "Learn a scale and shift for each feature after normalizing.",

  // Activations
  inplace: "Overwrite the input instead of making a new tensor. Saves memory, but breaks layers that need the input later.",
  negative_slope: "Slope for negative inputs, e.g. 0.01. ReLU uses 0.",
  alpha: "How far below zero the output can go: very negative inputs approach -alpha.",
  approximate: "\"tanh\" uses a faster approximation of GELU; \"none\" computes it exactly.",
  beta: "Sharpness of the curve. Higher values look more like ReLU.",
  threshold: "Where input × beta is above this, the output is just the input, for numerical stability.",
  init: "Starting value of the learned slope for negative inputs.",
  nonlinearity: "Activation used inside each step.",
  activation: "Activation used in the feed-forward network.",

  // Layout
  bias: "Learn an offset that's added to the output.",
  batch_first: "Input and output are [batch, sequence, features] instead of [sequence, batch, features].",
  bidirectional: "Also read the sequence backwards and join both directions, doubling the output features.",
  norm_first: "Normalize before attention and the feed-forward network (pre-norm), which often trains more stably.",
  padding_idx: "A token id whose vector stays zero and isn't trained, for padding.",
  start_dim: "First dimension to flatten. 1 keeps the batch dimension.",
  end_dim: "Last dimension to flatten. -1 means the last one.",
  dim: "The dimension this operates along. -1 means the last one.",
  dims: "The new order of the dimensions, e.g. [0, 2, 1] swaps the last two.",
  unflattened_size: "The shape that dimension dim is unfolded into.",
  split_size: "Size of each piece. The last piece may be smaller.",
  shape: "Shape of one input batch, including the batch dimension, e.g. [1, 3, 32, 32].",
  dtype: "Number type of the input tensor.",

  // Losses
  reduction: "How per-sample losses become one number: \"mean\" averages them, \"sum\" adds them, \"none\" keeps them all.",
  label_smoothing: "Moves a little of the target away from the true class, e.g. 0.1, which can reduce overconfidence.",
  delta: "Where the loss switches from squared to linear error. Smaller values are less sensitive to outliers.",
  margin: "How far apart matching and non-matching pairs should be.",
  blank: "Index of the blank label.",
  log_target: "The target is already given as log probabilities.",

  // Data
  source: "Where the training data comes from: a built-in dataset or your own files.",
  val_split: "Share of the data held out to check the model on, e.g. 0.2 for 20%.",
};

/** Where a parameter name means something different for a particular layer */
const MODULE_HELP: Record<string, Record<string, string>> = {
  TripletMarginLoss: { p: "Which norm measures distance: 2 is the usual Euclidean distance." },
  PairwiseDistance: { p: "Which norm measures distance: 2 is the usual Euclidean distance." },
  Softmax: { dim: "The dimension whose values become probabilities that sum to 1. -1 for the last one." },
  LogSoftmax: { dim: "The dimension whose values become log probabilities. -1 for the last one." },
  Concatenate: { dim: "The dimension the inputs are joined along. Every other dimension must match." },
  Split: { dim: "The dimension that's cut into pieces." },
  Squeeze: { dim: "The dimension of size 1 to remove." },
  Unsqueeze: { dim: "Where to insert a new dimension of size 1." },
  Unflatten: { dim: "The dimension that's unfolded into unflattened_size." },
  GLU: { dim: "The dimension that's split in half: one half gates the other." },
  CosineSimilarity: { dim: "The dimension the vectors lie along." },
  Upsample: { mode: "How new pixels are filled in: \"nearest\" copies neighbors, \"bilinear\" blends them." },
  EmbeddingBag: { mode: "How each bag's vectors are combined: \"sum\", \"mean\" or \"max\"." },
  MaxPool1d: { kernel_size: "Size of the window each output takes the maximum of." },
  MaxPool2d: { kernel_size: "Size of the window each output takes the maximum of, e.g. 2 for 2×2." },
  MaxPool3d: { kernel_size: "Size of the window each output takes the maximum of." },
  AvgPool1d: { kernel_size: "Size of the window each output averages." },
  AvgPool2d: { kernel_size: "Size of the window each output averages, e.g. 2 for 2×2." },
  ZeroPad2d: { padding: "Zeros added on each side of the height and width." },
  ReflectionPad2d: { padding: "Pixels added on each side, mirrored from the edge." },
  ReplicationPad2d: { padding: "Pixels added on each side, copied from the edge." },
  AlphaDropout: { p: "Chance that each value is dropped during training. Keeps the mean and variance, for SELU networks." },
};

export function paramHelp(moduleType: string, param: string): string | undefined {
  return MODULE_HELP[moduleType]?.[param] ?? PARAM_HELP[param];
}

/** Training settings, keyed by their label in the Training panel */
const TRAINING_HELP: Record<string, string> = {
  "Validation Split": "Share of the data held out to check the model on, e.g. 0.2 for 20%.",
  Type: "How the weights are updated from the gradients. Adam and AdamW are good defaults; SGD with momentum often does well on images.",
  "Learning Rate": "How big each update is. Too high and the loss jumps around or blows up; too low and training crawls. Around 1e-3 suits Adam.",
  "Weight Decay": "Pulls weights toward zero at every step, which reduces overfitting. 0.01 is common with AdamW; 0 turns it off.",
  Momentum: "How much of the previous update carries into the next, which smooths SGD's path. 0.9 is typical.",
  "LR Scheduler": "Changes the learning rate as training goes, usually lowering it so the model settles.",
  Epochs: "How many full passes over the training data.",
  "Batch Size": "Samples per update. Larger batches use more memory and give fewer, smoother updates.",
  "Loss Function": "What training minimizes. Cross-entropy for picking a class, MSE for predicting numbers.",
  Device: "Where training runs. Auto uses the GPU when there is one.",
  "Random Rotation (degrees)": "Turns each training image by a random angle up to this many degrees.",
  "Random Seed": "Makes runs repeatable: the same seed gives the same starting weights and data order. Left empty, a new one is picked and saved with the run.",
  "Deterministic Mode": "Makes GPU results exactly repeatable, at some cost in speed.",
  "Weight Initialization": "How the weights are set before training starts.",
  "Mixed Precision": "Does most of the math in 16-bit floats on the GPU, which is faster and uses less memory.",
  "Gradient Clipping": "Caps the size of each update, which helps when the loss suddenly spikes.",
  "Max Norm": "How large the gradients may be, all together, before they're scaled down.",
  "Early Stopping": "Ends training when the validation loss stops improving.",
  "Patience (epochs)": "How many epochs without improvement to wait before stopping.",
  "Min Delta": "The smallest drop in validation loss that counts as an improvement.",
  "Resume Checkpoint": "Start from weights saved by an earlier run instead of from scratch.",
};

export function trainingHelp(label: string): string | undefined {
  return TRAINING_HELP[label];
}

/** Layers that aren't torch.nn modules, and the torch function each one calls */
const TORCH_FUNCTIONS: Record<string, string> = {
  Add: "torch.add",
  Multiply: "torch.mul",
  MatMul: "torch.matmul",
  Concatenate: "torch.cat",
  Split: "torch.split",
  Permute: "torch.permute",
  Squeeze: "torch.squeeze",
  Unsqueeze: "torch.unsqueeze",
};

/** TensorBloom's own nodes, which have no PyTorch counterpart */
const NO_DOCS = new Set(["Input", "Output", "Data"]);

/** What a layer is in PyTorch, e.g. "torch.nn.Linear" or "torch.cat"; null for TensorBloom's own nodes */
export function torchName(moduleType: string): string | null {
  if (NO_DOCS.has(moduleType)) return null;
  return TORCH_FUNCTIONS[moduleType] ?? `torch.nn.${moduleType}`;
}

/** The PyTorch reference page for a layer, or null for TensorBloom's own nodes */
export function pytorchDocsUrl(moduleType: string): string | null {
  const name = torchName(moduleType);
  return name ? `https://pytorch.org/docs/stable/generated/${name}.html` : null;
}
