# Module Reference

<!-- Generated from app/src/lib/registry.ts by `npm run docs:modules` in app/. Don't edit by hand. -->

The layer palette has 86 modules in 15 categories. Each maps to its PyTorch counterpart with the same parameters, shown as a node you configure in the Properties panel, where the **?** beside each parameter explains it. The defaults below are what a new node starts with.

To add one, drag it from the **Layers** panel, click it there, or press **Ctrl+K** and type its name.

## Input / Output

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Input** | Network input tensor | `shape`=`[1, 3, 224, 224]`, `dtype`=`float32` |
| **Output** | Network output | — |

## Data

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Data** | Data source — built-in presets or custom tensor files | Built-in datasets or your own files; see [Custom Data](custom-data.md) |

## Linear

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Linear** | Fully connected layer: y = xA^T + b | `in_features`=`512`, `out_features`=`256`, `bias`=`true` |
| **Identity** | Pass-through (identity mapping) | — |

## Convolution

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Conv1d** | 1D convolution over temporal data | `in_channels`=`1`, `out_channels`=`32`, `kernel_size`=`3`, `stride`=`1`, `padding`=`0`, `dilation`=`1`, `groups`=`1`, `bias`=`true` |
| **Conv2d** | 2D convolution over images | `in_channels`=`3`, `out_channels`=`64`, `kernel_size`=`3`, `stride`=`1`, `padding`=`0`, `dilation`=`1`, `groups`=`1`, `bias`=`true` |
| **Conv3d** | 3D convolution over volumes | `in_channels`=`1`, `out_channels`=`32`, `kernel_size`=`3`, `stride`=`1`, `padding`=`0`, `dilation`=`1`, `groups`=`1`, `bias`=`true` |
| **ConvTranspose2d** | 2D transposed convolution (upsampling) | `in_channels`=`64`, `out_channels`=`32`, `kernel_size`=`3`, `stride`=`1`, `padding`=`0`, `output_padding`=`0`, `bias`=`true` |
| **ConvTranspose1d** | 1D transposed convolution (upsampling) | `in_channels`=`64`, `out_channels`=`32`, `kernel_size`=`3`, `stride`=`1`, `padding`=`0`, `bias`=`true` |

## Pooling

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **MaxPool2d** | 2D max pooling — downsample by taking maximum | `kernel_size`=`2`, `stride`, `padding`=`0` |
| **AvgPool2d** | 2D average pooling | `kernel_size`=`2`, `stride`, `padding`=`0` |
| **AdaptiveAvgPool2d** | Adaptive average pooling to target output size | `output_size`=`[1,1]` |
| **MaxPool1d** | 1D maximum pooling | `kernel_size`=`2`, `stride`, `padding`=`0` |
| **AvgPool1d** | 1D average pooling | `kernel_size`=`2`, `stride`, `padding`=`0` |
| **MaxPool3d** | 3D maximum pooling | `kernel_size`=`2`, `stride`, `padding`=`0` |
| **AdaptiveMaxPool2d** | Adaptive max pooling to target output size | `output_size`=`[1,1]` |
| **AdaptiveAvgPool1d** | Adaptive average pooling for 1D data | `output_size`=`1` |

## Normalization

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **BatchNorm1d** | 1D batch normalization | `num_features`=`64`, `eps`=`0.00001`, `momentum`=`0.1`, `affine`=`true` |
| **BatchNorm2d** | 2D batch normalization (after Conv2d) | `num_features`=`64`, `eps`=`0.00001`, `momentum`=`0.1`, `affine`=`true` |
| **LayerNorm** | Layer normalization (transformers) | `normalized_shape`=`[512]`, `eps`=`0.00001`, `elementwise_affine`=`true` |
| **GroupNorm** | Group normalization | `num_groups`=`32`, `num_channels`=`64`, `eps`=`0.00001`, `affine`=`true` |
| **RMSNorm** | Root mean square normalization | `normalized_shape`=`[512]`, `eps`=`0.00001` |
| **InstanceNorm1d** | 1D instance normalization (style transfer) | `num_features`=`64`, `eps`=`0.00001`, `affine`=`false` |
| **InstanceNorm2d** | 2D instance normalization (style transfer) | `num_features`=`64`, `eps`=`0.00001`, `affine`=`false` |
| **BatchNorm3d** | 3D batch normalization | `num_features`=`64`, `eps`=`0.00001`, `momentum`=`0.1`, `affine`=`true` |

## Activation

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **ReLU** | Rectified Linear Unit: max(0, x) | `inplace`=`false` |
| **LeakyReLU** | Leaky ReLU: max(neg_slope*x, x) | `negative_slope`=`0.01`, `inplace`=`false` |
| **GELU** | Gaussian Error Linear Unit | `approximate`=`none` |
| **SiLU** | Sigmoid Linear Unit (Swish): x * sigmoid(x) | `inplace`=`false` |
| **Mish** | Mish: x * tanh(softplus(x)) | `inplace`=`false` |
| **Sigmoid** | Sigmoid: 1 / (1 + exp(-x)) | — |
| **Tanh** | Hyperbolic tangent | — |
| **Softmax** | Softmax normalization along a dimension | `dim`=`-1` |
| **PReLU** | Parametric ReLU with learnable slope | `num_parameters`=`1`, `init`=`0.25` |
| **ELU** | Exponential Linear Unit | `alpha`=`1`, `inplace`=`false` |
| **SELU** | Scaled Exponential Linear Unit (self-normalizing) | `inplace`=`false` |
| **CELU** | Continuously Differentiable ELU | `alpha`=`1`, `inplace`=`false` |
| **Hardswish** | Efficient piecewise-linear Swish approximation | `inplace`=`false` |
| **Hardsigmoid** | Efficient piecewise-linear Sigmoid approximation | `inplace`=`false` |
| **ReLU6** | ReLU capped at 6 (mobile networks) | `inplace`=`false` |
| **LogSoftmax** | Log of softmax (use with NLLLoss) | `dim`=`-1` |
| **Softplus** | Smooth approximation of ReLU: log(1 + exp(x)) | `beta`=`1`, `threshold`=`20` |
| **GLU** | Gated Linear Unit — splits and gates with sigmoid | `dim`=`-1` |

## Recurrent

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **LSTM** | Long Short-Term Memory | `input_size`=`512`, `hidden_size`=`256`, `num_layers`=`1`, `bias`=`true`, `batch_first`=`true`, `dropout`=`0`, `bidirectional`=`false` |
| **GRU** | Gated Recurrent Unit | `input_size`=`512`, `hidden_size`=`256`, `num_layers`=`1`, `bias`=`true`, `batch_first`=`true`, `dropout`=`0`, `bidirectional`=`false` |
| **RNN** | Simple recurrent neural network | `input_size`=`128`, `hidden_size`=`128`, `num_layers`=`1`, `nonlinearity`=`tanh`, `bias`=`true`, `batch_first`=`true`, `bidirectional`=`false` |

## Transformer

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **MultiheadAttention** | Multi-head scaled dot-product attention | `embed_dim`=`512`, `num_heads`=`8`, `dropout`=`0`, `bias`=`true`, `batch_first`=`true` |
| **TransformerEncoderLayer** | Self-attention + FFN + norms + residuals | `d_model`=`512`, `nhead`=`8`, `dim_feedforward`=`2048`, `dropout`=`0.1`, `activation`=`relu`, `batch_first`=`true`, `norm_first`=`false` |
| **DecoderBlock** | LLM block (Llama, Mistral, Qwen): causal attention with rotary positions, then a gated MLP | `d_model`=`512`, `n_heads`=`8`, `n_kv_heads`=`8`, `head_dim`=`0`, `ffn_dim`=`1408`, `mlp`=`swiglu`, `norm`=`rmsnorm`, `norm_eps`=`0.00001`, `rope_theta`=`10000`, `bias`=`none`, `n_experts`=`0`, `experts_per_token`=`2` |
| **TransformerDecoderLayer** | Self-attn + cross-attn + FFN + norms | `d_model`=`512`, `nhead`=`8`, `dim_feedforward`=`2048`, `dropout`=`0.1`, `activation`=`relu`, `batch_first`=`true`, `norm_first`=`false` |
| **Transformer** | Full encoder-decoder transformer | `d_model`=`512`, `nhead`=`8`, `num_encoder_layers`=`6`, `num_decoder_layers`=`6`, `dim_feedforward`=`2048`, `dropout`=`0.1`, `batch_first`=`true` |

## Regularization

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Dropout** | Randomly zero elements during training | `p`=`0.5`, `inplace`=`false` |
| **Dropout2d** | Zero entire channels randomly | `p`=`0.5`, `inplace`=`false` |
| **Dropout3d** | Zero entire channels for volumetric data | `p`=`0.5`, `inplace`=`false` |
| **AlphaDropout** | Dropout preserving self-normalizing property (for SELU) | `p`=`0.5` |

## Embedding

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Embedding** | Lookup table: index -> dense vector | `num_embeddings`=`10000`, `embedding_dim`=`512`, `padding_idx` |
| **EmbeddingBag** | Embedding with sum/mean/max pooling (efficient for sparse) | `num_embeddings`=`10000`, `embedding_dim`=`128`, `mode`=`mean` |

## Loss Function

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **CrossEntropyLoss** | Cross entropy for multi-class classification | `reduction`=`mean`, `label_smoothing`=`0` |
| **MSELoss** | Mean Squared Error for regression | `reduction`=`mean` |
| **BCEWithLogitsLoss** | Binary cross entropy with sigmoid | `reduction`=`mean` |
| **L1Loss** | Mean Absolute Error loss | `reduction`=`mean` |
| **HuberLoss** | Robust regression loss (smooth L1) | `reduction`=`mean`, `delta`=`1` |
| **NLLLoss** | Negative log likelihood (after LogSoftmax) | `reduction`=`mean` |
| **KLDivLoss** | KL divergence (for VAEs, knowledge distillation) | `reduction`=`mean`, `log_target`=`false` |
| **TripletMarginLoss** | Metric learning with anchor/positive/negative | `margin`=`1`, `p`=`2` |
| **CosineEmbeddingLoss** | Cosine distance-based loss for similarity | `margin`=`0` |
| **CTCLoss** | Connectionist Temporal Classification (OCR, ASR) | `blank`=`0`, `reduction`=`mean` |

## Reshape

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Flatten** | Flatten dimensions into one | `start_dim`=`1`, `end_dim`=`-1` |
| **Unflatten** | Expand a dimension into multiple | `dim`=`1`, `unflattened_size`=`[8, 8]` |
| **ZeroPad2d** | Pad tensor with zeros around spatial dims | `padding`=`1` |
| **ReflectionPad2d** | Pad with reflection of input | `padding`=`1` |
| **ReplicationPad2d** | Pad by replicating border values | `padding`=`1` |
| **Permute** | Rearrange tensor dimensions | `dims`=`[0, 2, 1]` |
| **Squeeze** | Remove dimensions of size 1 | `dim` |
| **Unsqueeze** | Add a dimension of size 1 | `dim`=`0` |

## Math

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Add** | Element-wise addition (residual connection) | — |
| **Concat** | Concatenate tensors along a dimension | `dim`=`1` |
| **MatMul** | Matrix multiplication | — |
| **CosineSimilarity** | Cosine similarity between two tensors | `dim`=`1` |
| **PairwiseDistance** | Pairwise distance between tensor rows | `p`=`2` |
| **Multiply** | Element-wise multiplication | — |
| **Split** | Split tensor into chunks along a dimension | `split_size`=`1`, `dim`=`0` |

## Vision

| Module | What it does | Parameters (defaults) |
|---|---|---|
| **Upsample** | Upsample tensor spatially | `scale_factor`=`2`, `mode`=`nearest` |
| **PixelShuffle** | Sub-pixel convolution upsampling | `upscale_factor`=`2` |
| **PixelUnshuffle** | Inverse of PixelShuffle (downsampling) | `downscale_factor`=`2` |

## Your own layers

When none of these does the job, write one: **Insert > Custom Block** adds a reusable layer with its own `__init__` and `forward`, and **Convert to Code Block** turns any node into code. See [Code Blocks](code-blocks.md).
