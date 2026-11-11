# Language Models

TensorBloom can open a pretrained large language model (LLM) and show its architecture and what it takes to run or fine-tune it on your computer.

The models that fit are the ones that fit on one machine. As a rough guide for the weights alone:

| Model size | 16-bit | 8-bit | 4-bit |
|---|---|---|---|
| 1B | 2 GB | 1 GB | 0.6 GB |
| 8B | 16 GB | 8 GB | 4.5 GB |
| 70B | 140 GB | 70 GB | 38 GB |

Running also needs room for the KV cache, which grows with the context length. Fine-tuning needs more again. The **Language model** tab works all of this out for the model you import.

## Importing a model

**File > Import Language Model** (or **Import a language model** on the empty canvas) reads a model's architecture without downloading its weights:

- **Hugging Face**: enter a model id such as `Qwen/Qwen2.5-0.5B-Instruct`, or paste its URL. TensorBloom reads `config.json` and the Hub's summary of the weight files, so the parameter count is exact.
- **File on this computer**: a `.gguf` file, a `.safetensors` file, a `config.json`, or a folder with a model in it. Only the file headers are read.

Gated models (Llama, Gemma and others) need a Hugging Face token from an account that has accepted the model's license. Click **Use a Hugging Face token** and paste it, or run `huggingface-cli login` once and TensorBloom finds it. The token is kept until you close TensorBloom and is never saved in a project.

**Open as graph** draws the model: tokens, the embedding, one **DecoderBlock** per layer (collapsed into a group you can open), the final norm and `lm_head`. Every node is a real layer, so shapes, parameter counts and the generated code work as for any other model.

### DecoderBlock

DecoderBlock is the block Llama, Mistral, Qwen, Gemma and most current LLMs are built from: RMSNorm, causal self-attention with rotary positions and grouped-query heads, then a gated MLP (or a mixture of experts). It's also in the layer palette, so you can build and train a small LLM of your own from it.

## Memory needed

The **Language model** tab's **Overview** shows the architecture and a table of the memory needed to:

- **Run (chat)**: the weights, the KV cache for the chosen context length, and some overhead
- **LoRA fine-tune**: the frozen weights, the small trainable adapters with their optimizer state, and the activations of one training step (with gradient checkpointing)
- **Full fine-tune**: every weight with its gradient and optimizer state, about 16 bytes per parameter

for 16-bit, 8-bit and 4-bit weights. Each cell is marked green (fits), amber (tight) or red (too big) against your GPU's memory. On Apple silicon it's measured against about three quarters of the shared memory, and without a GPU against RAM. Hover a cell for the breakdown. These are estimates with batch size 1: real use varies with the library version and settings.

8-bit and 4-bit loading uses bitsandbytes, which needs an NVIDIA GPU. On a Mac, models run at 16 bits on the GPU.

GGUF files are made for llama.cpp. TensorBloom reads and sizes them; they can't be loaded into PyTorch.
