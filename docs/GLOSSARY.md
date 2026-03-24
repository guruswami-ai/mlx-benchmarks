# Glossary

Quick reference for terms used throughout these benchmarks. For deeper explanations, see [Basic Concepts](CONCEPTS.md), [Quantisation](QUANTISATION.md), and [Distributed Inference](DISTRIBUTED_INFERENCE.md).

---

## Inference Metrics

| Term | Meaning |
|------|---------|
| **TPS** | Tokens Per Second. How fast the model generates text. 30 TPS = fluent. 3 TPS = slow. |
| **TTFT** | Time To First Token. The pause before the model starts responding. |
| **Perplexity** | Quality score. Lower = better. Only meaningful as a ratio between quants of the same model. |
| **Prompt TPS** | How fast the model processes your input. Much faster than generation TPS. |

## Temperature

**Temperature** controls how "creative" or "random" the model's output is. It divides the model's log-probabilities by the temperature value before sampling.

- **0.0** - deterministic, always picks the most likely token. Best for factual Q&A and code.
- **0.3-0.7** - slight variation. Good for most tasks.
- **1.0** - full sampling from the model's natural distribution. Default.
- **1.5+** - increasingly chaotic. Can be creative or nonsense.

Temperature effectively multiplies uncertainty. A 2% perplexity increase at Q4 is imperceptible at temperature 0. At temperature 1.5, that same 2% gets amplified into noticeably different word choices.

## Quantisation

| Term | Meaning |
|------|---------|
| **F16 / BF16** | 16-bit full precision. Quality baseline. |
| **Q8 through Q2** | Integer quantisation at 8 to 2 bits per weight. |
| **FP8** | 8-bit floating point (DeepSeek V3). Different from Q8. |
| **K-quants** | llama.cpp mixed-precision quants. `_S`/`_M`/`_L` = how many layers get higher precision. |
| **IQ quants** | Importance quants. llama.cpp only. Use importance matrices for smarter compression. |
| **AWQ / GPTQ** | Alternative quantisation methods. Same goal, different algorithms. |

## Model Architecture

| Term | Meaning |
|------|---------|
| **Dense** | Every parameter activates every token. Llama, Qwen, Gemma, Mistral. |
| **MoE** | Mixture of Experts. Only a subset of parameters activate per token. Mixtral, DeepSeek, Kimi. |
| **GQA** | Grouped Query Attention. Shares KV heads to reduce cache size. |
| **MLA** | Multi-head Latent Attention. Compresses KV cache via learned projections. DeepSeek, Kimi. |

## Memory

| Term | Meaning |
|------|---------|
| **VRAM** | GPU-dedicated memory on NVIDIA cards. Hard limit for model size. |
| **Unified memory** | Shared CPU/GPU memory on Apple Silicon. No VRAM wall. |
| **KV cache** | Stored attention keys/values. Grows linearly with context length. |
| **Context window** | Maximum tokens the model can process at once (prompt + response). |

## Distributed Inference

| Term | Meaning |
|------|---------|
| **EP** | Embarrassingly Parallel. Independent model copies. Zero overhead. |
| **TP** | Tensor Parallelism. Split every layer across nodes. Lots of sync. |
| **PP** | Pipeline Parallelism. Assign whole layers to nodes. Minimal sync. |
| **RDMA** | Remote Direct Memory Access. Memory-to-memory transfers bypassing OS/CPU. |
| **JACCL** | Apple's RDMA communication library for TB5 distributed inference. |
| **All-reduce** | Every node contributes a value, every node gets the sum. Core TP operation. |

## File Formats

| Term | Meaning |
|------|---------|
| **GGUF** | llama.cpp model format. Used for NVIDIA/CPU inference. |
| **Safetensors** | Hugging Face / MLX model format. Used for Apple Silicon inference. |

## Hardware

| Term | Meaning |
|------|---------|
| **Metal** | Apple's GPU framework (equivalent to CUDA). |
| **MLX** | Apple's ML framework for Apple Silicon. |
| **CUDA** | NVIDIA's GPU programming framework. |
| **TB5** | Thunderbolt 5. Used for RDMA interconnect between Mac Studios. |
| **NVLink** | NVIDIA's high-speed GPU interconnect. 900 GB/s. Enterprise only. |

## Decoding Hugging Face Model Names

`mlx-community/DeepSeek-V3-0324-4bit`:

| Part | Meaning |
|------|---------|
| `mlx-community` | Community-converted models for MLX |
| `DeepSeek-V3` | Model family, version 3 |
| `0324` | March 2024 release |
| `4bit` | Quantised to 4 bits per weight |

`bartowski/Qwen2.5-32B-Instruct-GGUF`:

| Part | Meaning |
|------|---------|
| `bartowski` | The person who quantised it |
| `Qwen2.5` | Model family, version 2.5 (by Alibaba) |
| `32B` | 32 billion parameters |
| `Instruct` | Instruction-tuned |
| `GGUF` | File format for llama.cpp |

Inside the repo: `Qwen2.5-32B-Instruct-Q4_K_M.gguf` - 4-bit k-quant, medium variant.

## Perplexity Measurement

**The dataset matters.** Common choices:
- **WikiText** - Wikipedia articles. Standard benchmark.
- **allenai/tulu-3-sft-mixture** - instruction-following data.
- **C4, The Pile, RedPajama** - large web-crawl datasets.

Perplexity numbers are only comparable when measured on the same dataset with the same parameters. Typical settings: 256 samples, 512 tokens per sequence, batch size 8.
