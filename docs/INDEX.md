# Documentation Index

## How This Repo Is Organised

```
mlx-benchmarks/
├── README.md                    ← Start here
├── docs/                        ← Guides and reference (you are here)
│   ├── Beginner guides
│   ├── Hardware guides
│   ├── Deep dives
│   └── INDEX.md                 ← This file
├── results/                     ← Raw benchmark data (CSV)
│   ├── {model}/benchmark-results.csv
│   ├── {model}/perplexity-results.csv
│   ├── nvidia/                  ← NVIDIA GPU data (3080/4090/5090)
│   └── perplexity-all-models.csv
├── charts/
│   ├── output/{model}/          ← Per-model charts (PNG + SVG)
│   │   ├── README.md            ← Model page with all charts
│   │   ├── H_dashboard.png      ← 4-panel overview
│   │   ├── A_tps_vs_context.png
│   │   ├── B_quant_bars.png
│   │   ├── C_topology_comparison.png (distributed models)
│   │   ├── D_ppl_vs_tps.png     ← Perplexity vs speed
│   │   ├── F_memory_vs_context.png
│   │   └── G_ttft_vs_context.png
│   ├── output/nvidia/           ← NVIDIA per-GPU charts
│   ├── plot_mlx.py              ← Chart generator
│   ├── brand.py                 ← Guruswami brand colours/theme
│   └── watermark.py             ← Apply copyright overlay
├── assets/                      ← Hero images and infographics
│   ├── docs/                    ← Gemini-generated infographics
│   └── *.jpg                    ← README hero images
└── patches/                     ← MLX-LM distributed inference patches
```

---

## Beginner Guides (Start Here)

Read these in order if you are new to LLM inference.

| Guide | What you learn |
|-------|---------------|
| [Basic Concepts](CONCEPTS.md) | TPS, TTFT, perplexity, memory, Dense vs MoE |
| [Quantisation](QUANTISATION.md) | F16→Q1, which layers survive, K-quants, platforms |
| [Model Types](MODEL_TYPES.md) | Dense, MoE, multimodal, reasoning, coding, embeddings |
| [Model Scale](MODEL_SCALE.md) | 890K→1T parameters, what changes at each size |
| [Distributed Inference](DISTRIBUTED_INFERENCE.md) | TP, PP, EP - how to split models across machines |
| [Software Landscape](SOFTWARE_LANDSCAPE.md) | Ollama, llama.cpp, MLX, vLLM, CUDA, choosing tools |
| [Agentic vs Generative](AGENTIC_VS_GENERATIVE.md) | Chat vs agents, tool calling, MCP, APIs |
| [Beyond Text](BEYOND_TEXT.md) | Diffusion, TTS, ASR, video gen, emerging architectures |
| [Glossary](GLOSSARY.md) | Quick reference for every term |
| [The Yogi Method](YOGI_METHOD.md) | Brain inference benchmarks - be more like the puppy |

## Hardware Guides

| Guide | What you learn |
|-------|---------------|
| [NVIDIA Consumer GPUs](NVIDIA_CONSUMER_GUIDE.md) | VRAM limits, what fits, gaming cards for AI |
| [Apple Silicon and MLX](APPLE_SILICON_GUIDE.md) | Unified memory, MLX, TB5 clustering, RDMA |
| [The Chakra Cluster](CHAKRA_CLUSTER.md) | 5-node mesh, EP/TP/PP configs, choosing by task |
| [Interconnects](INTERCONNECTS.md) | TB5 vs NVLink vs ethernet, RDMA broadcasting |

## Deep Dives (Benchmark Results)

| Guide | What you learn |
|-------|---------------|
| [Findings](FINDINGS.md) | 13 key findings with data |
| [Methodology](METHODOLOGY.md) | Testing protocol, reproducibility |
| [RDMA Failure Modes](RDMA_FAILURE_MODES.md) | 6 TB5 failure modes and fixes |
| [Vision](VISION.md) | Consumer GPU → cluster learning pathway |

---

## Per-Model Benchmark Pages

Click any model to see its dashboard, charts, and raw data.

### Apple Silicon (MLX)

| Model | Params | Architecture | Charts |
|-------|--------|-------------|--------|
| [Llama 3.1 8B](../charts/output/llama-8b/) | 8B | Dense | TPS, TTFT, perplexity, memory, dashboard |
| [Mistral 7B](../charts/output/mistral-7b/) | 7B | Dense | TPS, TTFT, perplexity, memory, dashboard |
| [DeepSeek Coder 7B](../charts/output/deepseek-coder-7b/) | 6.7B | Dense | TPS, TTFT, perplexity, memory, dashboard |
| [Gemma 2 9B](../charts/output/gemma-9b/) | 9B | Dense | TPS, TTFT, perplexity, memory, dashboard |
| [Qwen 2.5 14B](../charts/output/qwen-14b/) | 14B | Dense | TPS, TTFT, perplexity, memory, dashboard |
| [Qwen 2.5 32B](../charts/output/qwen25-32b/) | 32B | Dense | TPS, TTFT, topology, perplexity, memory, dashboard |
| [Mixtral 8x7B](../charts/output/mixtral-8x7b/) | 47B (13B active) | MoE | TPS, TTFT, topology, perplexity, memory, dashboard |
| [Llama 3.1 405B](../charts/output/llama-405b/) | 405B | Dense | TPS, TTFT, topology, perplexity, memory, dashboard |
| [DeepSeek V3.2](../charts/output/deepseek-v3/) | 671B (37B active) | MoE + MLA | TPS, TTFT, topology, perplexity, memory, dashboard |
| [Kimi K2.5](../charts/output/kimi-k2.5/) | 1T+ (32B active) | MoE + MLA | TPS, TTFT, topology, memory, dashboard |

### NVIDIA (via LLM Space Heater)

| GPU | Data points | Charts |
|-----|------------|--------|
| [RTX 3080](../charts/output/nvidia/NVIDIA-GeForce-RTX-3080/) | 342 | TPS, TTFT, power, quality, dashboards |
| [RTX 4090](../charts/output/nvidia/NVIDIA-GeForce-RTX-4090/) | 510 | TPS, TTFT, power, quality, dashboards |
| [RTX 5090](../charts/output/nvidia/NVIDIA-GeForce-RTX-5090/) | 501 | TPS, TTFT, power, quality, dashboards |

---

## Raw Data

| File | Rows | Contains |
|------|------|---------|
| [perplexity-all-models.csv](../results/perplexity-all-models.csv) | 42 | Perplexity across 9 models, 5 quant levels |
| [nvidia-comparison.csv](../results/nvidia-comparison.csv) | - | Cross-platform overlap models |
| Per-model `benchmark-results.csv` | 284 total | TPS, TTFT, memory, topology, context |
| Per-model `perplexity-results.csv` | 42 total | Per-quant perplexity scores |
| NVIDIA summary CSVs | 1,353 total | TPS, TTFT, power, perplexity, KV cache |
