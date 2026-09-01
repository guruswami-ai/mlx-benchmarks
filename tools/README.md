# Diagnostic and benchmark tools

Written while getting Kimi-K3 (2.78 T) running in TP4 across four M3 Ultras.
See [`../docs/KIMI_K3_TP4.md`](../docs/KIMI_K3_TP4.md) for the results and
[`../docs/DISTRIBUTED_TROUBLESHOOTING.md`](../docs/DISTRIBUTED_TROUBLESHOOTING.md)
for the runbook these came out of.

| Tool | What it answers |
|---|---|
| `power_collect.py` | Cluster power/thermal/GPU telemetry from the `mactop` exporters. **Watts are a more honest health signal than GPU utilisation**, which reads 100 % while the GPU idles |
| `link_watch.py` | Is a TB5 link dropping *during* a run? Before/after checks cannot distinguish "was already down" from "dropped mid-run" |
| `rdma_stress.py` | Model-free `all_sum` escalation, 7 KB → 134 MB. Separates *node/transport* faults from *model/framework* faults in ~30 s instead of a 4-minute model load |
| `collective_bench.py` | Per-collective latency at the payload sizes a model actually uses |
| `moe_bench.py` | MoE gather cost vs batch size — how much batching recovers |
| `bench_matrix.py` | Concurrency × context sweep, streaming, with TTFT |
| `prefill_test.py` | Escalating prompt sizes so a hang localises to a value, not "somewhere" |
| `ctx_sweep.py`, `batch_sweep.py` | Simpler single-axis sweeps |
| `prefill_scaling_check.sh` | Does prefill scale the way the architecture claims? Sparse attention holds its rate near flat as context grows. A dense path loses rate in proportion to context. Fails when the measurement disagrees with `--expect` |
| `model_file_shadowing.py` | Is a checkpoint's bundled model code displacing your runtime's optimised implementation? Static check, no weights loaded |

Hostnames and paths are specific to this cluster; adjust the `NODES` list and
model paths before use.

## Why the last two exist

A checkpoint can ship its own copy of a model's code and declare it in
`config.json` as `model_file`. That declaration takes precedence over the
runtime's own implementation of the same architecture. Where the runtime ships
accelerated kernels, the bundled Python runs instead and the kernels are never
called.

Measured on GLM-5.3 4-bit (744 B MoE, MLA + DeepSeek Sparse Attention) on one
M3 Ultra 512 GB under oMLX 0.6.4. Same weights, same server, same settings. The
only change is `model_file` removed from a copied `config.json`:

| | Bundled file | Runtime kernels |
|---|---:|---:|
| 64 K prefill, 65,407 tokens | 2,341.7 s | **431.7 s** |
| Prefill rate | 28.0 tok/s | **151.7 tok/s** |
| Memory-guard restarts | 2 | **0** |
| 128 K prefill, 130,100 tokens | 5,968.8 s | **2,009.3 s** |
| Decode | 16.4 tok/s | 14.7 tok/s |
| Needle retrieval at 64 K and 128 K | correct | correct |

Output was correct throughout. Only speed changed.

The failure is hard to see. The runtime logged that its optimised module
registered and that its native kernels were available, and both statements were
true. The kernels loaded. They were never called. A single reading of 28 tok/s
for a 744 B model on Apple Silicon also looks reasonable, and it happens to
match the rate documented for a missing kernel build, which sends you to the
wrong cause.

Only the shape of the rate curve across two context depths shows it. With the
kernels engaged: 15,658 tokens at 151.9 tok/s and 47,101 tokens at 155.9 tok/s.
Context grew 3.01 times and the rate held at 1.03.

Two cautions. Where a runtime's own implementation carries a bug that the
bundled file avoids, the shadowing is protecting you, so measure correctness as
well as speed. And check any multi-token-prediction path separately, because
removing `model_file` moves that handling to the runtime as well.
