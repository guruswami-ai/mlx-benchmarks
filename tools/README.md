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

Hostnames and paths are specific to this cluster; adjust the `NODES` list and
model paths before use.
