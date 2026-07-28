#!/opt/omlx/venv/bin/python
"""Attribute Kimi-K3 TP4's 18x gap to the memory-bandwidth roofline.

Roofline says 27.4 ms/token (22.5 GB read per node at 819 GB/s -> 36 tok/s).
Observed is ~500 ms/token. This measures the prime suspect: per-collective
latency at the payload sizes the model actually uses.

Per decode step Kimi-K3 issues ~186 all_reduces:
    93 x attention o_proj   at hidden_size 7168      = 14 KB bf16
    92 x MoE expert sum     at moe_hidden  3584      =  7 KB bf16

If 186 x measured latency accounts for most of ~473 ms of unexplained time,
the fix is batching / speculative decoding (amortise the round-trips).
If it does not, the time is in the MoE gather/scatter or dispatch overhead and
the fix is somewhere else entirely.
"""
import socket
import sys
import time

import mlx.core as mx

WARMUP, ITERS = 20, 100
# payloads the model really uses, plus context either side
SIZES = [
    ("MoE  3584 bf16", 3584, mx.bfloat16),
    ("attn 7168 bf16", 7168, mx.bfloat16),
    ("      64K bf16", 32768, mx.bfloat16),
    ("       1M bf16", 524288, mx.bfloat16),
]


def bench(n, dtype, group):
    x = mx.ones((n,), dtype=dtype)
    for _ in range(WARMUP):
        mx.eval(mx.distributed.all_sum(x, group=group))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(mx.distributed.all_sum(x, group=group))
    return (time.perf_counter() - t0) / ITERS


def main():
    g = mx.distributed.init()
    rank, N = g.rank(), g.size()
    host = socket.gethostname().split(".")[0]

    results = {}
    for label, n, dt in SIZES:
        results[label] = bench(n, dt, g)

    if rank == 0:
        print(f"[{host}] TP{N} all_sum latency over jaccl/RDMA")
        print(f"{'payload':>16} {'bytes':>9} {'latency':>10} {'eff GB/s':>10}")
        print("-" * 50)
        for label, n, dt in SIZES:
            t = results[label]
            nbytes = n * (2 if dt == mx.bfloat16 else 4)
            print(f"{label:>16} {nbytes:>9,} {t*1e6:>8.0f} us {nbytes/t/1e9:>9.2f}")

        attn = results["attn 7168 bf16"]
        moe = results["MoE  3584 bf16"]
        budget = 93 * attn + 92 * moe
        print()
        print(f"  per-token collective budget: 93 x attn + 92 x MoE")
        print(f"                             = {budget*1000:.1f} ms/token")
        print(f"  memory-bandwidth roofline  =  27.4 ms/token")
        print(f"  observed                   = ~500   ms/token")
        print()
        share = budget / 0.500 * 100
        print(f"  collectives explain {share:.0f}% of observed step time")
        if share < 40:
            print("  -> NOT collective-bound. Look at MoE gather/scatter + dispatch.")
        else:
            print("  -> collective-bound. Batching / spec-decoding amortise this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
