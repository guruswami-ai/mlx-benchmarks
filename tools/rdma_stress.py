#!/opt/omlx/venv/bin/python
"""Does a node drop out of sustained large collectives, with no model involved?

The Kimi-K3 TP4 prefill hang always strands the same HOST (established by
permuting rank order). The remaining suspect is that node's network-service
config. This tests it directly: no model, no mlx-lm, just escalating all_sum
traffic — so a failure here is node-level and model-independent, and a clean pass
moves suspicion back to the model/prefill path.

Escalates payload and iteration count so a failure localises. Every rank prints
its own progress, so the one that stops is visible immediately.

Run: mlx.launch --hostfile <hf> rdma_stress.py
"""
import socket
import sys
import time

import mlx.core as mx

# (elements, iterations) — bf16, so bytes = 2x elements
STAGES = [
    (3_584, 200),          # MoE-sized, the model's most common collective
    (7_168, 200),          # attention-sized
    (1_048_576, 100),      # 2 MB
    (16_777_216, 50),      # 32 MB — well beyond anything the model issues
    (67_108_864, 20),      # 128 MB
]


def main():
    g = mx.distributed.init()
    rank, N = g.rank(), g.size()
    host = socket.gethostname().split(".")[0]

    for n, iters in STAGES:
        nbytes = n * 2
        t0 = time.time()
        x = mx.ones((n,), dtype=mx.bfloat16)
        for i in range(iters):
            x = mx.distributed.all_sum(x, group=g)
            mx.eval(x)
            # keep values bounded so this stays a transport test
            x = mx.ones((n,), dtype=mx.bfloat16)
            if i and i % max(1, iters // 4) == 0:
                print(f"[r{rank} {host}] {nbytes/1e6:8.2f} MB  {i}/{iters}", flush=True)
        el = time.time() - t0
        total = nbytes * iters
        print(
            f"[r{rank} {host}] STAGE OK  {nbytes/1e6:8.2f} MB x{iters}  "
            f"{el:6.2f}s  {total/el/1e9:5.2f} GB/s",
            flush=True,
        )

    print(f"[r{rank} {host}] ALL STAGES PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
