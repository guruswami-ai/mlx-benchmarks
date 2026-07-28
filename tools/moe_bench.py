#!/opt/omlx/venv/bin/python
"""Where does Kimi-K3 TP4's decode time actually go?

Established by measurement:
    collectives          5.0 ms/token   (1%)
    memory roofline     27.4 ms/token   (5.5%)
    observed           ~500   ms/token
    unexplained        ~468   ms/token  (93.5%)

Hypothesis under test: the MoE gather at batch 1. Each MoE layer routes top-16 of
896 experts, so a decode step does 92 x 16 = 1472 matrix-VECTOR products, each
against a different scattered expert slice with ZERO weight reuse. Scattered
matvec gets nowhere near the 819 GB/s peak.

If true, batching is the fix and the payoff is large: the same expert weights
serve many tokens, turning matvec into matmul with reuse.

Single node, no mesh needed. Dims are K3's TP4 per-rank shapes.
"""
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.switch_layers import SwitchGLU

MOE_HIDDEN = 3584           # routed_expert_hidden_size
INTER_LOCAL = 3072 // 4     # moe_intermediate_size sharded across TP4
N_EXPERTS = 896
TOP_K = 16
N_MOE_LAYERS = 92


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print(f"SwitchGLU  experts={N_EXPERTS}  in={MOE_HIDDEN}  hidden={INTER_LOCAL} "
          f"(TP4 per-rank)  top_k={TOP_K}")
    # SwitchGLU's activation is a GLU: it takes (up, gate), not one arg.
    moe = SwitchGLU(MOE_HIDDEN, INTER_LOCAL, N_EXPERTS,
                    activation=lambda up, gate: nn.silu(gate) * up)
    nn.quantize(moe, group_size=32, bits=4, mode="mxfp4")
    mx.eval(moe.parameters())

    nbytes = sum(v.nbytes for _, v in tree_flatten(moe.parameters()))
    print(f"one layer's expert bank = {nbytes/1e9:.2f} GB\n")

    print(f"{'batch':>6} {'ms/call':>9} {'x92 layers':>12} {'tok/s ceiling':>14} {'per-token ms':>13}")
    print("-" * 60)
    for B in (1, 4, 16, 64, 256):
        x = mx.random.normal((B, MOE_HIDDEN)).astype(mx.bfloat16)
        idx = mx.argpartition(
            mx.random.normal((B, N_EXPERTS)), kth=TOP_K, axis=-1
        )[:, :TOP_K]
        mx.eval(x, idx)
        t = timeit(lambda: moe(mx.expand_dims(x, 1), idx))
        layer_total = t * N_MOE_LAYERS
        per_tok = layer_total / B
        print(f"{B:>6} {t*1000:>9.2f} {layer_total*1000:>11.1f}ms "
              f"{1/per_tok:>13.1f} {per_tok*1000:>12.1f}")

    print("\n(tok/s ceiling counts ONLY the MoE expert layers -- attention,")
    print(" shared experts, norms and collectives are all on top of this.)")


if __name__ == "__main__":
    main()
