# Running Kimi-K3 (2.78T) on four M3 Ultras with MLX

**Stack:** mlx 0.32.0, mlx-lm 0.31.3, macOS **26.6 (25G72)**, kernel 25.6.0, 4× Mac Studio M3 Ultra 512 GB,
Thunderbolt 5 full mesh, JACCL/RDMA backend.

## What this actually is

**This experiment was not about finding the ideal use case. It was about finding out whether it
is possible at all.**

Kimi-K3 is a 2.78 trillion parameter frontier model — 1.42 TiB at 4-bit. The question was whether
the current best Apple silicon, with the current best MLX, can run it end to end. It can. Everything
below is a report on what that took and what it costs, not a pitch.

## What it can do, concretely

On four M3 Ultra Mac Studios, today:

| | |
|---|---|
| Model | **Kimi-K3, 2.78 T params**, MXFP4 (4-bit), 1.42 TiB on disk |
| Runs as | TP4 — tensor parallel across 4 nodes, ~420 GB resident per node |
| Context | **Full 1,048,576 tokens resident**, 29 GB of KV cache |
| Load time | 229.6 s from local NVMe |
| Generation | **~2.0 tok/s** decode single-stream (1.3 tok/s end-to-end incl. TTFT) · **21.2 tok/s aggregate** at 32 concurrent |
| Prefill | ~170 tok/s chunked — a full 1M context takes ~1.7 hours, cached thereafter |
| Quality | Unmodified weights; the 4-bit repack is **bit-exact** to the published MXFP4 release |
| Power | **Measured: 559 W mean, 977 W peak** for all four under batched inference (58 W idle) |
| Efficiency | **14.7 Wh per 1000 tokens** generated |
| Footprint | Four desktop machines. No rack, no three-phase, no liquid cooling, no datacentre |

### Power, measured

This is the number that surprised us, so it is worth stating precisely. Sampled from each node's
`mactop` Prometheus exporter at 2 s intervals, across a full concurrency sweep:

| state | cluster total | per node |
|---|---|---|
| idle (freshly booted, 30 s sample) | **58.0 W mean / 66.0 W peak** | 14.2–14.9 W |
| model load (267 s) | 218 W mean / **503 W peak** | 90–180 W peak |
| batched inference | 559 W mean / **977 W peak** | 234–258 W peak (GPU 135–141 W) |

At idle the SoC is almost entirely infrastructure — `system` 12.6 W, `dram` 0.58 W, `cpu` 0.22 W,
`gpu` 0.05 W. **Quote the metric you mean:** GPU-only power reads *milliwatts* at idle and ~141 W
under load, while total SoC reads 14 W and ~250 W. Mixing the two is an easy way to be wrong by
two orders of magnitude.

⚠️ **These are SoC package figures, not wall draw.** They come from each node's `mactop`
Prometheus exporter, which reports on-die power. Actual consumption at the socket will be higher —
PSU conversion losses, fans, SSD and the Ethernet PHY are all outside this measurement. Expect
roughly 15–25 % more at the wall. The circuit-headroom conclusion survives that margin
comfortably, but **if you need a real wall figure, use a plug meter** — do not quote these as such.

**Peak 977 W for the entire cluster**, against Apple's 480 W rating *per machine* — real draw is
roughly half the nameplate. That fits inside a US 15 A / 120 V circuit (1.8 kW) or an
AU/EU 10 A / 230 V one (2.3 kW) with margin. SoC temperatures peaked at 69–76 °C on passive
desktop cooling. Energy cost worked out to **14.7 Wh per 1000 tokens** generated.

So **"a frontier-scale model on a standard wall socket" is measured, not aspirational.** A
comparable datacentre deployment is on the order of 10 kW with three-phase power and forced
cooling. That — not the speed — is the property that is genuinely hard to replicate any other way.

## Where it solves a real problem

Most people who need a 2.78 T model should rent one. It will be faster and dramatically cheaper.
The cases where that is not an option are narrow but real:

**Truly air-gapped inference.** Data that legally or contractually cannot leave the building —
medical records, legal discovery, unreleased source, classified or defence work, personal
archives. The alternative here is not "use the API more cheaply", it is "do not do this at all".
Cost-per-token comparisons don't apply, because there is no token to compare against.

**Sovereignty and continuity.** No vendor, no rate limit, no model deprecation, no terms change,
no outage. What you have keeps working.

**Where the context is the deliverable.** A whole codebase, an entire case file, a full research
corpus — held in one context, queried repeatedly against a warm cache.

And the workloads that suit it are the ones where **throughput beats latency**: test-time compute
scaling (best-of-N, self-consistency, tree search), multi-agent decomposition, document-set
processing, synthetic data generation, overnight queues. See *Where it is useful* below for why
batching is aligned with these rather than a limitation.

## Where it does not make sense

**2.78 trillion parameters is an enormous amount of model.** It demands more RAM, more memory
bandwidth and more storage than is justifiable when a cloud endpoint is available and permitted.
Four 512 GB Mac Studios is a serious sum of money to generate 1.3 tokens per second.

If your data *can* go to a cloud provider, it almost certainly should. This is not a cheaper way
to buy inference and it is not trying to be.

## What would change the picture

Nothing here is a physical limit. The model does **27 ms of real work per token inside a 500 ms
step** — the rest is per-operation overhead in the runtime. Concretely:

- **Dispatch overhead in MLX** — measured 6× above the memory-bandwidth ideal across every
  component, 53× on the smallest. This is the single biggest lever, and it is software.
- **Speculative decoding** — worth *more* on dispatch-bound hardware than on the datacentre GPUs
  it was built for. Currently refused outright by mlx-lm in distributed mode.
- **RDMA weight streaming** — would cut cluster storage from 5.7 TiB to 1.42 TiB.
- **Hardware** — each generation brings more unified memory and bandwidth. What needs four
  machines today plausibly needs two later.

Take the dispatch overhead alone and single-stream generation goes from ~1.3 tok/s toward the
36 tok/s roofline. That is not a speculative claim; it is the gap between measured work and
measured time, quantified below.

---

## Read this first: the limitations

I want these up front rather than buried, because the headline number is not the useful number.

**Single-stream generation is ~1.3 tok/s.** That is not a chat model on this hardware. If you
want conversational latency, nothing here will give it to you.

**The usable throughput requires concurrency.** 14.1 tok/s aggregate at 16 concurrent streams,
18.6 at 32 — but per-stream drops to 0.58 tok/s. You get throughput, not responsiveness.

**Prefill is ~97 tok/s end-to-end (~170 tok/s in the chunked phase).** Filling the full 1M
context takes on the order of 2–3 hours before you generate anything.

**Every node needs the whole model on local disk.** Tensor parallelism splits *within* each
tensor, so every rank touches every file — 1.42 TiB per node, **5.7 TiB across the cluster** for
one model. Pipeline parallelism does select files per rank, but see below for why PP is not an
option here.

**Load time is ~4 minutes** (229.6 s) from local NVMe.

**The RDMA stack is fragile, and macOS 26.6 did not touch it.** `AppleThunderboltRDMA` is
version **0.0.1** on both 26.5.2 and 26.6 — and the kext **UUID is byte-identical**
(`91CA73CD-328C-3DFA-BFA4-08E38D61EF83`), so the binary was not rebuilt at all. (`IORDMAFamily`
*was* recompiled — its UUID changed — but the RDMA driver itself was not.) Protection domains are
kernel-allocated and never reclaimed on process exit, so **2-3 broadcasts exhaust the pool and
only a reboot recovers**. Budget your distributed operations per boot; a warm-up transfer is not
free.

**Distributed jobs still deadlock, and `FAST_SYNCH=0` does not prevent it.** We hit a hard hang on
26.6 with `MLX_METAL_FAST_SYNCH=0`, on a modest ~1100-token prompt, immediately *after* a full
32-way concurrency sweep had completed cleanly. All four ranks alive, mesh intact, no `bridge0`.
The signature differs from the `FAST_SYNCH=1` wedge and is worth learning to recognise:

| | `FAST_SYNCH=1` wedge | `FAST_SYNCH=0` deadlock |
|---|---|---|
| ranks 1-3 | ~100% CPU, GPU idle at 4 W | GPU 100% *util* but only 41-50 W |
| remaining rank | — | CPU 100%, GPU 7%, 25 W |
| clears without reboot? | no | yes (kill was enough) |

**In both cases power is the giveaway.** GPU utilisation reads 100% while drawing a fraction of
the 234-258 W a genuinely working node pulls. Watch watts, not utilisation.

**No speculative decoding.** `mlx_lm.server` refuses draft models in distributed mode outright.

**Hardware cost is absurd for what you get.** Four M3 Ultra 512 GB machines. As a general-purpose
inference box this is not defensible on price. That is a fair thing for the community to know.

---

## Where it *is* useful

The interesting property is not speed, it's **capacity**.

**The full 1M context fits, with room to spare.** Kimi-K3's KV cache costs **27 KB/token**, so
1,048,576 tokens is **29.0 GB** against 42.3 GB of headroom after weights. Two architectural
choices make that possible:

- **69 of 93 layers are KDA linear attention** — their state is **constant** in sequence length
  (109 MB total, regardless of context).
- The 24 MLA layers cache a **512-d latent** rather than materialised heads. Without that
  absorption you'd need ~1.18 MB/token → **1.18 TB** for 1M context. Flatly impossible.

So the use case this hardware serves is: **a frontier-scale model chewing on a hard problem with
an enormous context, for a long time, with many reasoning streams in flight.** Latency-tolerant,
throughput-sensitive, context-hungry.

**This is a technological feat, not a product.** It is not a chatbot, and pretending otherwise
would waste people's money.

### What it's genuinely good for

**1. Data that cannot leave the building.** This is the strongest justification and it isn't a
performance argument at all. Medical records, legal discovery, unreleased source, defence work,
personal archives — frontier-scale reasoning over a million tokens of it, entirely local. The
alternative is not "use the API more cheaply", it's "don't do this at all". Cost-per-token
comparisons simply don't apply.

**2. The context *is* the deliverable.** Whole-codebase reasoning, an entire case file, a complete
research corpus. 1M tokens resident in 29 GB is rare — most local setups can't hold a
frontier-scale model at all, let alone with the context filled.

**3. Test-time compute scaling — and this is why batching is not a limitation.** The tasks that
justify a 2.78 T model are hard problems, and hard problems are exactly where **parallel sampling
buys quality**: best-of-N, self-consistency, tree search, multi-agent debate. Those workloads are
batched *by construction*. 32 streams at 0.58 tok/s each is precisely the shape of a
self-consistency run — you are converting throughput into answer quality, which is the trade you
want on a hard problem.

So "batching is required" and "this model is for hard problems" point the same direction. The
natural consumers:

- **Test-time compute** — sample N reasoning traces, select the best
- **Multi-agent decomposition** — N agents on sub-problems
- **Document-set processing** — one stream per document
- **Synthetic data generation, eval harnesses** — inherently batched, latency-irrelevant
- **Overnight queues** — submit at night, collect in the morning

### The 1M context, honestly

Prefill runs at ~170 tok/s in the chunked phase, so a **full 1M-token prefill takes ~1.7 hours**.
That sounds fatal until you account for prompt caching: **if the corpus is the task, you pay it
once.** "Load the case file at 9am and interrogate it all day" works. "Paste 1M tokens with every
query" does not.

(Chunking is already handled upstream — the server prefills in 2048-token chunks. It is not
something you need to implement.)

### What it's bad at

Interactive chat. Anything with a human waiting. Fast tool-use loops where you iterate quickly.
And short queries — with a ~4 minute load and seconds of fixed per-request overhead, asking this
thing a one-liner is pure waste.

---

## The hardware, and how it's tuned

Four identical Mac Studios, dedicated to inference — no desktop session, no other services.

| | |
|---|---|
| Model | Mac Studio `Mac15,14`, Apple **M3 Ultra** |
| CPU | 32 cores (24 performance + 8 efficiency) |
| GPU | **80 cores**, Metal 4 |
| Memory | **512 GB** unified, ~819 GB/s |
| Storage | `muladhara` **8 TB** NVMe · other three **2 TB** each |
| Mesh | Thunderbolt 5 **full mesh** — 3 cables/node, 6 links, 80 Gb/s each, MTU 9000 |
| Management | 10 GbE per node (NAS, ssh) |

**Roles.** `muladhara` is the staging node — the 8 TB disk holds downloads, conversion scratch and
the master copy, and it acts as TP rank 0 / coordinator. The other three are pure workers. That
asymmetry matters: converting a 2.5 TB source into a 1.56 TB output needs ~4 TB free on one
machine, and the 2 TB nodes cannot do it.

**Tuning applied** (all persistent via LaunchDaemons, all verified after every reboot):

```
iogpu.wired_limit_mb    = 499712     # 488 GB wired ceiling -- without this the
                                     # 420 GB model gets paged and never loads
MLX_METAL_FAST_SYNCH    = 0          # global launchctl setenv; see the warning below
kern.ipc.maxsockbuf     = 33554432   # 32 MB
net.inet.tcp.sendspace  = 16777216   # 16 MB
net.inet.tcp.recvspace  = 16777216
kern.maxfiles           = 65536
MTU 9000 on all mesh interfaces
```

**Inference-only posture.** Anything that competes for RAM or GPU is off:

- **18 inference/voice/serving daemons disabled** with `launchctl disable` — not merely
  `bootout`, because that does not survive a reboot and TP4 bring-up forces several. This took
  each node from 230–415 GB of resident model weights down to ~9 GB at boot.
- **Headless**: no autologin, console owned by root, screen sharing off. A logged-in desktop
  session costs GPU and wired memory you need.
- **Kept running deliberately**: `bridge-killer` (destroys macOS's auto-created Thunderbolt
  `bridge0` every 5 s — in a full mesh the topology is a loop, STP blocks links, and RDMA dies),
  `tb5-init` (restores MTU 9000), `iogpu-wired-limit`, `boot-net-selfheal`, plus lightweight
  telemetry.

**Disk allocation.** This is the ugly part. TP needs the **whole** model on **every** node:
1.42 TiB × 4 = **5.7 TiB** for a single model. On the 2 TB workers that leaves ~140 GB spare, and
we had to clear every other model off the cluster to fit it. See the RDMA section below for why
this should not be necessary.

## The work required

### 1. Weight conversion: MXFP4 → MLX

The published quantisations are `compressed-tensors` **`mxfp4-pack-quantized`**: E2M1 nibbles with
E8M0 scales, group size 32. MLX has a native `mxfp4` mode, so this is a repack rather than a
requantisation — bit-exact, no quality loss (`max|diff| = 0.000e+00` on real K3 tensors).

The transform is small: `weight_scale` → `scales`, and `weight_packed` (uint8 nibble pairs) →
uint32 words:

```python
b = value.reshape(*value.shape[:-1], -1, 4).astype(mx.uint32)
packed = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
```

then declare `{"group_size": 32, "bits": 4, "mode": "mxfp4"}`. Conversion of the full 2.78 T model
took **8.5 minutes** and produced 1.5608 TB across 278 shards (0.561 bytes/param including scales).

### 2. `shard()` — the actual blocker

The MLX port of K3 ([PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx)) had no
distributed support at all. Writing `shard()` was the real work, and **two of the three bugs I hit
run without raising anything.**

Attention splits by head; FFN and expert stacks split on the intermediate dim. Experts are ~2.72 T
of the 2.78 T total, so sharding `moe_intermediate_size` is where essentially all the memory
saving lives.

**Trap 1 — `A_log` is `[head_dim]`, not `[num_heads]`.** A KDA block contains four vectors sized
96 or 128, and **only the 96 one may be sharded**:

```
A_log            [128]        head_dim   -> REPLICATE
o_norm.weight    [128]        head_dim   -> REPLICATE
f_a_proj.weight  [128, 7168]  head_dim   -> REPLICATE
b_proj.weight    [96, 7168]   num_heads  -> SHARD
dt_bias          [12288]      96*128     -> SHARD as (heads, head_dim)
```

128 divides by 4, so a head-wise slice of `A_log` *succeeds* — and then broadcasts the wrong decay
curve onto every head. No error, no shape mismatch, just degraded output. (Credit to PipeNetwork
for documenting this; I confirmed it against the checkpoint.)

**Trap 2 — the MoE all-reduce must precede the norm.** K3 sets `latent_moe_use_norm=True`. Each
rank's expert output is a *partial sum*, and RMSNorm does not commute with the reduction.
DeepSeek-V3's MoE — the obvious thing to copy, and what mlx-lm ships — all-reduces at the *end* of
the block. Porting that pattern gives **4.9 % relative error and no exception**: a model that
loads, runs, and produces confident wrong text. The weighted expert sum *is* linear, so it can
stay before the reduce.

**Trap 3 — `Conv1d.groups` is not a parameter.** `shard_inplace` rewrites the parameter tree;
`groups` is a plain Python attribute and survives untouched, so the depthwise conv claims 12288
groups against 3072 channels. This one fails loudly. General lesson: **any module holding shape
metadata outside its parameter tree needs fixing by hand after sharding.**

**Validate numerically — "it ran" proves nothing.** Two of those three produce running code. I
built a tiny model with the *real* `head_dim=128` (the gated-delta Metal kernel is specialised for
128 and won't compile below it), randomised **every** parameter first — `A_log` and `dt_bias`
initialise to zeros, and zeros make the decay uniform across channels, hiding trap 1 entirely —
then compared an unsharded reference pass against the sharded one:

```
TP2   max|diff| = 5.215e-08   rel = 3.029e-07   PASS
TP4   max|diff| = 3.725e-08   rel = 2.164e-07   PASS
TP8   max|diff| = 3.725e-08   rel = 2.164e-07   PASS
```

Plus a **negative control** that deliberately reintroduces the DeepSeek ordering and requires it
to be *detected* (rel = 4.885e-02). A test that cannot fail is worthless.

Both run as **N ranks on a single host over the ring backend** — full sharded forward pass, zero
RDMA budget spent. Given PD exhaustion, that's what made iteration possible at all.

### 3. Distributing 1.42 TiB

`mlx_lm.share` is a **true broadcast**, not N point-to-point copies — it moves each chunk with
`all_sum`, source contributing real bytes and peers contributing zeros. **Copying to 3 peers takes
about what copying to 1 takes.** Quote it as a wall-clock rate, never "per peer / aggregate".

Expected ~5.2 GB/s. We got 1.31 GB/s, because a 31 GB calibration transfer ran first and spent PD
budget. **Broadcast first after a clean boot.** A warm-up costs you 4×.

Also worth knowing: rank 0 reads in place (no temp copy, peak usage is 1× not 2×), and `--tmpdir`
*is* forwarded to peers.

### 4. Serving

`mlx_lm.server` already handles distributed TP — it calls `sharded_load` with a `tensor_group` and
only rank 0 binds the socket. One patch was needed: **K3 has no Jinja chat template.** Moonshot
renders chat in Python (`encoding_k3.build_chat_segments`). mlx-lm gates on
`tokenizer.chat_template is not None`, finds nothing, and silently falls back to naive
`role: content` concatenation — which is not K3's format at all (the real one is an XTML structure
with `<|open|>tag attrs<|sep|>` and `<|end_of_msg|>`). The renderer is present and
signature-compatible; you just have to flip the gate after *proving* it works on a probe message.

One more: `sharded_load` takes **one path for all ranks**, so the model must live at an identical
path on every node.

---

## Performance: what actually bounds this

The interesting result is that **none of the obvious suspects are the bottleneck.**

The memory-bandwidth roofline says this should be fast. Only **103 B of 2.78 T params are active**
per token; at TP4 each node reads **22.5 GB/token**, which at 819 GB/s is **27.4 ms → 36 tok/s**.
Observed was ~500 ms/token. An **18× gap.**

| suspect | measured | verdict |
|---|---|---|
| collectives | 5.0 ms/token | **1 %** — not it |
| MoE gather | 34.0 ms/token | **7 %** — not it |
| fixed per-call overhead | ~170 ms/token | **the problem** |

**Collectives are fine.** `all_sum` over TB5 RDMA/JACCL is **26 µs** at 7–14 KB, and 4.96 GB/s at
1 MB. Microsecond latency is exactly the win over TCP/IP; 186 collectives/token costs 5 ms. The
expensive, hard-won part of the setup is not the problem.

**The problem is a fixed floor of ~200–400 µs per module call**, regardless of how much work the
module does:

| component | params | bw-ideal | measured | overhead |
|---|---|---|---|---|
| router gate | 6.4 M | 4 µs | 234 µs | **53×** |
| latent down+up | 51 M | 35 µs | 478 µs | 14× |
| KDA attention | 115 M | 79 µs | 426 µs | 5× |
| MoE experts | 132 M | 91 µs | 370 µs | 4× |
| MLA attention | 230 M | 158 µs | 462 µs | 3× |

**Kimi-K3 at batch 1 is dispatch-bound.** The MoE sparsity is delivering exactly what it promises —
we simply can't exploit it, because per-op overhead swamps the work.

*(Caveat: each module was timed with `mx.eval` around it, a sync the real model doesn't pay
per-module, so ~170 ms is an upper bound and ~330 ms of the 500 ms remains unattributed. My
leading hypothesis is the 186 collectives acting as pipeline-drain barriers — cheap in isolation
but forcing sync points that prevent dispatch overlapping execution. Untested.)*

### Batching confirms it

Fixed overhead is precisely what batching amortises:

| concurrency | agg tok/s | per-stream |
|---|---|---|
| 1 | 1.32 | 1.32 |
| 4 | **5.00** | 1.25 |
| 16 | **14.11** | 0.88 |
| 32 | **18.56** | 0.58 |

Concurrency 4 is the proof: **4× the work in 38.4 s vs 36.4 s for one.** Nearly free. No patch
required — `mlx_lm.server` batches whenever every cache implements `merge()`, and K3's `KVCache`
and `ArraysCache` both do.

---

## What would move the needle

**Speculative decoding is the big one, and it's worth more here than on datacentre GPUs.**
[`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark) drafts **7 tokens in a
single parallel pass**. On a dispatch-bound system, verifying 7 tokens in one target forward
attacks precisely the dominant cost — where on a bandwidth-bound GB300 it's a smaller win. Three
blockers: DSpark consumes the target's auxiliary hidden states from layers 2/23/47/71/89 (fused
into the forward pass, not a standalone draft), it needs vLLM's `dspark` method and
`FLASHINFER_MLA`, and **mlx-lm refuses draft models in distributed mode**. K3 also ships no MTP
head (`num_nextn_predict_layers: 0`), so a separate draft is the only route.

**Reducing per-op dispatch overhead in MLX.** A 6× gap between measured and bandwidth-ideal across
every component, and 53× on the smallest. This is the single biggest lever for single-stream
latency and it lives inside MLX, not in model code.

### RDMA weight streaming — the fix for 5.7 TiB of duplicated disk

The disk requirement is the least defensible part of this setup, and it is **not fundamental**.

The key observation: **local disk is only touched during load.** Once `mx.eval(model.parameters())`
completes, all 420 GB per rank is resident in unified memory and the SSD is never read again for
the lifetime of the process. A hot model needs no disk at all. So a rank does not need a *copy* of
the weights — it needs a *path to receive them once*.

Today `sharded_load` calls `load_model(path, lazy=True)`, which mmaps local safetensors. There is
no way to source parameters from anywhere else. Hence: every node keeps 1.42 TiB it reads once and
then ignores forever.

**What would replace it:** stream each rank's slice from one node's SSD directly into the peers'
memory over RDMA at load time.

| | today | with RDMA streaming |
|---|---|---|
| Cluster disk for one model | **5.7 TiB** (1.42 × 4) | **1.42 TiB** (staging node only) |
| Pre-staging broadcast | 1.42 TiB, ~5 min at spec (we saw 20) | none |
| Load | 230 s from local NVMe | ~260–300 s over RDMA |
| Peer disk needed | 1.42 TiB | **zero** |

It is not faster — one SSD sourcing 1560 GB at ~6 GB/s is roughly the current load time — but it
**eliminates the separate distribution step entirely** and frees 4.3 TiB across the workers. On
2 TB nodes that is the difference between "one model, nothing else" and a usable machine.

**Is this a TP limitation?** Only partly, and the distinction matters:

- Pipeline parallelism *already* computes which files a rank needs and fetches only those.
- Tensor parallelism has **no file selection at all** — every rank lazily loads the full path and
  then calls `model.shard()`. The laziness keeps *memory* at 1/N (mmap faults in only the slice
  each rank keeps), but every rank must still be able to *open every file*.
- That is an implementation gap, not a property of TP. TP slices are **contiguous within each
  tensor**, and safetensors supports partial reads, so "read only my rows" is expressible today.

So there are two separable asks, and the first is much easier:

1. **TP slice selection** — let a TP rank read only its own rows/columns. This alone would cut
   what each rank must *read* from 1.42 TiB to ~390 GB, though it still requires file access.
2. **A non-mmap weight source** — let parameters be populated from a network/RDMA provider rather
   than a local file. This is what removes peer disk entirely.

Relevant upstream: [ml-explore/mlx#3208](https://github.com/ml-explore/mlx/issues/3208).

**One caveat on "hot model stays in memory".** True while the process lives — but PD exhaustion
forces reboots more often than you would like on this stack, and every reboot is a full reload. The
load path is therefore hit more than the "load once, serve forever" framing suggests, which makes
fixing it more valuable, not less.

**Further quantisation buys less than you'd think.** It's already MXFP4, and MXFP4 is the *source*
encoding so it cost nothing. Memory is not the binding constraint (420 of 488 GB used, and 1M
context already fits), and we're not bandwidth-bound, so lower bits won't buy tokens/s. The one
genuinely interesting angle is **fewer machines**: at ~0.31 bytes/param a 2-bit build would be
~0.86 TB, which could make **TP2 on two nodes** viable and halve the hardware. Whether a 2-bit
2.78 T model is still worth running is an open question, but it's the quantisation direction that
would actually change accessibility.

**Parallelism options are narrower than they look.** Only TP4 fits: TP2 needs ~790 GB/node, PP2/PP3
need ~780/~520 GB, all over the 488 GB ceiling. PP4 fits on memory but Metal's ~60 s command-buffer
timeout is documented fatal at 405B-class, and K3 is 7× that. A **TP2×PP2 hybrid** (~390 GB/node)
is the one untried configuration — `sharded_load` accepts both groups — but it needs a `pipeline()`
implementation and still risks the timeout.

---

## Should the converted model go on Hugging Face?

Honestly: **the patch and the recipe are more valuable than the weights.** It's 1.5608 TB to host,
and anyone with hardware to run it can convert from the published FP8/MXFP4 source in 8.5 minutes.
What's worth publishing is the `shard()` implementation, the validation harness, and the
conversion path — which are small, and which are what people actually get stuck on.

A lightweight repo with the modeling code, config, conversion script and validation tests would
serve better than 1.5 TB of weights that duplicate an existing upload.

---

## Reproducing

Per boot, in this order — the order matters:

```bash
# 1. Clean boot. Capture the GPU baseline BEFORE any MLX job:
#    a clean M3 Ultra reads 0.03-0.2 W. ~4 W means something is spinning.
sudo powermetrics -n1 -i300 --samplers gpu_power | grep "GPU Power"

# 2. Validate the mesh without changing it (safe, repeatable)
mlx.distributed_config --over thunderbolt --hosts <n1,n2,n3,n4> --dot

# 3. Configure -- EXACTLY ONCE PER BOOT. Running it twice corrupts ARP/RDMA
#    mappings and requires a reboot.
sudo mlx.distributed_config --over thunderbolt --hosts <...> \
     --backend jaccl --auto-setup --output-hostfile <hostfile>

# 4. (optional) MTU 9000 on the TB interfaces.
#    NOT required for RDMA: measured 9.05 vs 8.93 GB/s at 134 MB between MTU
#    9000 and 1500 -- within noise, because RDMA uses neither TCP/IP nor
#    Ethernet frames. mlx.distributed_config does NOT reset MTU either (no
#    `mtu` anywhere in its source; interfaces read 9000 immediately after
#    --auto-setup). Keep this only if you also use the `ring` backend over TB.
sudo tb5-init.sh --mtu-only     # on every node

# 5. Broadcast the model FIRST, before any other RDMA op
mlx_lm.share --path <model-DIR> --hostfile <hostfile> --dst <path> --tmpdir <same-fs>

# 6. Serve
mlx.launch --hostfile <hostfile> <server-script>
```

**Keep `MLX_METAL_FAST_SYNCH=0`.** Setting it to `1` is marginally *slower* (31 µs vs 26 µs) and it
wedges the fleet: ranks pin ~100 % CPU with an idle GPU, and it survives killing the workload.
It took an 8505-token prefill to trigger — 97/1299/2205-token runs all passed — so **short tests
give false negatives.** `FAST_SYNCH=1` busy-polls instead of sleeping, so a stalled sync appears as
spinning CPU, not a busy GPU. Diagnose by **watts, not GPU residency**: residency reads 100 % while
the GPU does nothing.

---

## Open questions for MLX maintainers

1. **PD lifecycle.** Protection domains appear never to be released on process exit, so 2–3
   broadcasts per boot exhaust the pool with reboot as the only recovery. Is there an intended
   release path?
2. **Teardown hang after a *successful* transfer.** All data landed and renamed on every peer, yet
   rank 0 stayed alive >20 min. The forced kill then leaves the stale state that causes (1) and (3).
3. **Unkillable ranks.** Wedged ranks sit in `Us+` uninterruptible wait; `SIGKILL` does nothing.
4. **`MLX_METAL_FAST_SYNCH` semantics.** What does it change, and is there a configuration where
   the collective path and the spin-wait issue are both satisfied?
5. **Draft models in distributed mode.** Currently refused outright. Given speculative decoding is
   worth *more* on dispatch-bound hardware, this is the highest-value gap.
6. **Per-op dispatch overhead.** 6× over bandwidth-ideal across every component, 53× on the
   smallest, at batch 1.
7. **TP file selection.** PP already computes which files a rank needs. Since safetensors supports
   partial reads and TP slices are contiguous within each tensor, could TP ranks read only their
   slices — turning 4 × 1.42 TiB of disk into 4 × ~355 GiB?

---

## Summary

A 2.78 T frontier model runs on four Mac Studios, with its **full 1M context resident in 29 GB**,
at **14–18 tok/s aggregate** across concurrent streams. Single-stream is ~1.3 tok/s and that is
not going to change without work inside MLX.

Almost nobody has this hardware, and at current prices it isn't a sensible way to buy inference.
What it does show is that the *architecture* — MoE sparsity plus linear attention plus latent KV
compression — maps unusually well onto large unified memory, and that the remaining gap is
**quantified and addressable overhead**, not a physical limit. The model is doing 27 ms of real
work per token in a 500 ms step.
