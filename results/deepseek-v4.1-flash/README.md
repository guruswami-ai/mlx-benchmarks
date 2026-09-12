# DeepSeek-V4.1-Flash on one M3 Ultra 512 GB

763B total parameters, 16B active at decode, `deepseek_v41`, served with oMLX 0.7.0.dev2 on a single
Mac Studio M3 Ultra. The build is `Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp`, 432.04 GB on disk, which
carries the vendor's own MXFP4 experts and MXFP8 attention through unrequantised. Only the two
Engram tables are requantised, to 4-bit affine at group 32.

**This set measures speed, memory and retrieval. It does not measure accuracy or perplexity.**

The authoritative performance numbers are `omlx-benchmark-20260912.csv` and
`continuous-batching-20260912.csv`, from the oMLX GUI benchmark. The ad-hoc files record probes
that were run first and are kept because they cover conditions the benchmark does not: Engram
residency, kernel fallback, and needle retrieval to 400K. Note both sources report prefill as
`accounting=prompt_estimate`, which is TTFT divided by prompt tokens, not a kernel-level measurement.

| File | What it holds |
|---|---|
| `prefill-ladder-20260912.csv` | prefill tok/s from 10K to 400K, Engram on SSD and in RAM, and with custom kernels off |
| `decode-engram-residency-20260912.csv` | decode tok/s across Engram residency and MTP state |
| `context-needle-20260912.csv` | needle retrieval at 32K, 128K, 256K and 400K |
| `mtp-acceptance-20260912.csv` | DSpark MTP draft acceptance against prompt length, from the engine's own telemetry |
| `omlx-benchmark-20260912.csv` | the oMLX GUI benchmark, 1K to 200K: TTFT, TPOT, prefill, decode, peak memory |
| `continuous-batching-20260912.csv` | batch 1 to 8: aggregate decode, per-request prefill, TTFT |
| `prefix-cache-20260912.csv` | boundary cache snapshot sizes and timings, including the 400K store |
| `memory-footprint-20260912.csv` | `ri_phys_footprint` at load and at peak, both Engram modes |
| `kernel-comparison-20260912.csv` | native custom kernels against the pure-MLX fallback |

## Four findings

**Engram residency is worth 40% of decode and nothing to prefill.** Moving the two Engram tables
from a memory-mapped SSD file into RAM took decode from 25.75 to 36.2 tok/s. Prefill at 400K was
373 tok/s either way, and the 400K wall time differed by 0.4 seconds in 1,067. That split is the
expected shape: Engram is a per-token sparse gather, so it costs decode and barely touches
compute-bound prefill. The cost is 114.44 GiB of residency. Quality does not change, because the
tensors and their 4-bit values are identical.

**KV cache costs 890 bytes per token, and it shows.** Peak footprint rose 1.64 GiB across a
397,555-token prefill with Engram resident, and 1.28 GiB across 397,552 tokens with Engram on SSD.
The prefill transient is bounded rather than proportional to context, so peak does not scale with
length. Needle retrieval passed at every length tested, with a fresh UUID in each prompt and
`cached_tokens` 0 in every record, so no result is a prefix-cache replay.

**Custom kernels are worth 2.3x on prefill and nothing on decode.** The native
`deepseek_v41_packed_attention` path gave 454 tok/s at 10K against 196 on the pure-MLX fallback.
Decode was unchanged at about 31 tok/s, which is consistent with prefill being compute-bound and
decode being bound elsewhere. The kernels are opt-in: `setup.py` gates the build on
`OMLX_WITH_CUSTOM_KERNEL`, and the Metal compiler ships only with full Xcode, never with Command
Line Tools.

**MTP draft acceptance is driven by content, not context length. CORRECTED.** An earlier version of
this file claimed acceptance falls with context. The oMLX benchmark run contradicts it: **96.8%
acceptance at 200,000 tokens** with 3.76 tokens per cycle and third-position drafts at 28 of 29,
against **63.6 to 81.7% at 1,024 tokens** with third-position drafts as low as 4 of 7, on the same
model in the same run. Acceptance is *higher* at 200K than at 1K.

The original claim came from comparing a short counting task against long repeated filler, so prompt
content and context length varied together and the difference was attributed to the wrong one. Across
all ten observations acceptance ranges 63.6 to 96.8% with no monotonic relationship to prompt length.
`mtp-acceptance-20260912.csv` carries a `content` column so the confound is visible.

What does hold: the draft step costs about 3% of the time, `backbone=3372.7ms` against `mtp=85.9ms`,
so MTP is nearly free and the backbone verify dominates. **Quote a decode figure with both the prompt
length and the content type**, because either can move it.

**Decode does not degrade with context.** The benchmark's TPOT sits in a 26.0 to 31.7 ms band from 4K
to 200K, a mean of 28.24 ms or 35.4 tok/s, with the best result 38.7 tok/s at 32K. The 1K and 8K rows
are outliers at 34.56 and 44.12 ms.

**Continuous batching scales to 7.38x.** Aggregate decode goes 29.2 to 215.4 tok/s from batch 1 to
batch 8. Per-request prefill collapses from 354.2 to 19.7 tok/s and TTFT grows from 2.9 to 27 s, so it
suits a shared endpoint and not a single interactive session. Ad-hoc concurrent HTTP requests do **not**
batch: an earlier test of four simultaneous requests showed no gain, because they never align into a
batch. Only the continuous-batching path aligns them.

**The Apple Neural Engine is idle.** `ane0_duty=0.0000`, `ane1_duty=0.0000`, `full_ane_tiles=0`,
`gpu_tail_tokens=199999` for both the mlp and gdn categories at 200K. Everything runs on the GPU.

## Read this before comparing

**Single runs, except decode.** Decode at Engram RAM is two runs; everything else is one. Treat
differences under about 10% as unresolved. The 128K and 256K prefill figures are 376 and 406 tok/s,
which are not monotonic, and that is the measurement noise rather than a real effect.

**Prompt provenance.** Prefill and needle prompts are a repeated filler sentence with a needle at
62% depth and a UUID prefix. Repeated filler produces few distinct n-grams, so it exercises a
smaller Engram working set than natural text would. **The Engram-on-SSD footprint figures are
therefore optimistic.** The Engram-in-RAM figures are unaffected, because there is no working set to
understate.

**Requests serialise.** `max_concurrent_requests` was raised from 1 to 4 and aggregate throughput
did not change. The cause is in the engine: DSpark sets `_omlx_mtp_rowwise_unsupported` and
verification rejects multiple rows, so MTP and batched decode are mutually exclusive for this model.
All figures here are single-stream.

**The memory guard never engaged.** Zero throttle events across every run, at a ceiling of 488 GB.
Peak with Engram resident was 405.14 GiB against a soft watermark of 414.8, so 9.7 GiB spare. The
guard has no model-specific prefill profile for `deepseek_v41`: `memory_monitor.py` excludes it
explicitly. That turns out not to matter here, because the generic estimator prices from KV length
and this KV is small.

## Not measured

Perplexity, MMLU-Pro and the other accuracy suites. Those need a harness run and belong in a later
dated file. No comparison with `../perplexity-all-models.csv` is possible from this set.

No second build was measured. `oQ3e-mtp` exists at 355.30 GB and 3.724 bits per weight, and nothing
above oQ4e exists: oMLX refuses any `oq_level` outside 3 and 4, and the vendor already ships FP4
experts, so more bits would store identical values in more bytes.

## Method

oMLX 0.7.0.dev2 at commit `b390b31`, mlx and mlx-metal 0.32.2, mlx-lm 0.31.3, macOS 26.6.2 build
25G83. Custom kernels built with Xcode 26.6 and ABI-verified through the extension's own
`abi_probe`. `MLX_METAL_FAST_SYNCH=0`. Memory guard 488 GB, `max_concurrent_requests` 1 except where
stated. Footprint read through `proc_pid_rusage(RUSAGE_INFO_V4)` offset 72, because resident set
size does not account for Metal buffers and under-reports by more than half. Prefill rates are
derived as `prompt_tokens / (wall - output_tokens / decode_rate)`, so they inherit the decode
figure's uncertainty.

For comparison on the same node, GLM-5.3 mixed 4/8-bit measured 12.2 tok/s decode, 162 tok/s prefill
at 10K, and 405.6 GiB resident. Its own results are in `../glm-5.3/`.
