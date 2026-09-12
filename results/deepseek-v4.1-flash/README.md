# DeepSeek-V4.1-Flash on one M3 Ultra 512 GB

763B total parameters, 16B active at decode, `deepseek_v41`, served with oMLX 0.7.0.dev2 on a single
Mac Studio M3 Ultra. The build is `Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp`, 432.04 GB on disk, which
carries the vendor's own MXFP4 experts and MXFP8 attention through unrequantised. Only the two
Engram tables are requantised, to 4-bit affine at group 32.

**This set measures speed, memory and retrieval. It does not measure accuracy or perplexity.**

| File | What it holds |
|---|---|
| `prefill-ladder-20260912.csv` | prefill tok/s from 10K to 400K, Engram on SSD and in RAM, and with custom kernels off |
| `decode-engram-residency-20260912.csv` | decode tok/s across Engram residency and MTP state |
| `context-needle-20260912.csv` | needle retrieval at 32K, 128K, 256K and 400K |
| `mtp-acceptance-20260912.csv` | DSpark MTP draft acceptance against prompt length, from the engine's own telemetry |
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

**MTP draft acceptance falls with context.** Acceptance ran 92 to 96% on short prompts, 76.9% at
10K and 73.8% at 397K, with tokens per cycle falling from 3.70 to 2.54. The draft step itself costs
about 3% of the time, `backbone=3798ms` against `mtp=122ms`, so MTP is nearly free and the backbone
verify dominates. **Quote a decode figure for this model with its prompt length attached.**

**The 400K prefill is cacheable, so its cost is paid once.** The scheduler stored 397,312 of
397,604 tokens as a boundary snapshot with 193 intermediate snapshots, at a store cost of 7.2 ms and
a later lookup cost of 14.5 ms. So the 17.8-minute prefill is a one-time cost for a given prefix,
not a per-request cost, which is what makes a long session practical. Prompts under one 2048-token
block store nothing and report `boundary_snapshot_unavailable`.

**Deeper draft positions are what decay.** The acceptance drop is not uniform across MTP depth. On
short prompts the three draft positions accept at 93%, 95% and 88%. At 397K they accept at 94%, 69%
and 38%. So the first draft token survives long context almost untouched and the third does not,
which is why tokens per cycle falls while first-position acceptance holds.

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
