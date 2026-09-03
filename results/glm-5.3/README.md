# GLM-5.3 on one M3 Ultra 512 GB

744B MoE (40B active), `glm_moe_dsa`, served with oMLX 0.6.4 on a single Mac Studio M3 Ultra.
Two builds measured: uniform 4-bit and mixed 4/8-bit (experts 4-bit, everything else 8-bit),
each with the multi-token-prediction head grafted back from the BF16 release. The mixed build
ships. Full recipe, overlay shard and scripts:
https://huggingface.co/guruswami-ai/glm-5.3-mlx-mixed-4_8bit-mtp-recipe

| File | What it holds |
|---|---|
| `accuracy-20260903.csv` | oMLX accuracy benchmark, two greedy runs, five suites, sampled |
| `mmlu-pro-categories-20260903.csv` | MMLU-Pro per category, both runs |
| `prefill-ladder-20260902.csv` | prefill tok/s and TTFT, 1K to 32K, both builds, cold, cached 0 |
| `decode-20260902.csv` | 400-token decode with and without the MTP head |
| `context-ladder-20260902.csv` | needle retrieval 64K to 230K on the mixed build, memory guard on |
| `perplexity-wikitext2-20260902.csv` | wikitext-2, 2048-token windows, our harness and the publisher's |

Headline: perplexity 2.823 mixed against 2.948 uniform 4-bit on the same harness; needle
retrieved at 234,545 tokens with zero guard trips; prefill 191 tok/s at 1K and 86 at 230K;
decode 20.5 tok/s with the head at depth 1 (23.8 on uniform 4-bit); MMLU-Pro 85 %, GSM8K
97 %, TruthfulQA 92 %, LiveCodeBench 58 to 63 %, SafetyBench 84 to 86 % on sampled sets.

Two things to read before comparing: the chat template opens every assistant turn with
`<think>` and ignores the harness thinking flag, so every score is at maximum reasoning
effort; and the two accuracy runs differ by 11 of 100 LiveCodeBench verdicts on identical
inputs, because MTP drafting is distribution-identical, not token-identical. Sampled sets of
100 carry about ten points of uncertainty.

The perplexity file is wikitext-2 at 2048 tokens and is not comparable with
`../perplexity-all-models.csv`, which is tulu-3 at 512.
