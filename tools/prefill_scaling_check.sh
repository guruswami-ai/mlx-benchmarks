#!/usr/bin/env bash
# check-prefill-scaling.sh — does prefill scale the way the architecture claims?
#
# Measures prefill rate at two context depths and compares them. A model with
# working sparse attention holds a near-flat rate as context grows, because each
# query attends to a fixed number of keys. A dense path loses rate in proportion
# to context.
#
# The near-flat claim holds inside a band, not forever. Two effects bend the
# curve down on a sparse model that is working correctly:
#   - the DSA indexer still scores every key before it selects its top-k, so
#     prefill carries a linear term even when attention itself is sparse;
#   - a memory guard that throttles the prefill chunk, or restarts the prefill,
#     costs wall time that this tool cannot separate from attention cost.
# Choose SMALL and LARGE inside the range you actually serve. A FAIL at 128K on
# a model that passes at 48K is more likely to be one of the two effects above
# than a dead sparse path.
#
# The printed rate is prompt_tokens divided by whole-request wall clock. It
# includes 8 decoded tokens and the network round trip, so it reads about 0.5%
# high at 16K and less at longer contexts. That bias favours PASS for sparse.
#
# The two prompts cannot share a prefix-cache entry: filler() stamps a
# millisecond nonce and a per-call tag on every line, so they differ at line 0
# and a repeat run differs from the previous one. The cached_tokens check is a
# second line of defence, not the guarantee.
#
# Why this exists. On 1 September 2026 a GLM-5.3 checkpoint was found to shadow
# oMLX's compiled sparse Metal kernels, because the checkpoint declared
# `model_file` in config.json. Prefill ran 5.4 times slower than necessary. The
# server logged three lines saying the optimised kernels loaded, and all three
# were true. Only the rate curve showed the fault.
#
# Fail-closed. A run that cannot measure is a FAIL, not a skip.
#
# Usage:
#   check-prefill-scaling.sh --url http://HOST:PORT --model NAME --expect sparse
#   check-prefill-scaling.sh --url ... --model ... --expect dense --small 8192 --large 24576
#
# Exit: 0 PASS, 1 FAIL, 2 usage or unreachable.

set -uo pipefail

# SMALL and LARGE must both fit inside the model's context window, and LARGE
# must be at least 2x SMALL. The verdict thresholds are derived from the growth
# ratio actually measured, not fixed, because "how much rate a dense path loses"
# is a function of how much the context grew.
URL=""; MODEL=""; EXPECT=""; SMALL=16384; LARGE=49152; TIMEOUT=3600
BAND=0.40         # share of the flat-to-dense span each verdict must clear

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --expect) EXPECT="$2"; shift 2 ;;
    --small) SMALL="$2"; shift 2 ;;
    --large) LARGE="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed -n '/^#/p'; exit 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$URL" ] && [ -n "$MODEL" ] && [ -n "$EXPECT" ] || {
  echo "FAIL: --url, --model and --expect are required" >&2; exit 2; }
case "$EXPECT" in sparse|dense) ;; *)
  echo "FAIL: --expect must be sparse or dense" >&2; exit 2 ;; esac

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "FAIL: no python3" >&2; exit 2; }

"$PY" - "$URL" "$MODEL" "$EXPECT" "$SMALL" "$LARGE" "$TIMEOUT" "$BAND" <<'PYEOF'
import json, os, subprocess, sys, tempfile, time

url, model, expect = sys.argv[1], sys.argv[2], sys.argv[3]
small, large, timeout = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
band = float(sys.argv[7])

def filler(approx_tokens, nonce):
    # ~25 tokens per line, measured against GLM-5.3's tokenizer. Content is
    # varied so the tokenizer cannot collapse it.
    # The nonce is on every line, so the two prompts diverge at line 0 and cannot
    # share a prefix-cache entry. A shared prefix would make the second reading
    # meaningless.
    n = max(1, approx_tokens // 25)
    return "\n".join(
        "run %s row %06d node=%d status=%s bytes=%d"
        % (nonce, i, i % 97, ("ok", "warn", "stale")[i % 3], (i * 7919) % 99999)
        for i in range(n))

NONCE = "%08x" % (int(time.time() * 1000) & 0xFFFFFFFF)

def measure(target, tag):
    body = {"model": model, "temperature": 0, "max_tokens": 8,
            "messages": [{"role": "user",
                          "content": filler(target, NONCE + tag)
                                     + "\n\nReply with the single word: ok"}]}
    # curl, not a Python socket. On a multi-homed host an unbound Python socket
    # can fail with "No route to host" where curl succeeds.
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(body, f)
    t0 = time.time()
    try:
        out = subprocess.run(
            ["curl", "-s", "-m", str(timeout), url,
             "-H", "Content-Type: application/json", "-d", "@" + path],
            capture_output=True, text=True)
    finally:
        os.unlink(path)
    el = time.time() - t0
    if out.returncode != 0 or not out.stdout.strip():
        print("FAIL: request at ~%d tokens failed: curl exit %d %s"
              % (target, out.returncode, out.stderr.strip()[:200]))
        sys.exit(1)
    try:
        d = json.loads(out.stdout)
    except Exception as e:
        print("FAIL: bad JSON at ~%d tokens: %s" % (target, out.stdout[:200])); sys.exit(1)
    if "error" in d:
        print("FAIL: server error at ~%d tokens: %s" % (target, str(d["error"])[:200]))
        sys.exit(1)
    u = d.get("usage") or {}
    pt = u.get("prompt_tokens")
    details = u.get("prompt_tokens_details")
    cached = (details or {}).get("cached_tokens")
    if not pt:
        print("FAIL: no prompt_tokens in usage at ~%d tokens" % target); sys.exit(1)
    if cached is None:
        # Fail closed. A runtime that does not report cached_tokens cannot prove
        # the prefill was cold, and a cache hit would read as a fast dense path.
        print("FAIL: runtime does not report prompt_tokens_details.cached_tokens.")
        print("      Cold prefill cannot be proven, so the rate is not evidence.")
        print("      Restart the server between measurements and re-run, or use a")
        print("      runtime that reports cache attribution.")
        sys.exit(1)
    if cached:
        print("FAIL: %d of %d tokens served from prefix cache. Rate is not measurable."
              % (cached, pt))
        print("      Restart the server or vary the prompt, then re-run.")
        sys.exit(1)
    rate = pt / el
    print("  %7d prompt tokens  %8.1fs  %7.1f tok/s" % (pt, el, rate))
    return pt, rate

print("prefill scaling check: model=%s expect=%s" % (model, expect))
p1, r1 = measure(small, "a")
p2, r2 = measure(large, "b")

ctx_growth = p2 / p1
retained = r2 / r1
print()
print("  context grew %.2fx, prefill rate retained %.2f of its value" % (ctx_growth, retained))

# The tool is only valid inside a band of growth ratios, and the measured ratio
# has to be checked because --small and --large are the user's to choose.
#
# Too little growth and the two hypotheses overlap: a dense path at 1.5x retains
# about 0.67, which a working sparse path also produces.
#
# Too much growth and the sparse reading stops being near-flat for reasons that
# have nothing to do with the sparse path. The indexer scores every key before
# it selects, and a memory guard may throttle the prefill chunk. Measured on
# GLM-5.3 under oMLX with the kernels demonstrably running: 152 tok/s at 15.7K
# and 64.8 tok/s at 130.1K, a retained value of 0.43 that this tool would call
# dense. It was not dense. Refusing the run is the honest answer.
if not (2.5 <= ctx_growth <= 3.5):
    print("FAIL: context grew %.2fx. This check is only valid between 2.5x and"
          % ctx_growth)
    print("      3.5x. Below that a dense path retains about %.2f, which a sparse"
          % (1.0 / ctx_growth))
    print("      path also produces. Above it, the indexer term and memory-guard")
    print("      throttling bend a working sparse curve down far enough to read as")
    print("      dense. Adjust --small and --large and re-run.")
    print("      This measures whether the sparse path runs. It does not measure")
    print("      how a model behaves at the top of its context window.")
    sys.exit(2)

# Thresholds derived from this run, not fixed. flat = 1.0 is the ideal sparse
# result; 1/growth is the ideal dense one. Each verdict must clear `band` of
# the span between them, which leaves an inconclusive gap in the middle.
dense_expected = 1.0 / ctx_growth
span = 1.0 - dense_expected
sparse_min = 1.0 - band * span
dense_max = dense_expected + band * span
print("  at %.2fx growth: dense would retain about %.2f, sparse about 1.00"
      % (ctx_growth, dense_expected))
print("  thresholds for this run: sparse >= %.2f, dense <= %.2f"
      % (sparse_min, dense_max))

if expect == "sparse":
    ok, boundary = retained >= sparse_min, sparse_min
else:
    ok, boundary = retained <= dense_max, dense_max

if ok:
    print("PASS")
    sys.exit(0)

if dense_max < retained < sparse_min:
    print("FAIL: inconclusive. %.2f retained falls between the two thresholds, so"
          % retained)
    print("      this run supports neither reading. Widen the gap between --small")
    print("      and --large, or measure on a quiet machine, and re-run.")
    sys.exit(1)

print("FAIL: measured scaling does not match --expect %s (%.2f against %.2f)"
      % (expect, retained, boundary))
if expect == "sparse":
    print("      Rule out the cheap causes first, in this order:")
    print("      1. Memory-guard throttling. Check the server log over the large run")
    print("         for prefill chunk reductions, prefill restarts or memory-pressure")
    print("         warnings. Those cost wall time that this tool reads as lost rate.")
    print("      2. Context out of band. The indexer scores every key before it")
    print("         selects, so rate falls at long context even when sparse attention")
    print("         is working. Re-run with --large inside the range you serve.")
    print("      3. Contention. Another process on the same GPU changes the reading.")
    print("      4. Only then, the sparse path itself. Check whether the checkpoint")
    print("         declares model_file in config.json, which shadows the runtime's")
    print("         own implementation. Run model_file_shadowing.py against the")
    print("         checkpoint directory.")
sys.exit(1)
PYEOF
