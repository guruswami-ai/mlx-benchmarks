#!/usr/bin/env bash
# check-prefill-scaling.sh — does prefill scale the way the architecture claims?
#
# Measures prefill rate at two context depths and compares them. A model with
# working sparse attention holds a near-flat rate as context grows, because each
# query attends to a fixed number of keys. A dense path loses rate in proportion
# to context.
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

# SMALL and LARGE must both fit inside the model's context window.
URL=""; MODEL=""; EXPECT=""; SMALL=16384; LARGE=49152; TIMEOUT=3600
PASS_RATIO=0.70   # sparse must retain at least this share of its rate
DENSE_RATIO=0.55  # dense is expected to fall below this

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --expect) EXPECT="$2"; shift 2 ;;
    --small) SMALL="$2"; shift 2 ;;
    --large) LARGE="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$URL" ] && [ -n "$MODEL" ] && [ -n "$EXPECT" ] || {
  echo "FAIL: --url, --model and --expect are required" >&2; exit 2; }
case "$EXPECT" in sparse|dense) ;; *)
  echo "FAIL: --expect must be sparse or dense" >&2; exit 2 ;; esac

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "FAIL: no python3" >&2; exit 2; }

"$PY" - "$URL" "$MODEL" "$EXPECT" "$SMALL" "$LARGE" "$TIMEOUT" "$PASS_RATIO" "$DENSE_RATIO" <<'PYEOF'
import json, os, subprocess, sys, tempfile, time

url, model, expect = sys.argv[1], sys.argv[2], sys.argv[3]
small, large, timeout = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
pass_ratio, dense_ratio = float(sys.argv[7]), float(sys.argv[8])

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

if expect == "sparse":
    ok = retained >= pass_ratio
    print("  sparse attention should hold rate near flat (>= %.2f)" % pass_ratio)
else:
    ok = retained <= dense_ratio
    print("  dense attention should lose rate with context (<= %.2f)" % dense_ratio)

if ok:
    print("PASS")
    sys.exit(0)

print("FAIL: measured scaling does not match --expect %s" % expect)
if expect == "sparse":
    print("      A sparse model losing rate this way indicates the sparse path is not running.")
    print("      Check whether the checkpoint declares model_file in config.json, which")
    print("      shadows the runtime's own implementation. Run model_file_shadowing.py")
    print("      against the checkpoint directory.")
sys.exit(1)
PYEOF
