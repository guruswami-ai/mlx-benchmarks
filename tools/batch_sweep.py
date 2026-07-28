#!/usr/bin/env python3
"""Concurrency sweep against the Kimi-K3 TP4 OpenAI endpoint.

Kimi-K3 at batch 1 is dispatch/fixed-overhead bound (~200-400 us floor per module
call, 6x above the bandwidth-ideal). Fixed overhead is exactly what batching
amortises, so aggregate throughput should scale far better than single-stream.

mlx_lm.server batches automatically when every cache implements merge() -- K3's
KVCache (MLA) and ArraysCache (KDA) both do. Defaults: decode-concurrency=32,
prompt-concurrency=8.

Each request gets a UNIQUE prompt so the prompt cache cannot serve it.
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_TOKENS = 48

PROMPTS = [
    "Explain why tensor parallelism needs an all-reduce after the output projection.",
    "Describe how a mixture-of-experts router chooses experts for a token.",
    "What is the difference between prefill and decode in transformer inference?",
    "Explain KV cache compression and why latent attention reduces memory.",
    "Why does linear attention have constant memory in sequence length?",
    "Describe speculative decoding and when it fails to help.",
    "What limits throughput when a model is dispatch-bound rather than bandwidth-bound?",
    "Explain how quantisation to 4 bits affects model quality and speed.",
]


def one(idx, out, lock):
    prompt = f"[req {idx}] " + PROMPTS[idx % len(PROMPTS)]
    body = json.dumps(
        {
            "model": "default_model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read().decode("utf-8", errors="replace"), strict=False)
        el = time.time() - t0
        with lock:
            out.append((d["usage"]["completion_tokens"], el))
    except Exception as e:  # noqa: BLE001
        with lock:
            out.append((0, time.time() - t0, str(e)))


def sweep(n):
    out, lock, threads = [], threading.Lock(), []
    t0 = time.time()
    for i in range(n):
        t = threading.Thread(target=one, args=(i, out, lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0
    toks = sum(o[0] for o in out)
    errs = sum(1 for o in out if len(o) > 2)
    return toks, wall, errs


def main():
    levels = [int(a) for a in sys.argv[1:]] or [1, 4, 16]
    print(f"{'concurrency':>12} {'tokens':>8} {'wall_s':>8} {'agg tok/s':>10} "
          f"{'per-stream':>11} {'errors':>7}")
    print("-" * 62)
    for n in levels:
        toks, wall, errs = sweep(n)
        agg = toks / wall if wall else 0
        print(f"{n:>12} {toks:>8} {wall:>8.1f} {agg:>10.2f} {agg/n:>11.2f} {errs:>7}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
