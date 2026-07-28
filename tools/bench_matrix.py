#!/usr/bin/env python3
"""Kimi-K3 TP4 benchmark matrix: concurrency x context.

Emits JSON to stdout for plotting. Every prompt is unique so the server's prompt
cache cannot serve it.

Measures, per point:
  ttft_s        time to first token (streaming) -- the interactive metric
  decode_tps    tokens/s once generation starts, per stream
  agg_tps       aggregate tokens/s across all concurrent streams
  prefill_tps   prompt tokens / (ttft - fixed overhead estimate)

Usage:
  bench_matrix.py concurrency   # sweep concurrency at short context
  bench_matrix.py context       # sweep context at concurrency 1
  bench_matrix.py interaction   # concurrency sweep at 8K context
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
FILLER = (
    "Distributed inference places weights across accelerators; the layout fixes "
    "both the memory ceiling and the communication pattern that follows. "
)


def make_prompt(uid, ctx_words):
    head = f"[uid {uid}] "
    if ctx_words <= 0:
        return head + "Explain tensor parallelism in two sentences."
    body = FILLER * max(1, ctx_words // 20)
    return head + body + "\n\nSummarise the above in two sentences."


def stream_one(uid, ctx_words, max_tokens, results, lock):
    body = json.dumps(
        {
            "model": "default_model",
            "messages": [{"role": "user", "content": make_prompt(uid, ctx_words)}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload, strict=False)
                except Exception:
                    continue
                delta = d.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
        total = time.time() - t0
        with lock:
            results.append(
                {"uid": uid, "ttft_s": ttft, "tokens": n, "wall_s": total,
                 "decode_tps": (n - 1) / (total - ttft) if ttft and total > ttft and n > 1 else None}
            )
    except Exception as e:  # noqa: BLE001
        with lock:
            results.append({"uid": uid, "error": str(e)[:120], "wall_s": time.time() - t0})


def run_point(concurrency, ctx_words, max_tokens):
    results, lock, threads = [], threading.Lock(), []
    t0 = time.time()
    for i in range(concurrency):
        t = threading.Thread(
            target=stream_one,
            args=(f"{concurrency}x{ctx_words}x{i}", ctx_words, max_tokens, results, lock),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0

    ok = [r for r in results if "error" not in r]
    toks = sum(r["tokens"] for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
    dtps = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    return {
        "concurrency": concurrency,
        "ctx_words": ctx_words,
        "max_tokens": max_tokens,
        "wall_s": round(wall, 2),
        "tokens": toks,
        "agg_tps": round(toks / wall, 3) if wall else None,
        "ttft_s_mean": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "ttft_s_min": round(min(ttfts), 2) if ttfts else None,
        "ttft_s_max": round(max(ttfts), 2) if ttfts else None,
        "decode_tps_mean": round(sum(dtps) / len(dtps), 3) if dtps else None,
        "errors": len(results) - len(ok),
    }


PLANS = {
    # interactive -> batch, short context
    "concurrency": [(c, 0, 48) for c in (1, 2, 4, 8, 16, 32)],
    # context scaling, single stream
    "context": [(1, w, 24) for w in (0, 1000, 4000, 16000, 48000)],
    # does batching still pay at long context?
    "interaction": [(c, 8000, 24) for c in (1, 4, 16)],
}


def main():
    plan = sys.argv[1] if len(sys.argv) > 1 else "concurrency"
    out = []
    for conc, ctx, mt in PLANS[plan]:
        r = run_point(conc, ctx, mt)
        out.append(r)
        print(json.dumps(r), file=sys.stderr, flush=True)
    print(json.dumps({"plan": plan, "points": out}, indent=2))


if __name__ == "__main__":
    main()
