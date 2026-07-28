#!/usr/bin/env python3
"""Long-prefill test for Kimi-K3 TP4 — the claim underpinning the 1M-context story.

Escalating context, ONE stream at a time, no concurrency sweep beforehand. The
2026-07-28 deadlock (FAST_SYNCH=0, macOS 26.6) followed heavy batching plus a
filled prompt cache, so both are deliberately excluded here: run the server with
K3_PROMPT_CACHE_SIZE=1 and hit it sequentially.

Escalates so a failure localises to a context size rather than "it hung
somewhere". Each step prints before it starts, so a hang is attributable.

  prefill_test.py [ctx_words ...]      default 2000 8000 32000 100000
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
FILLER = (
    "In distributed inference the placement of weights across accelerators fixes "
    "both the memory ceiling and the communication pattern that follows from it. "
)


def ask(uid, ctx_words, max_tokens=8, timeout=14400):
    prompt = f"[uid {uid}] " + (FILLER * max(1, ctx_words // 20))
    prompt += "\n\nReply with exactly: ACK"
    body = json.dumps(
        {
            "model": "default_model",
            "messages": [{"role": "user", "content": prompt}],
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
    with urllib.request.urlopen(req, timeout=timeout) as r:
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
            if d.get("choices", [{}])[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return ttft, n, time.time() - t0


def main():
    targets = [int(a) for a in sys.argv[1:]] or [2000, 8000, 32000, 100000]
    print(f"{'ctx_words':>10} {'~tokens':>9} {'TTFT s':>9} {'prefill tok/s':>14} "
          f"{'gen':>5} {'wall s':>9}")
    print("-" * 62)
    for i, w in enumerate(targets):
        approx = int(w * 1.1)
        print(f"  -> starting {w} words (~{approx} tok) ...", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            ttft, n, wall = ask(f"pf{i}", w)
        except Exception as e:  # noqa: BLE001
            print(f"{w:>10} {approx:>9}   FAILED after {time.time()-t0:.0f}s: {str(e)[:60]}")
            sys.stdout.flush()
            continue
        rate = approx / ttft if ttft else 0
        print(f"{w:>10} {approx:>9} {ttft:>9.1f} {rate:>14.1f} {n:>5} {wall:>9.1f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
