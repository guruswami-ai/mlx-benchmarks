#!/usr/bin/env python3
"""Prefill-scaling sweep for Kimi-K3 TP4.

For a frontier model used on hard problems, the decisive numbers are (a) how much
context it can ingest and how fast, and (b) whether decode holds up deep into that
context -- NOT interactive tok/s on a 30-token reply.

Each prompt is made UNIQUE so the server's prompt cache cannot serve it, and
max_tokens is kept tiny so wall time is dominated by prefill.
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
FILLER = (
    "In distributed inference, the arrangement of weights across accelerators "
    "determines both the memory ceiling and the communication pattern. "
)


def ask(prompt, max_tokens, timeout=7200):
    body = json.dumps(
        {
            "model": "default_model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    el = time.time() - t0
    d = json.loads(raw, strict=False)
    return d["usage"], el


def main():
    targets = [int(a) for a in sys.argv[1:]] or [2000, 8000, 24000]
    print(f"{'target':>8} {'prompt_tok':>11} {'wall_s':>8} {'prefill_tok/s':>14}")
    print("-" * 46)
    for i, n_words in enumerate(targets):
        # unique prefix defeats the prompt cache
        prompt = f"[run {i} seed {n_words}] " + (FILLER * (n_words // 20))
        prompt += "\n\nReply with exactly the word: ACK"
        try:
            usage, el = ask(prompt, 1)
        except Exception as e:  # noqa: BLE001
            print(f"{n_words:>8} {'-':>11} {'-':>8}  FAILED: {e}")
            continue
        pt = usage["prompt_tokens"]
        print(f"{n_words:>8} {pt:>11} {el:>8.1f} {pt/el:>14.1f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
