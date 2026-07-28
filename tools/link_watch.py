#!/usr/bin/env python3
"""Watch TB5 link state across the cluster and report the moment one drops.

The open question is whether the "TP prefill deadlock" is actually a link
dropping mid-run. Every software suspect came back clean (FAST_SYNCH, prompt
cache, batching, PD state, transport under rdma_stress), while vishuddha has now
lost a link on two consecutive boots -- a different peer each time, cable
physically seated at both ends.

A silent link drop would look EXACTLY like the deadlock: one rank stops
participating, peers spin forever, no error anywhere, and the failure follows the
host rather than the rank index.

Checking before and after a run cannot distinguish "was already down" from
"dropped during". This samples continuously and timestamps any change, so it can.

  link_watch.py [interval_s] [logfile]
"""
import json
import subprocess
import sys
import time

NODES = ["muladhara", "svadhisthana", "anahata", "vishuddha"]
EXPECT_PEERS = 3          # 4-node full mesh
EXPECT_ACTIVE = 3

PROBE = (
    'echo "$(system_profiler SPThunderboltDataType 2>/dev/null '
    '| grep -c \'Device Name: Mac15,14\') "'
    '"$(networksetup -listallhardwareports 2>/dev/null '
    '| awk \'/Hardware Port: Thunderbolt [0-9]/{getline; print $2}\' '
    '| while read i; do [ "$(ifconfig $i 2>/dev/null '
    '| awk \'/status:/{print $2}\')" = active ] && printf x; done | wc -c)"'
)


def sample(host):
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", host, PROBE],
            capture_output=True, text=True, timeout=20,
        ).stdout.split()
        return int(out[0]), int(out[1])
    except Exception:
        return None, None


def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    path = sys.argv[2] if len(sys.argv) > 2 else "link_watch.jsonl"
    prev = {}
    fh = open(path, "a")
    print(f"watching {len(NODES)} nodes every {interval:g}s -> {path}", flush=True)
    print(f"expecting peers={EXPECT_PEERS} active={EXPECT_ACTIVE} per node", flush=True)

    while True:
        t = time.time()
        rec = {"t": round(t, 1), "nodes": {}}
        changes = []
        for h in NODES:
            peers, active = sample(h)
            rec["nodes"][h] = {"peers": peers, "active": active}
            if h in prev and prev[h] != (peers, active):
                changes.append(f"{h}: peers {prev[h][0]}->{peers} active {prev[h][1]}->{active}")
            prev[h] = (peers, active)
        degraded = [
            h for h, v in rec["nodes"].items()
            if v["peers"] is not None and v["peers"] < EXPECT_PEERS
        ]
        rec["degraded"] = degraded
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

        stamp = time.strftime("%H:%M:%S")
        if changes:
            print(f"[{stamp}] *** LINK CHANGE *** {'; '.join(changes)}", flush=True)
        elif degraded:
            print(f"[{stamp}] degraded: {','.join(degraded)}", flush=True)
        time.sleep(max(0.0, interval - (time.time() - t)))


if __name__ == "__main__":
    main()
