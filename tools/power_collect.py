#!/usr/bin/env python3
"""Sample cluster power/thermal/GPU telemetry during a benchmark.

Scrapes the existing mactop Prometheus exporters (:9100 on every node) and writes
one JSONL record per sample. Run alongside a benchmark, then summarise over the
window with --summarise.

The point is to attach real wattage to every throughput number, so claims like
"runs off a standard power point" are measured rather than inferred from Apple's
480 W max-continuous rating.

  power_collect.py collect <out.jsonl> [interval_s]     # runs until killed
  power_collect.py summarise <out.jsonl> [t_start t_end]
"""
import json
import sys
import time
import urllib.request

NODES = ["muladhara", "svadhisthana", "anahata", "vishuddha"]

WANT = {
    "mactop_power_watts": ("component", ["cpu", "gpu", "ane", "dram", "system", "total"]),
    "mactop_memory_gb": ("type", ["used"]),
}
SCALARS = [
    "mactop_gpu_utilization_percent",
    "mactop_soc_temperature_celsius",
    "mactop_gpu_temperature_celsius",
    "mactop_gpu_freq_mhz",
]


def scrape(host, timeout=4):
    out = {}
    try:
        with urllib.request.urlopen(f"http://{host}:9100/metrics", timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name_part, value = line.rsplit(" ", 1)
            val = float(value)
        except ValueError:
            continue
        if "{" in name_part:
            base, labels = name_part.split("{", 1)
            labels = labels.rstrip("}")
            if base in WANT:
                key, keep = WANT[base]
                for kv in labels.split(","):
                    k, _, v = kv.partition("=")
                    v = v.strip('"')
                    if k == key and v in keep:
                        out[f"{base}.{v}"] = val
        elif name_part in SCALARS:
            out[name_part] = val
    return out


def collect(path, interval):
    with open(path, "a") as fh:
        while True:
            t = time.time()
            rec = {"t": round(t, 2), "nodes": {}}
            for h in NODES:
                s = scrape(h)
                if s:
                    rec["nodes"][h] = s
            tot = sum(
                v.get("mactop_power_watts.total", 0.0) for v in rec["nodes"].values()
            )
            rec["cluster_watts"] = round(tot, 2)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            time.sleep(max(0.0, interval - (time.time() - t)))


def summarise(path, t0=None, t1=None):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if t0 and d["t"] < t0:
                continue
            if t1 and d["t"] > t1:
                continue
            rows.append(d)
    if not rows:
        print("no samples in window")
        return

    cw = [r["cluster_watts"] for r in rows]
    span = rows[-1]["t"] - rows[0]["t"]
    energy_wh = (sum(cw) / len(cw)) * span / 3600

    print(f"samples {len(rows)}   window {span:.0f}s")
    print(f"cluster power   mean {sum(cw)/len(cw):7.1f} W   "
          f"peak {max(cw):7.1f} W   min {min(cw):7.1f} W")
    print(f"energy over window   {energy_wh:.2f} Wh")
    print()
    print(f"{'node':>14} {'mean W':>8} {'peak W':>8} {'gpu W pk':>9} "
          f"{'gpu %':>7} {'SoC C':>7} {'mem GB':>7}")
    print("-" * 68)
    for h in NODES:
        tot = [r["nodes"][h]["mactop_power_watts.total"]
               for r in rows if h in r["nodes"]]
        gpu = [r["nodes"][h].get("mactop_power_watts.gpu", 0)
               for r in rows if h in r["nodes"]]
        util = [r["nodes"][h].get("mactop_gpu_utilization_percent", 0)
                for r in rows if h in r["nodes"]]
        temp = [r["nodes"][h].get("mactop_soc_temperature_celsius", 0)
                for r in rows if h in r["nodes"]]
        mem = [r["nodes"][h].get("mactop_memory_gb.used", 0)
               for r in rows if h in r["nodes"]]
        if not tot:
            print(f"{h:>14}  (no samples)")
            continue
        print(f"{h:>14} {sum(tot)/len(tot):>8.1f} {max(tot):>8.1f} {max(gpu):>9.1f} "
              f"{max(util):>7.1f} {max(temp):>7.1f} {max(mem):>7.0f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "collect"
    if cmd == "collect":
        collect(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
    else:
        a = [float(x) for x in sys.argv[3:5]] if len(sys.argv) > 4 else [None, None]
        summarise(sys.argv[2], *a)
