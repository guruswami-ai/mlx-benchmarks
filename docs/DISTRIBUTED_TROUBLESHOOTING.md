# Troubleshooting a distributed MLX job on Apple silicon — a runbook

Written after a session that burned several hours and half a dozen reboots on failures that all
looked identical from the outside. **The ordering matters**: each step is cheap and rules out a
whole class, so working top-down is much faster than starting with the interesting hypotheses.

Stack this was derived from: mlx 0.32.0, mlx-lm 0.31.3, macOS 26.6 (25G72), 4× M3 Ultra 512 GB,
TB5 full mesh, `jaccl`/RDMA. Kext `AppleThunderboltRDMA` 0.0.1.

---

## Step 0 — Read the FIRST lines of the output, not the last

A distributed launch failure produces a cascade of `terminate()` tracebacks and pid-file cleanup
errors. **The real error is at the top.** Two specific traps:

- **`subprocess.CalledProcessError` about a pid file is cosmetic.** The remote process exited
  normally before cleanup ran. The job may have fully succeeded — verify the actual output before
  believing the exit code.
- **A pid path of literally `cat None`** means the remote process never started. Look for a
  permissions or path problem, not a network one.

```bash
mlx.launch --hostfile hf.json ./job.py 2>&1 | head -40      # head, not tail
```

## Step 1 — Is it erroring, or is it hanging? They are different problems

This single distinction routes the entire rest of the runbook.

| | Erroring | Hanging |
|---|---|---|
| Looks like | explicit message, exits in seconds | nothing; runs forever |
| Go to | Step 2 | Step 3 |

**The most common error is PD exhaustion**, and it is easy to mistake for something exotic:

```
RuntimeError: [jaccl] Couldn't allocate protection domain
```
(older MLX reported this as `RTR failed with errno 96` or `22`)

RDMA protection domains are kernel-allocated and **never reclaimed on process exit**. After 2–3
distributed operations, nothing will work. **Only a reboot recovers.** This is a budget you spend,
not a fault — plan on 2–3 RDMA operations per boot and make the important one first.

## Step 2 — Erroring: match the message

| Message | Cause | Fix |
|---|---|---|
| `Couldn't allocate protection domain` | PD pool exhausted | **reboot** |
| `RTR failed with errno 60` | `mlx.distributed_config` run more than once this boot; ARP/RDMA mappings corrupted | **reboot**, then configure exactly once |
| `RTR failed with errno 22` | `bridge0` is active | kill the bridge; check `bridge-killer` is running |
| `OSError: [Errno 66] Directory not empty` | `--dst` exists and is non-empty | clear target dirs before broadcasting |
| `permission denied` / exit 126 | target script not executable, or absent at that path on some node | script must be executable with a shebang at the **same path on every node** |
| `Try passing --dot to visualize the connectivity` | mesh is incomplete | Step 5 |
| `[METAL] Command buffer execution failed ... Timeout` | Metal's ~60 s command-buffer limit | not viable at this layer count — use TP, not PP |

## Step 3 — Hanging: read watts, not GPU utilisation

**GPU utilisation reports 100 % while the GPU does nothing.** It is useless as a health signal
here. Power is honest. Sample every node:

```bash
for h in n1 n2 n3 n4; do
  printf "%-14s " $h
  curl -s http://$h:9100/metrics | awk -F' ' '
    /^mactop_power_watts.component="total"/{t=$2}
    /^mactop_gpu_utilization_percent/{g=$2}
    END{printf "gpu=%3.0f%% total=%3.0fW\n", g, t}'
done
```

Interpret against these, measured on M3 Ultra (total SoC, **not** GPU-only, which reads
milliwatts at idle):

| total SoC | GPU util | meaning |
|---|---|---|
| ~14 W | low | idle |
| ~24 W | **7 %** | **this rank has dropped out** — it is the one to investigate |
| 40–50 W | **100 %** | stalled *in* a collective — spinning, not working |
| 234–258 W | 100 % | genuinely working; be patient |

**A hang with one rank low and the rest spinning is a collective deadlock.** Note which host, then
Step 4.

## Step 4 — Does it follow the host, or the rank index?

The single most valuable experiment, and it costs one boot. Re-run `mlx.distributed_config` with
the host order permuted so the suspect node sits at a different rank, and repeat the failure.

- **Follows the host** → node-specific: config, hardware, or state on that machine
- **Follows the rank index** → an MLX/collective issue, and worth reporting upstream

Do not skip this. We spent hours on node-specific theories before running it.

## Step 5 — Mesh integrity, cheapest first

```bash
# read-only, free, safe to repeat -- ALWAYS do this before reconfiguring
mlx.distributed_config --over thunderbolt --hosts n1,n2,n3,n4 --dot
```

**Count the edges. A 4-node full mesh has exactly 6.** `--dot` names the missing pair directly.
If an edge is missing, no amount of reconfiguring will help — the fault is below the network layer:

```bash
system_profiler SPThunderboltDataType | grep -c "Device Name: Mac15,14"   # expect 3 per node
```

If the Thunderbolt **controller** reports no device, bridges, `ifconfig` and STP are all
irrelevant.

⚠️ **A dropped link is usually NOT a bad cable.** A link that ran for months can vanish after the
TB subsystem is disturbed — plugging in a display on an unrelated bus was enough. **Force a link
renegotiation first** (reseat, or reboot the node) before replacing cables one at a time, which
will just exonerate each of them in turn.

## Step 6 — Per-node config drift

```bash
# TB interface names are DYNAMIC -- never hardcode en2..en7
networksetup -listallhardwareports | awk '/Hardware Port: Thunderbolt [0-9]/{getline; print $2}'

# orphan interfaces: UP,RUNNING at MTU 1500, no backing hardware port, persist until reboot
comm -23 <(ifconfig -l | tr ' ' '\n' | grep '^en' | sort) \
         <(networksetup -listallhardwareports | awk '/^Device:/{print $2}' | grep '^en' | sort)

# control plane must be ETHERNET, not a Thunderbolt link-local
python3 -c "import json;d=json.load(open('hf.json'));print([h['ips'] for h in d['hosts']])"
```

Things that are **not** worth chasing, each verified by measurement:

- **MTU.** RDMA uses neither TCP/IP nor Ethernet frames. Measured identical throughput at 9000 and
  1500 (9.05 vs 8.93 GB/s at 134 MB). *Still matters for the `ring` backend over TB.*
- **`mlx.distributed_config` resetting MTU.** It does not — zero `mtu` references in its source,
  and interfaces read 9000 immediately after `--auto-setup`.
- **Copying the hostfile to peers.** Not needed on mlx 0.32.0; verified by deleting it from all
  peers.
- **Missing `RDMA Thunderbolt N` network services.** Correlated with a failure, but the node passed
  134 MB collectives at 9.33 GB/s without them.

## Step 7 — Isolate the layer with a model-free test

Before blaming the model, test the transport alone. `rdma_stress.py` runs escalating `all_sum`
(7 KB → 134 MB) with no model loaded, so it separates *node/transport* from *model/framework*:

- **Passes** → transport is fine; the fault is in the model or framework path
- **Fails** → node or interconnect; go back to Steps 5–6

This costs ~30 s versus a 4-minute model load, and it exonerated our entire network layer.

## Step 8 — Bisect the workload

Escalate a single dimension so failure localises to a value rather than "somewhere":

```
90 tok      1 chunk                    PASS
2499 tok    2048 + 451 remainder       PASS
9699 tok    2048 x 4 + remainder       HANGS after first chunk
```

Change one variable at a time and re-run with each suspect *removed* — that is what turns "these
five things might matter" into "none of these five matter".

---

## Rules that save the most time

1. **Reboot between hang investigations.** Killing ranks that hold a large wired model leaves
   ~500 GB in page cache that the next load must fight for; any measurement taken in that state is
   worthless. We wasted a decisive experiment this way.
2. **Never hard-kill a rank holding a loaded model** if it can be avoided.
3. **`mlx.distributed_config` exactly once per boot.** Retrying corrupts ARP/RDMA mappings and
   makes things worse — which is precisely why the whole process feels random.
4. **Budget 2–3 RDMA operations per boot.** Do the one that matters first. A "quick calibration
   run" is not free: ours cost 4× throughput on the real transfer.
5. **`--dot` before every reconfigure.** Read-only, free, and it catches physical faults before you
   spend the one config you get.
6. **State which power metric you mean.** GPU-only and total SoC differ ~250× at idle
   (39–55 mW vs 14 W). Mixing them silently produces conclusions that are wrong by two orders of
   magnitude.
