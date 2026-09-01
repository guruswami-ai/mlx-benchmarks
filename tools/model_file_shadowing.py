#!/usr/bin/env python3
"""Detect a checkpoint whose bundled model code shadows an optimised runtime path.

An MLX checkpoint may ship its own implementation of a model and declare it in
config.json as "model_file", with trust_remote_code enabled. That declaration
takes precedence over the host runtime's own implementation of the same
architecture. When the runtime ships accelerated kernels, the bundled Python
runs instead and the kernels are never called.

Nothing reports this. The runtime can log that its optimised module registered
and that its native kernels are available, and both statements are true. The
model simply never reaches them.

Measured effect on GLM-5.3 4-bit (744B MoE, MLA + DeepSeek Sparse Attention) on
an M3 Ultra under oMLX 0.6.4: 64K prefill took 2341.7 s with the bundled file
and 431.7 s without it, on the same weights and the same server. Prefill went
from 28.0 to 151.7 tok/s. Output was correct in both cases.

This script is static. It does not load weights and it does not run inference.

Usage:
    check-model-file-shadowing.py /path/to/checkpoint [/path/to/another ...]
    check-model-file-shadowing.py --runtime-only

Exit: 0 nothing found, 1 at least one checkpoint is shadowing an available
optimised path, 2 usage error.

No warranty. Verify against your own runtime before acting on it.
"""
import json
import os
import sys

# Attention-path symbols only. Broader hints such as "gather" or "kernel" also
# match MoE matmul helpers and generic kernel builders, which are not the thing
# a bundled model file displaces.
ACCEL_HINTS = ("sparse_mla", "exact_block", "sparse_attention",
               "flash_attention", "fast_attention")


def runtime_kernels():
    """Report which optimised implementations the installed runtime offers."""
    found = {}
    try:
        import omlx  # noqa: F401
        for mt in ("glm_moe_dsa",):
            syms = _accelerated_for(mt)
            found["omlx.custom_kernels.%s" % mt] = syms or "no accelerated symbols"
    except ImportError:
        found["omlx"] = "not installed"
    for mod in ("mlx_lm", "mlx_vlm"):
        try:
            m = __import__(mod)
            found[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            found[mod] = "not installed"
    return found


def inspect(path):
    """Return (verdict, detail) for one checkpoint directory."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return "SKIP", "no config.json"
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except Exception as exc:
        return "SKIP", "unreadable config.json: %s" % exc

    model_file = cfg.get("model_file")
    model_type = cfg.get("model_type", "?")
    trc = cfg.get("trust_remote_code")
    bundled = sorted(f for f in os.listdir(path) if f.endswith(".py"))

    if not model_file:
        return "OK", "model_type=%s, no model_file declared" % model_type

    present = model_file in bundled
    detail = "model_type=%s, model_file=%s%s, trust_remote_code=%s" % (
        model_type, model_file, "" if present else " (DECLARED BUT ABSENT)", trc)

    if not present:
        return "WARN", detail
    accel = _accelerated_for(model_type)
    if accel:
        return "SHADOWING", detail + "\n      runtime offers: %s" % accel
    return "WARN", detail + "\n      no optimised runtime path found for this model_type"


def _accelerated_for(model_type):
    """Does the installed runtime have an accelerated path for this model_type?

    The kernels live in a `fast` submodule, not in the package __init__, so both
    are searched. Looking only at the package reports nothing and is the reason
    an earlier version of this check was useless.
    """
    names = []
    for modname in ("omlx.custom_kernels.%s.fast" % model_type,
                    "omlx.custom_kernels.%s" % model_type):
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        names += [s for s in dir(mod) if any(h in s for h in ACCEL_HINTS)]
    uniq = sorted(set(names))
    return ", ".join(uniq) if uniq else None


def main(argv):
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 2
    print("runtime:")
    for k, v in runtime_kernels().items():
        print("  %-38s %s" % (k, v))
    print()

    paths = [a for a in argv if not a.startswith("-")]
    if "--runtime-only" in argv or not paths:
        if not paths:
            print("no checkpoint given. Pass one or more checkpoint directories.")
        return 0

    worst = 0
    for p in paths:
        verdict, detail = inspect(p)
        print("%-11s %s" % (verdict, p))
        print("      %s" % detail)
        if verdict == "SHADOWING":
            worst = 1
    if worst:
        print()
        print("SHADOWING means the checkpoint's bundled implementation will be loaded")
        print("in place of the runtime's optimised one. To test, copy the checkpoint")
        print("directory using symlinks for the weights, remove \"model_file\" from the")
        print("copied config.json, omit the bundled .py, and compare prefill rate at")
        print("two context depths. Do not edit the original checkpoint until you have")
        print("measured the difference.")
        print()
        print("Note the reverse risk: where a runtime's own implementation has a bug")
        print("that the bundled file avoids, the shadowing is protecting you. Measure")
        print("correctness as well as speed before changing anything.")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
