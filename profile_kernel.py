"""Measure device-side kernel time for sparse_attn_sharedkv (TileLang vs Ascend C).

The perf-compare wall-clock includes api.py host overhead (.contiguous(), a
.cpu() sync, dict lookups), which dominates tiny workloads (decode). This script
separates DEVICE time (npu.Event, what aclgraph deployment actually pays) from
WALL time, and optionally prints a per-op device breakdown via torch_npu.profiler
(no msprof CSV navigation needed).

SHAREDKV-ONLY: the SEPARATE metadata operator (~53ms prefill for the TileLang
port) is precomputed ONCE outside the timed region and passed in via `metadata=`,
exactly as a serving loop / perf_compare do. The timed/profiled region runs ONLY
the sharedkv op, so wall_ms / device_ms / the per-op table are sharedkv kernels
alone -- NOT metadata+sharedkv combined. (An earlier version timed the whole
metadata+sharedkv `once()`, folding the ~53ms metadata into the reported number.)

Usage (on the NPU container)::

    cd /sdb/yq/dsv4_planB
    python profile_kernel.py --scenario swa_prefill            # both impls, wall vs device
    python profile_kernel.py --scenario swa_prefill --table    # + per-op device table
    python profile_kernel.py --scenario swa_decode --impl tilelang --table

For a deeper cube/vector pipe breakdown, fall back to msprof on a single impl.
"""

from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import sparse_attn_sharedkv_perf_compare as P  # noqa: E402


def _time(sharedkv, inp, cfg, md, iters, warmup):
    """Time ONLY the sharedkv op (metadata `md` precomputed, passed in)."""
    for _ in range(warmup):
        sharedkv(inp, cfg, md)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    t0 = perf_counter()
    start.record()
    for _ in range(iters):
        sharedkv(inp, cfg, md)
    end.record()
    torch.npu.synchronize()
    wall_ms = (perf_counter() - t0) / iters * 1e3
    dev_ms = start.elapsed_time(end) / iters
    return wall_ms, dev_ms


def _table(sharedkv, inp, cfg, md, n=10):
    """Per-op device table for the sharedkv op only (metadata not run here)."""
    try:
        import torch_npu  # noqa: F401

        prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
        )
        with prof:
            for _ in range(n):
                sharedkv(inp, cfg, md)
            torch.npu.synchronize()
        print(prof.key_averages().table(sort_by="self_npu_time_total", row_limit=25))
    except Exception as e:  # noqa: BLE001
        print(f"  (torch_npu.profiler table unavailable: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["tilelang", "ascendc", "both"], default="both")
    ap.add_argument("--scenario", default="swa_prefill", choices=list(P.SCENARIOS))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--table", action="store_true", help="per-op device breakdown")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    cfg = P.SCENARIOS[args.scenario]
    impls = ["tilelang", "ascendc"] if args.impl == "both" else [args.impl]

    # The Ascend C path needs the custom op library registered (perf_compare does
    # this in its own main(); importing it as a module does NOT). This registers
    # torch.ops.custom.* AND torch_npu.npu_sparse_attn_sharedkv_metadata.
    if "ascendc" in impls:
        try:
            import torch_npu  # noqa: F401
            import custom_ops  # noqa: F401  (registers Ascend C ops)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[fatal] could not import custom_ops (Ascend C op): {exc!r}\n"
                "  Build/install the Ascend C operator, or use --impl tilelang.",
                file=sys.stderr,
            )
            return

    inp = P.stage_on_npu(P.build_inputs(cfg, dtype))
    print(f"scenario={args.scenario} dtype={args.dtype} iters={args.iters}")
    print("  (sharedkv-only: metadata precomputed once, passed in, NOT timed)")
    print(
        f"{'impl':10s} {'wall_ms':>10s} {'device_ms':>10s}  (device = real sharedkv kernel time)"
    )
    sk_fns = {"tilelang": P.tilelang_sharedkv, "ascendc": P.ascendc_sharedkv}
    md_fns = {"tilelang": P.tilelang_metadata, "ascendc": P.ascendc_metadata}
    for impl in impls:
        md = md_fns[impl](inp, cfg)  # the SEPARATE metadata op, built ONCE
        torch.npu.synchronize()
        wall, dev = _time(sk_fns[impl], inp, cfg, md, args.iters, args.warmup)
        print(f"{impl:10s} {wall:10.4f} {dev:10.4f}")

    if args.table:
        for impl in impls:
            print(f"\n=== per-op device breakdown: {impl} (sharedkv only) ===")
            md = md_fns[impl](inp, cfg)
            torch.npu.synchronize()
            _table(sk_fns[impl], inp, cfg, md)


if __name__ == "__main__":
    main()
