"""Measure device-side kernel time for sparse_attn_sharedkv (TileLang vs Ascend C).

The perf-compare wall-clock includes api.py host overhead (.contiguous(), a
.cpu() sync, dict lookups), which dominates tiny workloads (decode). This script
separates DEVICE time (npu.Event, what aclgraph deployment actually pays) from
WALL time, and optionally prints a per-op device breakdown via torch_npu.profiler
(no msprof CSV navigation needed).

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


def _time(once, inp, cfg, iters, warmup):
    for _ in range(warmup):
        once(inp, cfg)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    t0 = perf_counter()
    start.record()
    for _ in range(iters):
        once(inp, cfg)
    end.record()
    torch.npu.synchronize()
    wall_ms = (perf_counter() - t0) / iters * 1e3
    dev_ms = start.elapsed_time(end) / iters
    return wall_ms, dev_ms


def _table(once, inp, cfg, n=10):
    try:
        import torch_npu  # noqa: F401

        prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
        )
        with prof:
            for _ in range(n):
                once(inp, cfg)
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

    inp = P.stage_on_npu(P.build_inputs(cfg, dtype))
    print(f"scenario={args.scenario} dtype={args.dtype} iters={args.iters}")
    print(
        f"{'impl':10s} {'wall_ms':>10s} {'device_ms':>10s}  (device = real kernel time)"
    )
    fns = {"tilelang": P.tilelang_once, "ascendc": P.ascendc_once}
    for impl in impls:
        wall, dev = _time(fns[impl], inp, cfg, args.iters, args.warmup)
        print(f"{impl:10s} {wall:10.4f} {dev:10.4f}")

    if args.table:
        for impl in impls:
            print(f"\n=== per-op device breakdown: {impl} ===")
            _table(fns[impl], inp, cfg)


if __name__ == "__main__":
    main()
