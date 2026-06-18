"""Run ONE sparse_attn_sharedkv implementation in a tight loop for msprof.

This isolates a single backend + scenario so msprof attributes AI-Core (cube) /
AI-Vector time and per-op durations cleanly. It reuses the perf-compare input
builders so the workload is identical to the benchmark.

Usage (on the NPU container)::

    cd /sdb/yq/dsv4_planB
    # TileLang kernel:
    msprof --output=./prof_tl_swa \
        --application="python profile_kernel.py --impl tilelang --scenario swa_prefill --iters 20"
    # Ascend C reference (for comparison):
    msprof --output=./prof_ac_swa \
        --application="python profile_kernel.py --impl ascendc --scenario swa_prefill --iters 20"

Then look under ``prof_*/**/mindstudio_profiler_output/`` for
``op_summary_*.csv`` (per-op duration, aic/aiv time) and the
``*_aicore_*`` utilization, and paste the op-summary rows for the kernel.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import sparse_attn_sharedkv_perf_compare as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["tilelang", "ascendc"], default="tilelang")
    ap.add_argument("--scenario", default="swa_prefill", choices=list(P.SCENARIOS))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    cfg = P.SCENARIOS[args.scenario]

    inp = P.stage_on_npu(P.build_inputs(cfg, dtype))
    once = P.tilelang_once if args.impl == "tilelang" else P.ascendc_once

    for _ in range(args.warmup):
        once(inp, cfg)
    P._sync()
    for _ in range(args.iters):
        once(inp, cfg)
    P._sync()
    print(f"done: impl={args.impl} scenario={args.scenario} iters={args.iters}")


if __name__ == "__main__":
    main()
