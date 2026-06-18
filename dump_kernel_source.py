"""Dump the Ascend C (CCE) source that TileLang generates for the
``sparse_attn_sharedkv`` kernel, for side-by-side comparison with the hand-written
Ascend C reference under ``ops-transformer/.../op_kernel``.

This confirms whether the TileLang port actually reproduces the Ascend C
structure (matmul tiling, cube/vector split, sync pattern) -- the codegen runs at
build time, so it only needs the CANN toolchain, not a device run.

Run on the NPU container::

    cd /sdb/yq/dsv4_planB
    python dump_kernel_source.py --scenario swa_prefill
    # writes swa_prefill_generated.cce and prints a structural digest

Commit the ``*_generated.cce`` (``git add -f``) so it can be reviewed, or paste
the printed digest.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sparse_attn_sharedkv_tilelang"
    ),
)

from kernel import build_sparse_attn_sharedkv  # noqa: E402

# Build params per scenario, mirroring how api.py derives them from the
# perf/test configs. Only shapes that affect the generated code structure matter.
CONFIGS = {
    "swa_prefill": dict(
        scenario=1, max_seq=8192, total_tokens=8192, seqused_kv=8192, topk_cmp=0
    ),
    "swa_decode": dict(
        scenario=1, max_seq=1, total_tokens=1, seqused_kv=8193, topk_cmp=0
    ),
}


def _digest(src: str) -> str:
    """A compact structural summary so divergence from Ascend C is visible fast."""
    lines = src.splitlines()
    # Count the key Ascend C primitives the codegen should emit.
    patterns = {
        "Mmad/gemm": r"\bMmad\b|Matmul|gemm",
        "LoadData (L1->L0)": r"LoadData",
        "DataCopy": r"\bDataCopy\b",
        "DataCopyPad": r"DataCopyPad",
        "Nd2Nz": r"Nd2Nz|nd2nz",
        "SetFlag": r"SetFlag|set_flag",
        "WaitFlag": r"WaitFlag|wait_flag",
        "CrossCoreSetFlag": r"CrossCore.*SetFlag|set_cross",
        "CrossCoreWaitFlag": r"CrossCore.*WaitFlag|wait_cross",
        "PipeBarrier": r"PipeBarrier|pipe_barrier",
        "barrier_all/SyncAll": r"SyncAll|barrier_all|BarrierAll",
        "Softmax": r"Softmax|softmax",
        "Brcb": r"\bBrcb\b|brcb",
        "Fixpipe": r"Fixpipe|fixpipe|FixPipe",
        "for-loops": r"\bfor\s*\(",
        "Exp": r"\bExp\b|\bexp\(",
    }
    out = [f"total lines: {len(lines)}, chars: {len(src)}", "op counts:"]
    for name, pat in patterns.items():
        n = len(re.findall(pat, src))
        out.append(f"  {name:24s}: {n}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="swa_prefill", choices=list(CONFIGS))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    c = CONFIGS[args.scenario]
    ori_block_size = 128
    ori_table_len = math.ceil(c["seqused_kv"] / ori_block_size)
    ori_block_num = ori_table_len + 1

    func = build_sparse_attn_sharedkv(
        batch=1,
        max_seq=c["max_seq"],
        total_tokens=c["total_tokens"],
        ori_block_num=ori_block_num,
        ori_block_size=ori_block_size,
        ori_table_len=ori_table_len,
        cmp_block_num=1,
        cmp_block_size=1,
        cmp_table_len=1,
        n_heads=64,
        n_kv_heads=1,
        head_dim=512,
        topk_cmp=c["topk_cmp"],
        cmp_ratio=4,
        scenario=c["scenario"],
        ori_win_left=127,
        softmax_scale=0.04419417,
        dtype=args.dtype,
        core_num=24,
    )
    src = func.get_kernel_source()
    out_path = f"{args.scenario}_generated.cce"
    with open(out_path, "w") as f:
        f.write(src)
    print(f"wrote {out_path}")
    print("=" * 60)
    print(_digest(src))


if __name__ == "__main__":
    main()
