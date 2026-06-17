#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Standalone perf comparison: Ascend C vs TileLang for the
SparseAttnSharedkv operator *pair* (metadata + sharedkv).

This is NOT a pytest -- run it directly on an Ascend NPU host:

    python sparse_attn_sharedkv_perf_compare.py
    python sparse_attn_sharedkv_perf_compare.py --scenarios scfa_prefill
    python sparse_attn_sharedkv_perf_compare.py --warmup 10 --iters 50
    python sparse_attn_sharedkv_perf_compare.py --only tilelang   # one side only

The call order matches both reference test flows 1:1: the metadata op
is invoked first, and its output is fed into the sharedkv op as the
``metadata=`` argument (see
``sparse_attn_sharedkv/tests/pytest/batch/sparse_attn_sharedkv_process.py``
for Ascend C and ``sparse_attn_sharedkv_tilelang/
test_sparse_attn_sharedkv.py::_call_metadata_then_sharedkv`` for TileLang).

For each scenario the script reports TileLang's performance as a
*percentage* of Ascend C -- ``AscendC_latency / TileLang_latency * 100%``.
Performance is the inverse of latency, so this is NOT the two run times
divided directly; a value < 100% means TileLang is slower, > 100% means
TileLang is faster:

    * metadata  op : its own percentage
    * sharedkv  op : its own percentage
    * overall (md + sk together) : combined percentage

Plus a grand summary aggregated across all scenarios run.

Methodology notes
-----------------
* **Warm up before timing.** The TileLang sharedkv kernel is JIT-compiled
  on its first call for a given config (tens of seconds); the NPU
  allocator and op caches also need to settle. We therefore run several
  untimed warm-up iterations before collecting any measurement.
* Every timed boundary is bracketed by ``torch.npu.synchronize()`` so we
  measure device completion, not async dispatch.
* All inputs are staged on the NPU once, *outside* the timed region, so
  H2D copies are not counted.
* **Caveat on the metadata op:** the Ascend C metadata op runs on the AI
  CPU (a device kernel), whereas the TileLang ``metadata`` is a host-side
  Python port of that scheduler. Their wall-clock comparison is therefore
  host-vs-device in nature -- we report it because it is part of the call
  chain, but read the metadata percentage with that asymmetry in mind.
  The sharedkv percentage is the apples-to-apples kernel comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

import numpy as np
import torch

# ---- Make the TileLang port importable (api / metadata / golden). ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_TL_DIR = os.path.join(_HERE, "sparse_attn_sharedkv_tilelang")
if _TL_DIR not in sys.path:
    sys.path.insert(0, _TL_DIR)


# ---- Scenario table: mirrors the Ascend C / TileLang paramsets 1:1. ----
# The three *prefill* cases drive S1 = 8192 (the requested 8K long
# sequence). Decode cases (S1 = 1) are kept for completeness / debugging.
SCENARIOS = {
    "scfa_prefill": dict(
        scenario=3,
        layout_q="TND",
        B=1,
        S1=8192,
        T1=8192,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 8192],
        seqused_kv=[8192],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "swa_prefill": dict(
        scenario=1,
        layout_q="TND",
        B=1,
        S1=8192,
        T1=8192,
        N1=64,
        N2=1,
        D=512,
        K=0,
        block_num1=65,
        block_num2=1,
        block_size1=128,
        block_size2=1,
        cu_seqlens_q=[0, 8192],
        seqused_kv=[8192],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "cfa_prefill": dict(
        scenario=2,
        layout_q="TND",
        B=1,
        S1=8192,
        T1=8192,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 8192],
        seqused_kv=[8192],
        softmax_scale=0.04419417,
        cmp_ratio=128,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "scfa_decode": dict(
        scenario=3,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "swa_decode": dict(
        scenario=1,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=0,
        block_num1=65,
        block_num2=1,
        block_size1=128,
        block_size2=1,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "cfa_decode": dict(
        scenario=2,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=128,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
}

DEFAULT_SCENARIOS = ["scfa_prefill", "swa_prefill", "cfa_prefill"]


# ---- Ascend C PA_ND stride padding (verbatim from the reference flow). ----
# Mirrors ``create_tensor_with_stride_padding`` in
# ``sparse_attn_sharedkv/tests/pytest/batch/sparse_attn_sharedkv_process.py``:
# the Ascend C op consumes a stride-padded paged-KV layout (an extra
# ``pad_len`` elements of stride between block rows). We replicate it so
# the Ascend C side sees exactly the tensor its own suite feeds it.
def create_tensor_with_stride_padding(src_tensor, pad_len):
    import torch_npu  # noqa: F401  (registers the npu backend)

    if src_tensor is None or src_tensor.dim() != 4:
        return src_tensor

    device = src_tensor.device
    dtype = src_tensor.dtype
    shape = list(src_tensor.shape)  # e.g. [65, 128, 1, 512]

    row_logical_size = shape[1] * shape[2] * shape[3]
    stride_0 = row_logical_size + pad_len
    new_strides = [stride_0, shape[2] * shape[3], shape[3], 1]

    physical_numel = stride_0 * shape[0]
    storage_tensor = torch.full(
        (physical_numel,), float("nan"), dtype=dtype, device=device
    )
    raw_storage = storage_tensor.untyped_storage()

    target_tensor = torch.empty((0,), dtype=dtype, device=device)
    target_tensor.set_(raw_storage, 0, shape, new_strides)

    for i in range(shape[0]):
        target_tensor[i].copy_(src_tensor[i])

    torch_npu.npu.synchronize()
    return target_tensor


# ---- Input synthesis (reuses the TileLang golden data generators). ----
def build_inputs(cfg, dtype, seed=42):
    """Generate paged KV / Q / indices on CPU for one scenario.

    No CPU golden is computed -- this is a perf benchmark, so only the
    shapes / dtypes matter. Both implementations consume the *same*
    generated tensors so the workload is identical.
    """
    import golden as G  # local import: tilelang dir is on sys.path

    torch.manual_seed(seed)
    np.random.seed(seed)

    layout_q = cfg["layout_q"]
    B, S1 = cfg["B"], cfg["S1"]
    T1 = cfg.get("T1", S1 * B)
    N1, N2, D = cfg["N1"], cfg["N2"], cfg["D"]
    K = cfg.get("K", 0)
    cmp_ratio = cfg["cmp_ratio"]
    bs1, bn1 = cfg["block_size1"], cfg["block_num1"]
    bs2, bn2 = cfg.get("block_size2"), cfg.get("block_num2")
    seqused_kv = cfg["seqused_kv"]
    cu = torch.tensor(cfg["cu_seqlens_q"], dtype=torch.int32)
    scenario = cfg["scenario"]

    if layout_q == "TND":
        q = (torch.rand((T1, N1, D)) * 20 - 10).to(dtype)
    else:
        q = (torch.rand((B, S1, N1, D)) * 20 - 10).to(dtype)

    ori_pa, ori_bt, _ = G.gen_ori_kv_paged(
        B=B,
        N2=N2,
        D=D,
        block_num=bn1,
        block_size=bs1,
        seqused_kv=seqused_kv,
        dtype=dtype,
        data_range=(-10, 10),
        rng=np.random.default_rng(seed),
    )

    if scenario >= 2:
        cmp_pa, cmp_bt, _, _ = G.gen_cmp_kv_paged(
            B=B,
            N2=N2,
            D=D,
            block_num=bn2,
            block_size=bs2,
            seqused_kv=seqused_kv,
            cmp_ratio=cmp_ratio,
            dtype=dtype,
            data_range=(-5, 10),
            rng=np.random.default_rng(seed + 1),
        )
    else:
        cmp_pa, cmp_bt = None, None

    if scenario == 3:
        g = torch.Generator()
        g.manual_seed(seed + 7)
        cmp_idx = G.gen_cmp_sparse_indices_tnd(
            B=B,
            T1=T1,
            N2=N2,
            K=K,
            cu_seqlens_q=cu,
            seqused_kv=seqused_kv,
            cmp_ratio=cmp_ratio,
            cmp_mask_mode=3,
            rng=g,
        )
    else:
        cmp_idx = None

    sinks = (torch.rand(N1) * 2 - 1).to(torch.float32)
    return dict(
        q=q,
        ori_pa=ori_pa,
        ori_bt=ori_bt,
        cmp_pa=cmp_pa,
        cmp_bt=cmp_bt,
        cmp_idx=cmp_idx,
        sinks=sinks,
        cu_seqlens_q=cu,
        seqused_kv=torch.tensor(seqused_kv, dtype=torch.int32),
    )


def stage_on_npu(c, stride_pad=True):
    """Move all inputs to the NPU once (outside the timed region).

    Shared read-only tensors are uploaded a single time. The Ascend C
    side additionally receives a stride-padded paged-KV copy (its
    reference contract); the TileLang api forces ``.contiguous()`` so it
    uses the plain paged layout.
    """
    import torch_npu  # noqa: F401

    def dev(t):
        return None if t is None else t.npu().contiguous()

    inp = dict(
        q_npu=dev(c["q"]),
        sinks_npu=dev(c["sinks"]),
        cu_seqlens_q_npu=dev(c["cu_seqlens_q"]),
        cu_seqlens_q_cpu=c["cu_seqlens_q"],
        seqused_kv_npu=dev(c["seqused_kv"]),
        seqused_kv_cpu=c["seqused_kv"],
        ori_bt_npu=dev(c["ori_bt"]),
        cmp_bt_npu=dev(c["cmp_bt"]),
        cmp_idx_npu=dev(c["cmp_idx"]),
        ori_pa_npu=dev(c["ori_pa"]),  # TileLang: contiguous paged KV
        cmp_pa_npu=dev(c["cmp_pa"]),
        empty_npu=torch.tensor([]).npu(),
    )
    if stride_pad:
        inp["ori_pa_pad_npu"] = create_tensor_with_stride_padding(dev(c["ori_pa"]), 64)
        inp["cmp_pa_pad_npu"] = (
            create_tensor_with_stride_padding(dev(c["cmp_pa"]), 64)
            if c["cmp_pa"] is not None
            else None
        )
    else:
        inp["ori_pa_pad_npu"] = inp["ori_pa_npu"]
        inp["cmp_pa_pad_npu"] = inp["cmp_pa_npu"]
    return inp


def _sync():
    torch.npu.synchronize()


# ---- One metadata + sharedkv iteration, per implementation. ----
def tilelang_once(inp, cfg):
    """Run TileLang metadata then sharedkv; return (md_ms, sk_ms)."""
    from api import sparse_attn_sharedkv as tl_sharedkv
    from metadata import sparse_attn_sharedkv_metadata as tl_metadata

    scenario = cfg["scenario"]
    N1, N2, D, B = cfg["N1"], cfg["N2"], cfg["D"], cfg["B"]
    K = cfg.get("K", 0)
    layout_q = cfg["layout_q"]
    max_seqlen_q = cfg.get("T1") if layout_q == "TND" else cfg["S1"]
    max_seqlen_kv = int(max(cfg["seqused_kv"]))

    md_kwargs = dict(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=inp["cu_seqlens_q_cpu"] if layout_q == "TND" else None,
        seqused_kv=inp["seqused_kv_cpu"],
        batch_size=B,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q=layout_q,
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=scenario >= 2,
    )
    if scenario >= 2:
        md_kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        md_kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]
    if scenario == 3:
        md_kwargs["cmp_topk"] = K

    _sync()
    t0 = perf_counter()
    md = tl_metadata(**md_kwargs)
    _sync()
    t1 = perf_counter()
    with torch.device("npu"):
        tl_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_npu"],
            cmp_kv=inp["cmp_pa_npu"],
            cmp_sparse_indices=inp["cmp_idx_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cmp_block_table=inp["cmp_bt_npu"],
            cu_seqlens_q=inp["cu_seqlens_q_npu"] if layout_q == "TND" else None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            metadata=md,
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"] if scenario >= 2 else None,
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
            return_softmax_lse=True,
            topk_cmp=K,
        )
    _sync()
    t2 = perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3


def ascendc_once(inp, cfg):
    """Run Ascend C metadata then sharedkv; return (md_ms, sk_ms)."""
    import torch_npu

    scenario = cfg["scenario"]
    N1, N2, D, B = cfg["N1"], cfg["N2"], cfg["D"], cfg["B"]
    K = cfg.get("K", 0)
    layout_q = cfg["layout_q"]
    max_seqlen_q = cfg.get("T1") if layout_q == "TND" else cfg["S1"]
    ori_max_s2 = int(max(cfg["seqused_kv"]))
    empty = inp["empty_npu"]

    md_kwargs = dict(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=inp["cu_seqlens_q_npu"],
        cu_seqlens_ori_kv=empty,
        cu_seqlens_cmp_kv=empty,
        seqused_q=empty,
        seqused_kv=inp["seqused_kv_npu"],
        batch_size=B,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=ori_max_s2,
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q=layout_q,
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=scenario >= 2,
        device="npu:0",
    )
    if scenario == 2:
        md_kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        md_kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]
    elif scenario == 3:
        md_kwargs["cmp_topk"] = K
        md_kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        md_kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]

    cu_q = inp["cu_seqlens_q_npu"] if layout_q == "TND" else None

    _sync()
    t0 = perf_counter()
    md = torch_npu.npu_sparse_attn_sharedkv_metadata(**md_kwargs)
    _sync()
    t1 = perf_counter()
    if scenario == 1:  # SWA
        torch.ops.custom.npu_sparse_attn_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_pad_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cu_seqlens_q=cu_q,
            seqused_q=None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            metadata=md,
            softmax_scale=cfg["softmax_scale"],
            ori_mask_mode=cfg["ori_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
        )
    elif scenario == 2:  # CFA
        torch.ops.custom.npu_sparse_attn_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_pad_npu"],
            cmp_kv=inp["cmp_pa_pad_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cmp_block_table=inp["cmp_bt_npu"],
            cu_seqlens_q=cu_q,
            seqused_q=None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            metadata=md,
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"],
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
        )
    else:  # SCFA
        torch.ops.custom.npu_sparse_attn_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_pad_npu"],
            cmp_kv=inp["cmp_pa_pad_npu"],
            cmp_sparse_indices=inp["cmp_idx_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cmp_block_table=inp["cmp_bt_npu"],
            cu_seqlens_q=cu_q,
            seqused_q=None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            metadata=md,
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"],
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
        )
    _sync()
    t2 = perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3


# ---- Timing loop + stats. ----
def bench(once_fn, inp, cfg, warmup, iters):
    """Warm up, then collect ``iters`` (md_ms, sk_ms) samples."""
    _sync()
    for _ in range(warmup):
        once_fn(inp, cfg)
    md_times, sk_times = [], []
    for _ in range(iters):
        md_ms, sk_ms = once_fn(inp, cfg)
        md_times.append(md_ms)
        sk_times.append(sk_ms)
    return md_times, sk_times


def stats(ts):
    a = sorted(ts)
    n = len(a)
    median = a[n // 2] if n % 2 else 0.5 * (a[n // 2 - 1] + a[n // 2])
    return dict(median=median, mean=sum(a) / n, min=a[0], max=a[-1])


def _perf_pct(ac_ms, tl_ms):
    # Performance is the inverse of latency, so TileLang's performance as a
    # percentage of Ascend C is (AscendC_latency / TileLang_latency) * 100,
    # NOT a raw division of the two run times. <100% => TileLang is slower
    # than Ascend C; >100% => TileLang is faster.
    return float("inf") if tl_ms <= 0 else ac_ms / tl_ms * 100.0


def _fmt_line(label, ac_ms, tl_ms):
    pct = _perf_pct(ac_ms, tl_ms)
    return (
        f"  {label:<10}  AscendC={ac_ms:9.4f} ms   TileLang={tl_ms:9.4f} ms"
        f"   TileLang perf = {pct:6.1f}% of Ascend C"
    )


# ---- Driver. ----
def run_scenario(name, dtype, warmup, iters, only, stride_pad):
    cfg = SCENARIOS[name]
    print(
        f"\n================ {name}  "
        f"(scenario={cfg['scenario']}, dtype={dtype}, S1={cfg['S1']}, "
        f"seqused_kv={cfg['seqused_kv']}) ================",
        flush=True,
    )

    cpu_inputs = build_inputs(cfg, dtype)
    inp = stage_on_npu(cpu_inputs, stride_pad=stride_pad)

    result = {"name": name}

    if only in ("both", "ascendc"):
        ac_md, ac_sk = bench(ascendc_once, inp, cfg, warmup, iters)
        result["ac_md"] = stats(ac_md)
        result["ac_sk"] = stats(ac_sk)
    if only in ("both", "tilelang"):
        tl_md, tl_sk = bench(tilelang_once, inp, cfg, warmup, iters)
        result["tl_md"] = stats(tl_md)
        result["tl_sk"] = stats(tl_sk)

    # Per-op absolute medians.
    if "ac_md" in result and "tl_md" in result:
        ac_md_m, ac_sk_m = result["ac_md"]["median"], result["ac_sk"]["median"]
        tl_md_m, tl_sk_m = result["tl_md"]["median"], result["tl_sk"]["median"]
        print(
            "  -- median latency (lower=better); "
            "TileLang perf = AscendC / TileLang x 100% --"
        )
        print(_fmt_line("metadata", ac_md_m, tl_md_m))
        print(_fmt_line("sharedkv", ac_sk_m, tl_sk_m))
        print(_fmt_line("overall", ac_md_m + ac_sk_m, tl_md_m + tl_sk_m))
    else:
        # Single-impl run: just dump the numbers it produced.
        for side, mk, sk in (
            ("AscendC", "ac_md", "ac_sk"),
            ("TileLang", "tl_md", "tl_sk"),
        ):
            if mk in result:
                print(
                    f"  [{side}] metadata median={result[mk]['median']:.4f} ms"
                    f"   sharedkv median={result[sk]['median']:.4f} ms"
                    f"   overall={result[mk]['median'] + result[sk]['median']:.4f} ms"
                )
    return result


def print_summary(results, only):
    if only != "both":
        print("\n(only one implementation was run; no percentages computed)")
        return
    ok = [r for r in results if "ac_md" in r and "tl_md" in r]
    if not ok:
        return
    print("\n========== GRAND SUMMARY (TileLang perf as % of Ascend C) ==========")
    print(
        f"{'scenario':<14}{'metadata':>12}{'sharedkv':>12}{'overall':>12}"
        "   (perf% = AscendC / TileLang)"
    )
    sum_ac_md = sum_ac_sk = sum_tl_md = sum_tl_sk = 0.0
    for r in ok:
        ac_md, ac_sk = r["ac_md"]["median"], r["ac_sk"]["median"]
        tl_md, tl_sk = r["tl_md"]["median"], r["tl_sk"]["median"]
        sum_ac_md += ac_md
        sum_ac_sk += ac_sk
        sum_tl_md += tl_md
        sum_tl_sk += tl_sk
        print(
            f"{r['name']:<14}{_perf_pct(ac_md, tl_md):>11.1f}%"
            f"{_perf_pct(ac_sk, tl_sk):>11.1f}%"
            f"{_perf_pct(ac_md + ac_sk, tl_md + tl_sk):>11.1f}%"
        )
    print("-" * 62)
    print(
        f"{'TOTAL (sum)':<14}"
        f"{_perf_pct(sum_ac_md, sum_tl_md):>11.1f}%"
        f"{_perf_pct(sum_ac_sk, sum_tl_sk):>11.1f}%"
        f"{_perf_pct(sum_ac_md + sum_ac_sk, sum_tl_md + sum_tl_sk):>11.1f}%"
    )
    print(
        "\nInterpretation: each value is TileLang performance as a percent of "
        "Ascend C,\n  perf% = AscendC_latency / TileLang_latency x 100  "
        "(the inverse of a raw time ratio).\n  <100% => TileLang is slower "
        "than Ascend C; >100% => TileLang is faster."
    )
    print(
        "Note: the metadata percentage compares a host-side Python port "
        "(TileLang) against\nan on-device AI CPU kernel (Ascend C) -- "
        "read it with that asymmetry in mind."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        choices=list(SCENARIOS.keys()),
        help="scenarios to run (default: the three 8K prefill cases)",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument(
        "--warmup", type=int, default=5, help="untimed warm-up iterations (default 5)"
    )
    ap.add_argument(
        "--iters", type=int, default=30, help="timed iterations (default 30)"
    )
    ap.add_argument(
        "--only",
        default="both",
        choices=["both", "tilelang", "ascendc"],
        help="run a single implementation (skips percentage reporting)",
    )
    ap.add_argument(
        "--no-stride-pad",
        action="store_true",
        help="skip the Ascend C PA_ND stride padding "
        "(by default it is applied to match the reference flow)",
    )
    args = ap.parse_args()

    # Fail early with a readable message if a backend is missing.
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - host dependent
        print(
            f"[fatal] torch_npu unavailable: {exc!r}\n"
            "This benchmark must run on an Ascend NPU host.",
            file=sys.stderr,
        )
        return 2
    if not torch.npu.is_available():
        print(
            "[fatal] torch.npu.is_available() == False; need an NPU.", file=sys.stderr
        )
        return 2
    if args.only in ("both", "ascendc"):
        try:
            import custom_ops  # noqa: F401  (registers torch.ops.custom.*)
        except Exception as exc:  # pragma: no cover - host dependent
            print(
                f"[fatal] could not import custom_ops (Ascend C op): {exc!r}\n"
                "Build/install the Ascend C operator, or use "
                "--only tilelang.",
                file=sys.stderr,
            )
            return 2

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(
        f"config: scenarios={args.scenarios} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters} only={args.only} "
        f"stride_pad={not args.no_stride_pad}"
    )

    results = []
    for name in args.scenarios:
        try:
            results.append(
                run_scenario(
                    name,
                    dtype,
                    args.warmup,
                    args.iters,
                    args.only,
                    not args.no_stride_pad,
                )
            )
        except Exception as exc:  # keep going so one bad case isn't fatal
            import traceback

            print(f"[error] scenario {name} failed: {exc!r}", file=sys.stderr)
            traceback.print_exc()

    print_summary(results, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
