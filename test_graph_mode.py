"""Test whether GRAPH MODE (capture / replay) removes the host launch tax for the
TileLang sharedkv kernel -- for prefill AND decode -- and probe what graph APIs
this torch_npu/CANN exposes, plus whether the op is trainable (autograd).

Why: the per-call host cost (~0.9ms, sequence-length INDEPENDENT) is eager-mode
Python + framework launch overhead (torch.empty x7 + ptr-convert x18 + device/dtype
validation + the ACL launch, which is itself only ~0.03ms). In production this is
removed by graph mode: the op sequence is captured ONCE and replayed, skipping the
host dispatch. This script measures eager vs graph-replay to confirm, on the SWA
kernel whose device work is already at Ascend C parity.

Run on the NPU container:
    python test_graph_mode.py --scenario swa_decode   # fixed shape -> graph-friendly
    python test_graph_mode.py --scenario swa_prefill  # one fixed prefill shape

Reads the result with:
  * eager  per-call ~= host tax (~0.9ms) + device
  * graph  per-call ~= device only (host tax gone) IF capture/replay works
  * if graph replay << eager  -> graph mode removes the host tax (the decode answer)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sparse_attn_sharedkv_tilelang"
    ),
)

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

from sparse_attn_sharedkv_perf_compare import SCENARIOS, build_inputs, stage_on_npu
from api import (
    _get_kernel,
    sparse_attn_sharedkv,
    DEFAULT_BLOCK_I,
    DEFAULT_CORE_NUM,
)
from metadata import sparse_attn_sharedkv_metadata as tl_metadata


def probe_apis() -> None:
    """Discover which graph-capture / training facilities this stack exposes."""
    npu = torch_npu.npu
    print("=== graph API probe ===")
    print("  torch_npu.npu.NPUGraph              :", hasattr(npu, "NPUGraph"))
    print("  torch_npu.npu.graph (ctx mgr)       :", hasattr(npu, "graph"))
    print(
        "  torch_npu.npu.make_graphed_callables:",
        hasattr(npu, "make_graphed_callables"),
    )
    try:
        import torchair  # noqa: F401

        print("  torchair (GE graph / torch.compile) :", True)
    except Exception as e:  # noqa: BLE001
        print(
            "  torchair                            :",
            f"NOT available ({type(e).__name__})",
        )
    print()


def build_metadata(inp, cfg):
    """Build the COMPANION metadata operator's output (TileLang port of
    SparseAttnSharedkvMetadata). This is a SEPARATE op, OUT of the sharedkv perf
    scope: in serving AND in sparse_attn_sharedkv_perf_compare it is computed and
    timed on its own and passed INTO sharedkv via `metadata=...`. The TileLang
    metadata port is known-slow (~53ms for prefill) and is not what we are
    optimizing here. Returns the device int32 flat tensor the kernel consumes."""
    q = inp["q_npu"]
    dev = q.device
    N1, N2, D, B = cfg["N1"], cfg["N2"], cfg["D"], cfg["B"]
    cu = inp["cu_seqlens_q_npu"].to(torch.int32)
    S_max = int((cu[1:] - cu[:-1]).max().item())
    return (
        tl_metadata(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=inp["cu_seqlens_q_cpu"],
            seqused_kv=inp["seqused_kv_cpu"],
            batch_size=B,
            max_seqlen_q=cfg.get("T1", S_max),
            max_seqlen_kv=int(max(cfg["seqused_kv"])),
            ori_mask_mode=cfg["ori_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        .to(torch.int32)
        .to(dev)
        .reshape(-1)
    )


def build_swa_kernel_call(inp, cfg, md, contiguous=True):
    """Replicate api.sparse_attn_sharedkv's SWA (scenario 1) host prep and return
    (func, args), with the metadata operator's output `md` PASSED IN (precomputed
    once, exactly as a serving loop / perf_compare does -- metadata is a separate
    op, not part of the sharedkv host prep). This covers only the genuine sharedkv
    prep: shapes, the .item() for max_seq, dummies, _get_kernel lookup, the 11
    .contiguous(). Capturing func(*args) captures only the kernel launch.

    contiguous=False returns the RAW args (no .contiguous()), so the caller can
    time the prep and the 11 .contiguous() separately."""
    q = inp["q_npu"]  # [T1, N1, D], TND
    dtype = q.dtype
    dev = q.device
    N1, N2, D, B = cfg["N1"], cfg["N2"], cfg["D"], cfg["B"]
    T1 = q.shape[0]

    cu = inp["cu_seqlens_q_npu"].to(torch.int32)
    q_prefix = cu[:-1].contiguous()
    act_q_lens = (cu[1:] - cu[:-1]).contiguous()
    S_max = int(act_q_lens.max().item())

    ori_kv = inp["ori_pa_npu"]
    ori_bt = inp["ori_bt_npu"].to(torch.int32)
    cmp_kv = torch.zeros((1, 1, N2, D), dtype=dtype, device=dev)
    cmp_bt = torch.zeros((B, 1), dtype=torch.int32, device=dev)
    cmp_idx = torch.zeros((T1, N2, DEFAULT_BLOCK_I), dtype=torch.int32, device=dev)
    sinks = inp["sinks_npu"].to(torch.float32)
    seqused_kv = inp["seqused_kv_npu"]

    func = _get_kernel(
        batch=B,
        max_seq=S_max,
        total_tokens=T1,
        ori_block_num=ori_kv.shape[0],
        ori_block_size=ori_kv.shape[1],
        ori_table_len=ori_bt.shape[1],
        cmp_block_num=1,
        cmp_block_size=1,
        cmp_table_len=1,
        n_heads=N1,
        n_kv_heads=N2,
        head_dim=D,
        topk_cmp=0,
        cmp_ratio=4,
        scenario=1,
        ori_win_left=cfg["ori_win_left"],
        softmax_scale=float(cfg["softmax_scale"]),
        dtype="bfloat16" if dtype == torch.bfloat16 else "float16",
        core_num=DEFAULT_CORE_NUM,
    )
    raw = [
        q,
        ori_kv,
        ori_bt,
        cmp_kv,
        cmp_bt,
        cmp_idx,
        q_prefix,
        act_q_lens,
        seqused_kv,
        sinks,
        md,
    ]
    args = [a.contiguous() for a in raw] if contiguous else raw
    return func, args


def bench_eager(func, args, iters=100):
    for _ in range(10):
        func(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        func(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_graph(func, args, iters=100):
    npu = torch_npu.npu
    if not hasattr(npu, "NPUGraph") or not hasattr(npu, "graph"):
        print(
            "  [graph] NPUGraph/graph not exposed -> try torchair/torch.compile (see notes)"
        )
        return None
    for _ in range(10):  # warmup before capture (required)
        func(*args)
    torch.npu.synchronize()
    g = npu.NPUGraph()
    try:
        with npu.graph(g):
            func(*args)  # captured: torch.empty allocs + ACL launch
    except Exception as e:  # noqa: BLE001
        print(f"  [graph] CAPTURE FAILED: {type(e).__name__}: {e}")
        print("          (likely a host-sync inside the captured region; that is the")
        print("           thing to fix to make this op graph-capturable.)")
        return None
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_metadata(inp, cfg, iters=100):
    """Time the SEPARATE metadata operator (TileLang port) alone. Out of the
    sharedkv perf scope -- shown only so it is not silently folded into the
    sharedkv host prep (the bug that made an earlier (A) read ~53ms for prefill)."""
    for _ in range(3):  # fewer warmups: the prefill metadata port is ~53ms/call
        build_metadata(inp, cfg)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        build_metadata(inp, cfg)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_prep(inp, cfg, md, iters=100):
    """Time the sharedkv host PREP segment BEFORE the `func(...)` line, with the
    metadata operator's output PASSED IN (md, precomputed once) -- so this is the
    genuine sharedkv prep ONLY, NOT the separate metadata op: scenario/shape math,
    the .item() host sync, dummies, .to(device), the cached _get_kernel lookup, and
    the 11 .contiguous(). These are the lines the user's model says graph-mode
    serving "moves out of the loop". HOISTABLE (run once), NOT what graph capture
    removes. build_swa_kernel_call(inp, cfg, md) is the faithful replica. Warmup
    first so the JIT compile + kernel cache are hot (per-call is then a dict
    lookup, the real serving cost)."""
    for _ in range(10):
        build_swa_kernel_call(inp, cfg, md)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        build_swa_kernel_call(inp, cfg, md)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_prep_no_contig(inp, cfg, md, iters=100):
    """(A1) The prep BEFORE the .contiguous(): scenario/shape math, the .item()
    host sync, dummy allocs, .to(device), the cached _get_kernel lookup. Excludes
    the 11 .contiguous() (build with contiguous=False)."""
    for _ in range(10):
        build_swa_kernel_call(inp, cfg, md, contiguous=False)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        build_swa_kernel_call(inp, cfg, md, contiguous=False)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_contiguous(inp, cfg, md, iters=100):
    """(A2) JUST the 11 .contiguous() calls on the prepared args (built once)."""
    _, raw = build_swa_kernel_call(inp, cfg, md, contiguous=False)
    for _ in range(10):
        [a.contiguous() for a in raw]
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        [a.contiguous() for a in raw]
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def build_full_api_call(inp, cfg, md):
    """Return a thunk running the REAL public api.sparse_attn_sharedkv once
    (scenario 1 / SWA), as a naive eager serving loop would: ALL the sharedkv prep
    (the bench_prep segment) PLUS func() PLUS device, every call. metadata=md is
    PASSED IN (the separate op, precomputed once -- as perf_compare does), so this
    is the sharedkv-only per-call cost and should be ~= prep + func_eager (it does
    NOT include the ~53ms metadata op)."""
    q = inp["q_npu"]
    kwargs = dict(
        ori_kv=inp["ori_pa_npu"],
        ori_block_table=inp["ori_bt_npu"],
        cu_seqlens_q=inp["cu_seqlens_q_npu"],
        seqused_kv=inp["seqused_kv_npu"],
        sinks=inp["sinks_npu"],
        metadata=md,
        softmax_scale=float(cfg["softmax_scale"]),
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q="TND",
        layout_kv="PA_ND",
        return_softmax_lse=True,
    )  # cmp_kv / cmp_sparse_indices omitted -> api resolves scenario 1 (SWA)

    def run():
        return sparse_attn_sharedkv(q, **kwargs)

    return run


def bench_thunk(fn, iters=100):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def probe_training(func, args) -> None:
    """Does a gradient flow through the op? (Forward-only kernel -> no grad_fn.)"""
    print("=== training (autograd) probe ===")
    out = func(*args)
    o = out[0] if isinstance(out, (list, tuple)) else out
    print("  out.requires_grad:", getattr(o, "requires_grad", None))
    print("  out.grad_fn      :", getattr(o, "grad_fn", None))
    print("  -> grad_fn is None => the kernel is NOT autograd-integrated; gradients")
    print("     do not flow. Gradient TRAINING needs a backward kernel first.")
    print("     (Graph mode still applies to the FORWARD, e.g. inference / frozen.)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="swa_decode", choices=list(SCENARIOS))
    ap.add_argument("--iters", type=int, default=100)
    args_ = ap.parse_args()

    cfg = SCENARIOS[args_.scenario]
    assert cfg["scenario"] == 1, "this probe builds SWA (scenario 1) args only"
    inp = stage_on_npu(build_inputs(cfg, torch.bfloat16))

    probe_apis()
    # Metadata = a SEPARATE op, precomputed ONCE and passed in (as serving /
    # perf_compare do). Kept OUT of the sharedkv prep so it is not folded in.
    md = build_metadata(inp, cfg)
    func, kargs = build_swa_kernel_call(inp, cfg, md)
    probe_training(func, kargs)

    print(f"=== timing ({args_.scenario}, iters={args_.iters}) ===")

    # (M) the SEPARATE metadata operator (out of scope; shown so it is not hidden).
    meta = bench_metadata(inp, cfg, args_.iters)
    print(
        f"  (M) metadata op (SEPARATE)   per-call: {meta:.4f} ms  <- OUT of scope, own op"
    )

    # (A) sharedkv host prep BEFORE the func(...) line, metadata PASSED IN --
    #     HOISTED out of the loop by a serving loop (run once). The segment the
    #     user's model points at, now WITHOUT the metadata op.
    prep = bench_prep(inp, cfg, md, args_.iters)
    print(
        f"  (A) sharedkv prep before func() /call: {prep:.4f} ms  <- hoist removes (run once)"
    )

    # (B) func(*args) eager: the kernel-launch call itself (cython plumbing -
    #     validate/alloc/ptr/lib.call - plus device execution).
    eager = bench_eager(func, kargs, args_.iters)
    print(
        f"  (B) func(*args) eager        per-call: {eager:.4f} ms  <- the launch call"
    )

    # (C) graph replay of func(*args): device only; (B)'s host plumbing gone.
    graph = bench_graph(func, kargs, args_.iters)
    if graph is not None:
        print(
            f"  (C) graph replay of func()   per-call: {graph:.4f} ms  <- device only"
        )

    # (D) full real api call eager: sharedkv prep + func every call (naive eager
    #     serving), metadata PASSED IN -- so (D) is sharedkv-only, ~= (A)+(B).
    full = bench_thunk(build_full_api_call(inp, cfg, md), args_.iters)
    print(
        f"  (D) FULL api eager (sharedkv)/call: {full:.4f} ms  <- naive serving (~A+B)"
    )

    print()
    print("  metadata op (M) is a SEPARATE op, excluded below (out of scope).")
    print("  decomposition (does (D) ~= (A)+(B)? does graph remove (B)'s host part?):")
    print(f"    naive eager serving / call        = (D)            {full:.4f} ms")
    print(
        f"      sharedkv prep before func()     = (A)            {prep:.4f} ms   [hoist removes]"
    )
    print(f"      func() launch + device          ~ (B)            {eager:.4f} ms")
    if graph is not None:
        print(
            f"        func() HOST plumbing          ~ (B)-(C)        {eager - graph:.4f} ms   [GRAPH CAPTURE removes]"
        )
        print(
            f"        DEVICE work                   ~ (C)            {graph:.4f} ms   [irreducible]"
        )
        print(
            f"    graph-mode serving / call         ~ (C)            {graph:.4f} ms   [prep hoisted + func replayed]"
        )
        print(
            f"    => total host tax removed         = (D)-(C)        {full - graph:.4f} ms "
            f"({100 * (1 - graph / full):.0f}% of naive)"
        )

    # ---- WHICH host part dominates? Split (A) into (A1) prep-no-contiguous and
    #      (A2) the 11 .contiguous(); (B)-(C) is the JIT-wrapper + cython forward
    #      plumbing. Only meaningful when the host is EXPOSED -- i.e. DECODE (tiny
    #      device). On PREFILL the func() plumbing OVERLAPS the ~1.75ms device, so
    #      (B)-(C)~=0 there (hidden), and only the serial prep shows in wall time.
    a1 = bench_prep_no_contig(inp, cfg, md, args_.iters)
    a2 = bench_contiguous(inp, cfg, md, args_.iters)
    print()
    print("  host-part breakdown (which of the 3 dominates; read on DECODE = exposed):")
    print(
        f"    (A1) prep before func, NO contiguous  = {a1:.4f} ms  [shape/dummies/_get_kernel/.item]"
    )
    print(
        f"    (A2) the 11 .contiguous()             = {a2:.4f} ms  [near-free if already contiguous]"
    )
    if graph is not None:
        print(
            f"    (P)  JIT wrapper + cython forward     = (B)-(C) {eager - graph:.4f} ms  [validate/alloc/ptr; ACL launch ~0.03]"
        )
        print(f"    (Dev) device kernel                   = (C)     {graph:.4f} ms")
        resid = full - a1 - a2 - eager
        print(
            f"    (R)  real-api residual prep           = (D)-A1-A2-B {resid:.4f} ms  [api does a bit more than the replica]"
        )
    print(
        "    NB: on PREFILL (P) overlaps device (~0; hidden); on DECODE it is exposed."
    )

    print()
    print("Notes:")
    print(
        "  * (M) the metadata op is a SEPARATE operator (its own ~53ms prefill cost),"
    )
    print(
        "    timed + passed in here as serving/perf_compare do; it is NOT part of the"
    )
    print("    sharedkv host prep and is excluded from (A)/(D)/the decomposition.")
    print("  * The user's model = 'graph hoists the lines before func()'. That is (A),")
    print("    and (A) IS hoisted -- but by hoisting, not by capture. Graph CAPTURE")
    print("    additionally removes (B)-(C), the host plumbing INSIDE func(). Both")
    print("    together is why graph serving (C) << naive serving (D).")
    print("  * For decode (tiny device (C)) the host tax (A)+(B-C) is almost the whole")
    print(
        "    latency -> graph mode is the 9.5% fix. For prefill (C) dominates -> graph"
    )
    print("    barely helps (device-bound, already at Ascend C parity).")
    print("  * PREFILL needs ONE graph per shape (static); DECODE is naturally fixed-")
    print("    shape -> the canonical graph use case (decode loop).")
    print("  * TRAINING (gradients): blocked by no-backward (see probe above), not by")
    print("    graph mode. The forward can still be graph-captured (inference/frozen).")


if __name__ == "__main__":
    main()
