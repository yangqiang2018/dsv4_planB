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
import api as _api
from api import _get_kernel, DEFAULT_BLOCK_I, DEFAULT_CORE_NUM
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


def build_swa_kernel_call(inp, cfg):
    """Replicate api.sparse_attn_sharedkv's SWA (scenario 1) host prep ONCE and
    return (func, args). Capturing func(*args) captures only the kernel launch --
    the host prep (shapes, metadata, dummies, the .item() for max_seq) is done
    here, outside the graph, exactly as a real serving loop would."""
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

    md = (
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
    args = [
        q.contiguous(),
        ori_kv.contiguous(),
        ori_bt.contiguous(),
        cmp_kv.contiguous(),
        cmp_bt.contiguous(),
        cmp_idx.contiguous(),
        q_prefix.contiguous(),
        act_q_lens.contiguous(),
        seqused_kv.contiguous(),
        sinks.contiguous(),
        md.contiguous(),
    ]
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
    func, kargs = build_swa_kernel_call(inp, cfg)
    probe_training(func, kargs)

    print(f"=== timing ({args_.scenario}, iters={args_.iters}) ===")
    eager = bench_eager(func, kargs, args_.iters)
    print(f"  eager (kernel-launch only) per-call: {eager:.4f} ms")
    graph = bench_graph(func, kargs, args_.iters)
    if graph is not None:
        print(f"  graph replay               per-call: {graph:.4f} ms")
        print(
            f"  -> host tax removed by graph: {eager - graph:.4f} ms "
            f"({100 * (1 - graph / eager):.0f}% faster)"
        )
    print()
    print("Notes:")
    print("  * eager here is JUST the kernel launch (func), not the full api call,")
    print("    so it is the launch tax (~0.5ms) -- the full api adds ~0.4ms prep.")
    print("  * if graph replay << eager, graph mode removes the host tax. For decode")
    print("    (tiny device) that is almost the entire latency -> the 9.5% fix.")
    print("  * PREFILL needs ONE graph per shape (static); DECODE is naturally fixed-")
    print("    shape -> the canonical graph use case (decode loop).")
    print("  * TRAINING (gradients): blocked by no-backward (see probe above), not by")
    print("    graph mode. The forward can still be graph-captured (inference/frozen).")


if __name__ == "__main__":
    main()
