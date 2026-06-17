"""Minimal reproduction for the tilelang-ascend upstream issue:
cube-side `T.copy` (GM->L1) silently drops per-row writes.

A whole-buffer GM->L1 copy on the cube core works, but writing per-row
into sub-rows of an L1 buffer (`kv_l1[i:i+1, :]`) silently produces zeros:
it compiles, runs without error, but the L1 rows are never filled. The
analogous per-row gather into a UB buffer on the vector core works fine
(it is what the bundled sparse-attention kernels do). This blocks moving a
PageAttention paged-KV gather onto the cube core; Ascend C does that via
`DataCopyPA` (block-table-resolved gather GM->L1), so the hardware supports
it -- TileLang just doesn't expose a per-row / indirect GM->L1 path.

`Output` should equal `KV` in both modes. The identity gemm is only there
because there is no L1->GM copy, so the L1 contents must be read back
through L0C.

Expected on an Atlas A3 / Ascend 910_93:
    per_row=False: max_abs_diff=0.0000 PASS
    per_row=True : max_abs_diff~1.0    FAIL   (Output all zeros)

NOTE: the full, NPU-verified probe is ``probe_cube_gather.py`` (paged 4D
KV, block PASS / loop_direct FAIL). This file trims the paging away to
isolate the single per-row-L1-write point for the upstream issue. Do NOT
add ``from __future__ import annotations`` here -- it stringifies the
prim_func annotations and breaks TVMScript parsing.

Run on an Ascend NPU host:
    python issue_repro_cube_l1_write.py
"""

import sys

import torch

import tilelang
from tilelang import language as T

tilelang.disable_cache()
tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[2])
def build(N, D, per_row, dtype="bfloat16"):
    accum = "float"

    # `per_row` is branched at the Python level (outside @T.prim_func): a
    # body-level `if` would become a TIR If, not a compile-time choice.
    if per_row:

        @T.prim_func
        def kern(
            KV: T.Tensor([N, D], dtype),  # type: ignore
            ident: T.Tensor([N, N], dtype),  # type: ignore
            Output: T.Tensor([N, D], dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                kv_l1 = T.alloc_L1([N, D], dtype)
                ident_l1 = T.alloc_L1([N, N], dtype)
                acc = T.alloc_L0C([N, D], accum)
                T.annotate_address({kv_l1: 0, ident_l1: N * D * 2, acc: 0})
                with T.Scope("C"):
                    for i in range(N):  # per-row write into L1 sub-rows
                        T.copy(KV[i : i + 1, :], kv_l1[i : i + 1, :])
                    T.barrier_all()
                    T.copy(ident, ident_l1)
                    T.barrier_all()
                    T.gemm_v0(ident_l1, kv_l1, acc, init=True)
                    T.barrier_all()
                    T.copy(acc, Output)

        return kern

    @T.prim_func
    def kern(
        KV: T.Tensor([N, D], dtype),  # type: ignore
        ident: T.Tensor([N, N], dtype),  # type: ignore
        Output: T.Tensor([N, D], dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            kv_l1 = T.alloc_L1([N, D], dtype)
            ident_l1 = T.alloc_L1([N, N], dtype)
            acc = T.alloc_L0C([N, D], accum)
            T.annotate_address({kv_l1: 0, ident_l1: N * D * 2, acc: 0})
            with T.Scope("C"):
                T.copy(KV, kv_l1)  # whole-buffer write into L1
                T.barrier_all()
                T.copy(ident, ident_l1)
                T.barrier_all()
                T.gemm_v0(ident_l1, kv_l1, acc, init=True)
                T.barrier_all()
                T.copy(acc, Output)

    return kern


def run(per_row):
    N, D = 64, 512
    torch.manual_seed(0)
    KV = (torch.rand(N, D) * 2 - 1).to(torch.bfloat16)
    ident = torch.eye(N, dtype=torch.bfloat16)
    kernel = build(N, D, per_row)
    with torch.device("npu"):
        out = kernel(KV.npu(), ident.npu())
        torch.npu.synchronize()
    diff = (out.cpu().float() - KV.float()).abs().max().item()
    tag = "PASS" if diff < 1e-2 else "FAIL"
    print(f"per_row={str(per_row):5}: max_abs_diff={diff:.4f}  {tag}")


def main():
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - host dependent
        print(f"[fatal] torch_npu unavailable: {exc!r}", file=sys.stderr)
        return 2
    if not torch.npu.is_available():
        print(
            "[fatal] torch.npu.is_available() == False; need an NPU.", file=sys.stderr
        )
        return 2
    run(per_row=False)  # whole-buffer -> PASS (0.0)
    run(per_row=True)  # per-row L1   -> FAIL (~1.0, Output all zeros)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
