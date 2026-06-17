"""Cube-side paged-gather probe -- now with an isolation control.

First run: cube indirect gather compiled + ran (no crash) but the output
was ~zero (max_diff = 1.0). That could mean the indirect gather didn't
actually fill kv_l1, OR the "extract L1 back to GM" path (identity gemm,
the only route since L1->GM copy is unsupported) is itself broken. The
two were coupled, so we isolate.

Two modes, same extraction chain (ident @ kv_l1 -> L0C -> GM):
  - mode "block":  fill kv_l1 with a plain block copy KV[0, 0:N, 0, :]
                   (the example's proven GM->L1 form). Control.
  - mode "gather": fill kv_l1 with the two-step paged indirect gather.

Read-out:
  - block PASS, gather FAIL -> extraction chain is fine; the indirect
    gather is the (fixable) bug.
  - block FAIL              -> the extraction chain (identity gemm /
    ident copy / L0C->GM) is the bug; the gather may well be fine and we
    need a different verification.

Run on an Ascend NPU host:
    python probe_cube_gather.py
"""

import sys

import torch

import tilelang
from tilelang import language as T

tilelang.disable_cache()
tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[4])
def build_probe(block_num, block_size, N, D, table_len, mode, dtype="bfloat16"):
    indices_dtype = "int32"
    accum_dtype = "float"
    elem = 2  # bf16 / fp16 bytes
    kv_shape = [block_num, block_size, 1, D]
    bt_shape = [1, table_len]
    idx_shape = [1, N]
    ident_shape = [N, N]
    out_shape = [N, D]
    l1_kv = 0
    l1_ident = N * D * elem  # kv_l1 [N, D] then ident_l1 [N, N]

    # NOTE: `mode` is branched at the Python level (outside @T.prim_func) --
    # a body-level `if mode ==` would become a TIR If, not a compile-time
    # specialization. So define one prim_func per mode.
    if mode == "block":

        @T.prim_func
        def probe(
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            block_table: T.Tensor(bt_shape, indices_dtype),  # type: ignore
            indices: T.Tensor(idx_shape, indices_dtype),  # type: ignore
            ident: T.Tensor(ident_shape, dtype),  # type: ignore
            Output: T.Tensor(out_shape, dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                kv_l1 = T.alloc_L1([N, D], dtype)
                ident_l1 = T.alloc_L1([N, N], dtype)
                acc = T.alloc_L0C([N, D], accum_dtype)
                T.annotate_address({kv_l1: l1_kv, ident_l1: l1_ident, acc: 0})
                with T.Scope("C"):
                    # Control: proven GM->L1 block copy (one whole block).
                    T.copy(KV[0, 0:N, 0, :], kv_l1)
                    T.barrier_all()
                    T.copy(ident, ident_l1)
                    T.barrier_all()
                    T.gemm_v0(ident_l1, kv_l1, acc, init=True)
                    T.barrier_all()
                    T.copy(acc, Output)

        return probe

    if mode == "loop_direct":

        @T.prim_func
        def probe(
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            block_table: T.Tensor(bt_shape, indices_dtype),  # type: ignore
            indices: T.Tensor(idx_shape, indices_dtype),  # type: ignore
            ident: T.Tensor(ident_shape, dtype),  # type: ignore
            Output: T.Tensor(out_shape, dtype),  # type: ignore
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                kv_l1 = T.alloc_L1([N, D], dtype)
                ident_l1 = T.alloc_L1([N, N], dtype)
                acc = T.alloc_L0C([N, D], accum_dtype)
                T.annotate_address({kv_l1: l1_kv, ident_l1: l1_ident, acc: 0})
                with T.Scope("C"):
                    # Per-row, 2D single-row slices, FIXED block (no indirection).
                    for i in range(N):
                        T.copy(KV[0, i : i + 1, 0, :], kv_l1[i : i + 1, :])
                    T.barrier_all()
                    T.copy(ident, ident_l1)
                    T.barrier_all()
                    T.gemm_v0(ident_l1, kv_l1, acc, init=True)
                    T.barrier_all()
                    T.copy(acc, Output)

        return probe

    @T.prim_func
    def probe(
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        block_table: T.Tensor(bt_shape, indices_dtype),  # type: ignore
        indices: T.Tensor(idx_shape, indices_dtype),  # type: ignore
        ident: T.Tensor(ident_shape, dtype),  # type: ignore
        Output: T.Tensor(out_shape, dtype),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            kv_l1 = T.alloc_L1([N, D], dtype)
            ident_l1 = T.alloc_L1([N, N], dtype)
            acc = T.alloc_L0C([N, D], accum_dtype)
            T.annotate_address({kv_l1: l1_kv, ident_l1: l1_ident, acc: 0})
            with T.Scope("C"):
                # Test: two-step paged indirect gather, per row, on the cube.
                for i in range(N):
                    logical = indices[0, i]
                    phys = block_table[0, logical // block_size]
                    row = logical % block_size
                    T.copy(KV[phys, row : row + 1, 0, :], kv_l1[i : i + 1, :])
                T.barrier_all()
                T.copy(ident, ident_l1)
                T.barrier_all()
                T.gemm_v0(ident_l1, kv_l1, acc, init=True)
                T.barrier_all()
                T.copy(acc, Output)

    return probe


def _run(mode, block_num, block_size, N, D, table_len, KV, block_table, indices, ident):
    if mode in ("block", "loop_direct"):
        golden = KV[0, 0:N, 0, :].to(torch.float32)
    else:
        golden = torch.zeros(N, D, dtype=torch.bfloat16)
        for i in range(N):
            logical = int(indices[0, i])
            phys = int(block_table[0, logical // block_size])
            row = logical % block_size
            golden[i] = KV[phys, row, 0, :]
        golden = golden.to(torch.float32)

    kernel = build_probe(block_num, block_size, N, D, table_len, mode)
    with torch.device("npu"):
        out = kernel(KV.npu(), block_table.npu(), indices.npu(), ident.npu())
        torch.npu.synchronize()
    out_cpu = out.cpu().to(torch.float32)
    max_diff = (out_cpu - golden).abs().max().item()
    ok = torch.allclose(out_cpu, golden, rtol=1e-2, atol=1e-2)
    print(f"[{mode:6}] max_abs_diff = {max_diff:.6f}  {'PASS' if ok else 'FAIL'}")
    return ok


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

    torch.manual_seed(0)
    block_num, block_size, N, D, table_len = 8, 64, 64, 512, 4

    KV = (torch.rand(block_num, block_size, 1, D) * 2 - 1).to(torch.bfloat16)
    perm = torch.randperm(block_num)[:table_len].to(torch.int32)
    block_table = perm.reshape(1, table_len)
    indices = torch.randint(0, table_len * block_size, (1, N), dtype=torch.int32)
    ident = torch.eye(N, dtype=torch.bfloat16)

    args = (block_num, block_size, N, D, table_len, KV, block_table, indices, ident)
    block_ok = _run("block", *args)
    direct_ok = _run("loop_direct", *args)
    gather_ok = _run("gather", *args)

    print("-" * 56)
    if gather_ok:
        print("cube indirect gather works -- B can proceed.")
    elif not block_ok:
        print("extraction chain itself is broken; ignore the rest.")
    elif not direct_ok:
        print(
            "even 2D per-row L1 write fails on cube -> no per-row cube->L1 path; "
            "route via vector gather -> UB -> L1 instead."
        )
    else:
        print(
            "2D per-row L1 write is OK -> the INDIRECT addressing (KV[phys,row], "
            "runtime scalars on cube) is the bug."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
