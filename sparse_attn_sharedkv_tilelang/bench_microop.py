"""Micro-benchmarks: per-op cost on the vector core. Each case is its own
explicit @T.prim_func -- closure-injected bodies share one AST, the JIT cache
keys on the AST, and every case silently reruns the first kernel. Run on NPU:
    python sparse_attn_sharedkv_tilelang/bench_microop.py
"""

import time

import tilelang
import torch
from tilelang import language as T

tilelang.disable_cache()
tilelang.cache.clear_cache()

H, W, D, REPS, LAUNCH = 32, 128, 512, 2000, 5


def _shell(body_name: str):
    @tilelang.jit
    def _make():
        @T.prim_func
        def k(
            Src: T.Tensor([H, D], "float"),  # type: ignore[valid-type]
            Out: T.Tensor([1], "float"),  # type: ignore[valid-type]
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a = T.alloc_ub([H, W], "float")
                b = T.alloc_ub([H, W], "float")
                blk = T.alloc_ub([16, D], "float")
                msk = T.alloc_ub([32], "uint8")
                T.annotate_address({a: 0, b: 16 * 1024, blk: 32 * 1024, msk: 64 * 1024})
                with T.Scope("V"):
                    T.tile.fill(a, 1.0)
                    T.tile.fill(b, 2.0)
                    T.barrier_all()
                    T.tile.compare(msk, b[0, :], T.float32(0.0), "GT")
                    T.barrier_all()
                    if body_name == "noise":
                        pass
                    elif body_name == "mul_fused":
                        for _ in T.serial(REPS):
                            T.tile.mul(a, a, b)
                    elif body_name == "mul_split":
                        for _ in T.serial(REPS):
                            for h in range(H):
                                T.tile.mul(a[h, :], a[h, :], b[h, :])
                    elif body_name == "mul_scalar":
                        for _ in T.serial(REPS):
                            for h in range(H):
                                T.tile.mul(a[h, :], a[h, :], b[h, 0])
                    elif body_name == "select":
                        for _ in T.serial(REPS):
                            for h in range(H):
                                T.tile.select(
                                    a[h, :],
                                    msk,
                                    b[h, :],
                                    -1.0,
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                    elif body_name == "select_fused":
                        # whole-tile select with the 128-bit row mask cycling
                        # over 32 rows -- correctness of the cycling must be
                        # checked separately before kernel use.
                        for _ in T.serial(REPS):
                            T.tile.select(a, msk, b, -1.0, "VSEL_TENSOR_SCALAR_MODE")
                    elif body_name == "add_fused":
                        for _ in T.serial(REPS):
                            T.tile.add(a, a, b)
                    elif body_name == "dma_rows":
                        for _ in T.serial(REPS):
                            for r in range(16):
                                T.copy(Src[r, :], blk[r, :])
                    elif body_name == "dma_block":
                        for _ in T.serial(REPS):
                            T.copy(Src[0:16, :], blk)
                    elif body_name == "flag":
                        for _ in T.serial(REPS):
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                    elif body_name == "barrier":
                        for _ in T.serial(REPS):
                            T.barrier_all()
                    T.barrier_all()
                    T.copy(a[0, 0:1], Out[0:1])

        return k

    return _make()


CASES = [
    ("noise", 0),
    ("mul_fused", 1),
    ("mul_split", H),
    ("mul_scalar", H),
    ("select", H),
    ("select_fused", 1),
    ("add_fused", 1),
    ("dma_rows", 16),
    ("dma_block", 1),
    ("flag", 1),
    ("barrier", 1),
]


def main():
    src = torch.randn(H, D, dtype=torch.float32).npu()
    out = torch.zeros(1, dtype=torch.float32).npu()
    res = {}
    for name, ops in CASES:
        fn = _shell(name)
        for _ in range(5):
            fn(src, out)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(LAUNCH):
            fn(src, out)
        torch.npu.synchronize()
        us = (time.perf_counter() - t0) / LAUNCH * 1e6
        res[name] = (us, ops)
        print(f"{name:12s} {us:10.1f} us/launch ({REPS}x{ops} ops)")
    base = res["noise"][0]
    print(f"\n-- net ns/op (minus noise {base:.1f} us) --")
    for n, (us, ops) in res.items():
        if ops:
            print(f"{n:12s} {(us - base) / (REPS * ops) * 1000:9.2f} ns/op")


if __name__ == "__main__":
    main()
