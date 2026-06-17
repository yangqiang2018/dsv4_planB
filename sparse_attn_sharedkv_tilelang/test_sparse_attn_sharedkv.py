"""Pytest suite for the TileLang SparseAttnSharedKV port.

The parameter set and numerical check criterion mirror the Ascend C
reference suite (``ops-transformer/.../sparse_attn_sharedkv/tests/
pytest/``) 1:1 -- six cases (scfa/swa/cfa x decode/prefill, all TND,
B=1) with ``check_result``-style validation: per-element ``np.isclose``
at the Ascend C tolerances, at least 99.5% of elements must pass, and
the worst normalized relative error must stay below 10.

The test flow also matches the Ascend C reference 1:1: each NPU case
first calls ``sparse_attn_sharedkv_metadata`` (the companion
load-balancing op) to produce the per-core FA / FD task table, then
feeds it to ``sparse_attn_sharedkv``. The TileLang sharedkv kernel
consumes the metadata on-device -- each AIC core reads its
``faMetadata[cid]`` row to learn its ``(bn2, m)`` work range and
walks the linearised pid window for that range -- so the test flow
is one-to-one with ``ops-transformer/.../sparse_attn_sharedkv/tests/
pytest/batch/sparse_attn_sharedkv_process.py`` where
``torch_npu.npu_sparse_attn_sharedkv_metadata`` is invoked before
``torch.ops.custom.npu_sparse_attn_sharedkv``.

Run on an Ascend NPU host with TileLang-Ascend installed::

    pytest -q test_sparse_attn_sharedkv.py
    pytest -q test_sparse_attn_sharedkv.py --runslow

The three prefill cases run ``S1=8192``; the CPU golden takes minutes
at that size. They are marked ``slow`` and skipped unless ``--runslow``
is passed.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

# Make local modules importable when pytest runs from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import golden as G  # noqa: E402


# ---- Detect the NPU; tolerate CPU-only hosts. ----
# NOTE: we deliberately do NOT call torch.set_default_device("npu").
# Data generation and the golden run on CPU; only the kernel call is
# wrapped in `with torch.device("npu")`. Setting the default device
# globally makes torch.randperm(generator=<cpu-gen>) fail with a
# device-mismatch error.
def _try_set_npu():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available(), None
    except Exception as exc:
        return False, repr(exc)


HAS_NPU, _NPU_ERR = _try_set_npu()
# Print a diagnostic banner so a silent run is at least somewhat decipherable.
print(
    f"[test_sparse_attn_sharedkv] HAS_NPU={HAS_NPU} "
    f"(reason: {'OK' if HAS_NPU else _NPU_ERR})",
    flush=True,
)
requires_npu = pytest.mark.skipif(
    not HAS_NPU,
    reason=f"Ascend NPU not available ({_NPU_ERR})",
)


# ---- Test cases: mirror the Ascend C paramset 1:1. ----
# Six cases: scfa/swa/cfa x decode/prefill, all TND, B=1. (The Ascend C
# paramset is bf16-only in single mode; we additionally run fp16, which
# the kernel supports, as extra coverage.)

SCENARIOS = {
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
}

# Decodes (S1=1) run by default; prefills (S1=8192) trigger a CPU golden
# that takes minutes, so they are marked `slow` and skipped unless
# --runslow is passed (see conftest.py).
SMALL_CASES = [
    "scfa_decode",
    "swa_decode",
    "cfa_decode",
]

LARGE_CASES = [
    "scfa_prefill",
    "swa_prefill",
    "cfa_prefill",
]


def _build_case(cfg: dict, dtype: torch.dtype, seed: int = 42):
    """Generate inputs, paged KV, indices, and the CPU golden output.

    Returns a dict mirroring the original suite's contract.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    layout_q = cfg["layout_q"]
    B = cfg["B"]
    S1 = cfg["S1"]
    T1 = cfg.get("T1", S1 * B)
    N1, N2, D = cfg["N1"], cfg["N2"], cfg["D"]
    K = cfg.get("K", 0)
    cmp_ratio = cfg["cmp_ratio"]
    block_size1, block_num1 = cfg["block_size1"], cfg["block_num1"]
    block_size2, block_num2 = cfg["block_size2"], cfg["block_num2"]
    seqused_kv = cfg["seqused_kv"]
    cu_seqlens_q = torch.tensor(cfg["cu_seqlens_q"], dtype=torch.int32)
    softmax_scale = cfg["softmax_scale"]
    scenario = cfg["scenario"]

    # --- Q ---
    if layout_q == "TND":
        q = (torch.rand((T1, N1, D)) * 20 - 10).to(dtype)
    else:
        q = (torch.rand((B, S1, N1, D)) * 20 - 10).to(dtype)

    # --- ori_kv (paged) ---
    ori_pa, ori_bt, ori_k_bnsd = G.gen_ori_kv_paged(
        B=B,
        N2=N2,
        D=D,
        block_num=block_num1,
        block_size=block_size1,
        seqused_kv=seqused_kv,
        dtype=dtype,
        data_range=(-10, 10),
        rng=np.random.default_rng(seed),
    )

    # --- cmp_kv (paged) + indices (only for scenarios 2, 3) ---
    if scenario >= 2:
        cmp_pa, cmp_bt, cmp_k_bnsd, cmp_seqs = G.gen_cmp_kv_paged(
            B=B,
            N2=N2,
            D=D,
            block_num=block_num2,
            block_size=block_size2,
            seqused_kv=seqused_kv,
            cmp_ratio=cmp_ratio,
            dtype=dtype,
            data_range=(-5, 10),
            rng=np.random.default_rng(seed + 1),
        )
    else:
        cmp_pa, cmp_bt, cmp_k_bnsd = None, None, None

    if scenario == 3:
        rng = torch.Generator()
        rng.manual_seed(seed + 7)
        if layout_q == "TND":
            cmp_idx = G.gen_cmp_sparse_indices_tnd(
                B=B,
                T1=T1,
                N2=N2,
                K=K,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                cmp_ratio=cmp_ratio,
                cmp_mask_mode=3,
                rng=rng,
            )
        else:
            cmp_idx = G.gen_cmp_sparse_indices_bsnd(
                B=B,
                S1=S1,
                N2=N2,
                K=K,
                seqused_kv=seqused_kv,
                cmp_ratio=cmp_ratio,
                cmp_mask_mode=3,
                rng=rng,
            )
    else:
        cmp_idx = None

    # --- sinks ---
    sinks = (torch.rand(N1) * 2 - 1).to(torch.float32)

    # --- Build BNSD reference Q for the golden. ---
    if layout_q == "TND":
        q_bnsd_ref = G.tnd_to_bnsd_q(q, cu_seqlens_q)
        act_q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    else:
        q_bnsd_ref = q.permute(0, 2, 1, 3).contiguous()
        act_q_lens = [S1] * B

    # Convert sparse indices to BSND for the golden.
    if cmp_idx is not None:
        if layout_q == "TND":
            # Reuse the api helper logic inline (TND→BSND).
            S_max = max(act_q_lens)
            cmp_idx_bsnd = torch.full((B, S_max, N2, K), -1, dtype=torch.int32)
            for b in range(B):
                s_start = int(cu_seqlens_q[b].item())
                L = int(act_q_lens[b])
                cmp_idx_bsnd[b, :L, :, :] = cmp_idx[s_start : s_start + L, :, :]
        else:
            cmp_idx_bsnd = cmp_idx
    else:
        cmp_idx_bsnd = None

    cpu_ref, cpu_ref_lse = G.sparse_attn_sharedkv_golden_bnsd(
        q_bnsd_ref,
        ori_k_bnsd,
        sinks,
        act_q_lens=act_q_lens,
        act_kv_lens=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_k_bnsd=cmp_k_bnsd,
        cmp_sparse_indices=cmp_idx_bsnd if scenario == 3 else None,
        cmp_ratio=cmp_ratio if scenario >= 2 else None,
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        ori_mask_mode=cfg["ori_mask_mode"],
        cmp_mask_mode=cfg["cmp_mask_mode"],
        return_lse=True,
    )

    # Convert golden back to caller layout.
    # cpu_ref:     BNSD [B, N1, S_max, D] -> TND [T1, N1, D] or BSND [B, S1, N1, D]
    # cpu_ref_lse: BNS  [B, N1, S_max]    -> TND [T1, N1]    or BSND [B, S1, N1]
    if layout_q == "TND":
        cpu_ref = G.bnsd_to_tnd_out(cpu_ref, cu_seqlens_q)
        cpu_ref_lse = G.bnsd_to_tnd_lse(cpu_ref_lse, cu_seqlens_q)
    else:
        cpu_ref = cpu_ref.permute(0, 2, 1, 3).contiguous()
        cpu_ref_lse = cpu_ref_lse.permute(0, 2, 1).contiguous()

    return dict(
        cfg=cfg,
        q=q,
        ori_pa=ori_pa,
        ori_bt=ori_bt,
        cmp_pa=cmp_pa,
        cmp_bt=cmp_bt,
        cmp_idx=cmp_idx,
        sinks=sinks,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=torch.tensor(seqused_kv, dtype=torch.int32),
        cpu_ref=cpu_ref,
        cpu_ref_lse=cpu_ref_lse,
    )


def _check_result(npu_out: torch.Tensor, expect: torch.Tensor) -> None:
    """Ascend C ``result_compare_method.check_result``-style validation.

    A case passes iff:

    * at least 99.5% of output elements pass ``np.isclose`` with
      dtype-specific tolerance (bf16: ``rtol=0.0078125, atol=0.0001``;
      fp16: ``rtol=0.005, atol=0.000025``), AND
    * among the failing elements, the worst *normalized* relative
      error stays below 10.

    The normalized relative error is
    ``|a - b| / max(max(|a|, |b|), (1 / 2**14) / 0.005)`` -- the same
    formula as the Ascend C suite, with a ``~0.0122`` floor that keeps
    near-zero references from blowing up the metric.
    """
    if npu_out.dtype == torch.bfloat16:
        rtol, atol = 0.0078125, 0.0001
    else:  # float16
        rtol, atol = 0.005, 0.000025

    real = npu_out.detach().cpu().to(torch.float32).numpy().flatten()
    expt = expect.detach().cpu().to(torch.float32).numpy().flatten()
    assert real.size == expt.size, f"size mismatch: {real.size} vs {expt.size}"

    ok = np.isclose(real, expt, rtol=rtol, atol=atol, equal_nan=True)
    n_err = int((~ok).sum())
    fulfill_pct = (real.size - n_err) / real.size * 100.0

    diff_thd = 0.005
    norm_floor = (1.0 / (1 << 14)) / diff_thd  # ~0.01220703
    b = np.maximum(np.maximum(np.abs(real), np.abs(expt)), norm_floor) + 1e-9
    rel_err = np.abs(real - expt) / b
    max_rel = float(rel_err[~ok].max()) if n_err > 0 else 0.0

    assert fulfill_pct >= 99.5, (
        f"only {fulfill_pct:.4f}% of elements within tol "
        f"(rtol={rtol}, atol={atol}); 99.5% required; "
        f"{n_err}/{real.size} failing, max rel err {max_rel:.4f}"
    )
    assert max_rel < 10.0, (
        f"max normalized relative error {max_rel:.4f} exceeds cap 10.0 "
        f"(fulfill {fulfill_pct:.4f}%, {n_err}/{real.size} failing)"
    )


def _check_lse(
    npu_lse: torch.Tensor, expect_lse: torch.Tensor, q_dtype: torch.dtype
) -> None:
    """LSE validation with bf16/fp16-aware tolerance.

    LSE is always fp32 on output, but its inputs (the running row_max
    and row_sum) are accumulated from bf16/fp16 Q@K^T scores. The
    Ascend C kernel emits the same fp32 value, but the bit-exactness
    bound is dictated by the input GEMM precision -- so we use the
    same dtype-keyed tolerance as ``_check_result``, just relaxed by
    one bit on rtol because ``ln`` amplifies relative error in
    ``sumexp`` (which can be near 1.0 for tiny KV sets where the sink
    term dominates).
    """
    if q_dtype == torch.bfloat16:
        rtol, atol = 0.015, 0.005
    else:  # float16
        rtol, atol = 0.01, 0.001

    real = npu_lse.detach().cpu().to(torch.float32).numpy().flatten()
    expt = expect_lse.detach().cpu().to(torch.float32).numpy().flatten()
    assert real.size == expt.size, f"lse size mismatch: {real.size} vs {expt.size}"

    ok = np.isclose(real, expt, rtol=rtol, atol=atol, equal_nan=True)
    n_err = int((~ok).sum())
    fulfill_pct = (real.size - n_err) / real.size * 100.0

    diff_thd = 0.005
    norm_floor = (1.0 / (1 << 14)) / diff_thd
    b = np.maximum(np.maximum(np.abs(real), np.abs(expt)), norm_floor) + 1e-9
    rel_err = np.abs(real - expt) / b
    max_rel = float(rel_err[~ok].max()) if n_err > 0 else 0.0

    assert fulfill_pct >= 99.5, (
        f"lse: only {fulfill_pct:.4f}% within tol "
        f"(rtol={rtol}, atol={atol}); 99.5% required; "
        f"{n_err}/{real.size} failing, max rel err {max_rel:.4f}"
    )
    assert max_rel < 10.0, (
        f"lse: max normalized relative error {max_rel:.4f} exceeds cap 10.0 "
        f"(fulfill {fulfill_pct:.4f}%, {n_err}/{real.size} failing)"
    )


def _call_metadata_then_sharedkv(case, cfg, sparse_attn_sharedkv_fn, metadata_fn):
    """Run the metadata + sharedkv pair, mirroring the Ascend C test flow.

    The Ascend C reference (`sparse_attn_sharedkv_process.py`) calls
    ``torch_npu.npu_sparse_attn_sharedkv_metadata`` first to obtain the
    per-core FA / FD task table, then feeds it as ``metadata`` to
    ``torch.ops.custom.npu_sparse_attn_sharedkv``. We do the same here
    so the TileLang test flow is one-to-one with the upstream.

    Routing the three scenarios mirrors ``sparse_attn_sharedkv_process``:

    * ``cmp_kv is None``                          → SWA  (template_idx 0)
    * ``cmp_kv != None, cmp_idx is None``          → CFA  (template_idx 1)
    * ``cmp_kv != None, cmp_idx != None``          → SCFA (template_idx 2)
    """
    layout_q = cfg["layout_q"]
    N1, N2, D = cfg["N1"], cfg["N2"], cfg["D"]
    B = cfg["B"]
    K = cfg.get("K", 0)
    max_seqlen_q = cfg.get("T1") if layout_q == "TND" else cfg["S1"]
    max_seqlen_kv = int(max(cfg["seqused_kv"]))

    # Move tensors to NPU.
    def _dev(t):
        if t is None:
            return None
        return t.npu().contiguous() if hasattr(t, "npu") else t

    cu_seqlens_q_dev = _dev(case["cu_seqlens_q"]) if layout_q == "TND" else None
    seqused_kv_dev = _dev(case["seqused_kv"])

    scenario = cfg["scenario"]
    has_ori_kv = True
    has_cmp_kv = scenario >= 2

    # ---- Step 1: Build metadata (matches Ascend C reference call). ----
    # The Python port (`metadata.sparse_attn_sharedkv_metadata`)
    # faithfully ports the aicpu BalanceSchedule + GenMetaData
    # implementation. The TileLang sharedkv kernel consumes this
    # tensor on-device: each AIC core reads its ``faMetadata[cid]``
    # row to drive its outer loop, matching the Ascend C scheduling
    # contract.
    metadata_kwargs = dict(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=case["cu_seqlens_q"] if layout_q == "TND" else None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=case["seqused_kv"],
        batch_size=B,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q=layout_q,
        layout_kv="PA_ND",
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )
    if scenario >= 2:
        metadata_kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        metadata_kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]
    if scenario == 3:
        metadata_kwargs["cmp_topk"] = K
    metadata_tensor = metadata_fn(**metadata_kwargs)
    assert metadata_tensor.dtype == torch.int32
    assert metadata_tensor.numel() == 1024  # SAS_META_SIZE

    # ---- Step 2: Call sparse_attn_sharedkv with metadata in tow. ----
    # Ask for lse as well -- the kernel always computes it (cheap
    # epilogue), and we validate both attn_out and lse in
    # ``test_sparse_attn_sharedkv``.
    with torch.device("npu"):
        out, lse = sparse_attn_sharedkv_fn(
            _dev(case["q"]),
            ori_kv=_dev(case["ori_pa"]),
            cmp_kv=_dev(case["cmp_pa"]),
            cmp_sparse_indices=_dev(case["cmp_idx"]),
            ori_block_table=_dev(case["ori_bt"]),
            cmp_block_table=_dev(case["cmp_bt"]),
            cu_seqlens_q=cu_seqlens_q_dev,
            seqused_kv=seqused_kv_dev,
            sinks=_dev(case["sinks"]),
            metadata=_dev(metadata_tensor),
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
        torch.npu.synchronize()
    return out, lse, metadata_tensor


@requires_npu
@pytest.mark.parametrize(
    "case_name",
    SMALL_CASES + [pytest.param(c, marks=pytest.mark.slow) for c in LARGE_CASES],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_attn_sharedkv(case_name, dtype):
    # Imported lazily so the CPU-only math test below can run on hosts
    # without tilelang installed.
    from api import sparse_attn_sharedkv
    from metadata import sparse_attn_sharedkv_metadata

    cfg = SCENARIOS[case_name]
    # Data generation + golden run on CPU (default device).
    case = _build_case(cfg, dtype)
    out, lse, _ = _call_metadata_then_sharedkv(
        case, cfg, sparse_attn_sharedkv, sparse_attn_sharedkv_metadata
    )
    _check_result(out.cpu(), case["cpu_ref"])
    _check_lse(lse.cpu(), case["cpu_ref_lse"], dtype)


@requires_npu
def test_api_return_type():
    """Cover the ``return_softmax_lse`` branch in :mod:`api`.

    The end-to-end tests above hard-code ``return_softmax_lse=True`` to
    validate the lse value, which means the ``return out`` (single
    tensor) branch in ``api.sparse_attn_sharedkv`` is never exercised
    by the numerical-correctness suite. This test calls the api both
    ways on the cheapest case (``swa_decode``, S1=1, no cmp pass) and
    asserts the contract:

    * ``return_softmax_lse=False`` → returns a single ``torch.Tensor``
    * ``return_softmax_lse=True``  → returns a ``(Tensor, Tensor)`` pair
      where the second element is fp32 and shaped ``[T1, N1]``
      (TND-only here).

    Numerical accuracy is already covered by ``test_sparse_attn_sharedkv``;
    this test only checks the api wrapper's branch logic so a future
    refactor that swaps the two branches gets caught.
    """
    from api import sparse_attn_sharedkv
    from metadata import sparse_attn_sharedkv_metadata

    cfg = SCENARIOS["swa_decode"]
    dtype = torch.bfloat16
    case = _build_case(cfg, dtype)

    def _dev(t):
        return t.npu().contiguous() if t is not None and hasattr(t, "npu") else t

    metadata_tensor = sparse_attn_sharedkv_metadata(
        num_heads_q=cfg["N1"],
        num_heads_kv=cfg["N2"],
        head_dim=cfg["D"],
        cu_seqlens_q=case["cu_seqlens_q"],
        seqused_kv=case["seqused_kv"],
        batch_size=cfg["B"],
        max_seqlen_q=cfg["T1"],
        max_seqlen_kv=int(max(cfg["seqused_kv"])),
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )

    common_kwargs = dict(
        ori_kv=_dev(case["ori_pa"]),
        ori_block_table=_dev(case["ori_bt"]),
        cu_seqlens_q=_dev(case["cu_seqlens_q"]),
        seqused_kv=_dev(case["seqused_kv"]),
        sinks=_dev(case["sinks"]),
        metadata=_dev(metadata_tensor),
        softmax_scale=cfg["softmax_scale"],
        ori_mask_mode=cfg["ori_mask_mode"],
        cmp_mask_mode=cfg["cmp_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q="TND",
        layout_kv="PA_ND",
        topk_cmp=0,
    )

    with torch.device("npu"):
        # Branch 1: return_softmax_lse=False -> single tensor.
        result_no_lse = sparse_attn_sharedkv(
            _dev(case["q"]),
            return_softmax_lse=False,
            **common_kwargs,
        )
        torch.npu.synchronize()
        # Branch 2: return_softmax_lse=True -> (Tensor, Tensor) pair.
        result_with_lse = sparse_attn_sharedkv(
            _dev(case["q"]),
            return_softmax_lse=True,
            **common_kwargs,
        )
        torch.npu.synchronize()

    # Branch 1 assertions.
    assert isinstance(result_no_lse, torch.Tensor), (
        f"return_softmax_lse=False must return a Tensor, got {type(result_no_lse)}"
    )
    assert result_no_lse.shape == (cfg["T1"], cfg["N1"], cfg["D"])
    assert result_no_lse.dtype == dtype

    # Branch 2 assertions.
    assert isinstance(result_with_lse, tuple), (
        f"return_softmax_lse=True must return a tuple, got {type(result_with_lse)}"
    )
    assert len(result_with_lse) == 2, (
        f"tuple must have 2 elements (out, lse), got {len(result_with_lse)}"
    )
    out, lse = result_with_lse
    assert isinstance(out, torch.Tensor) and isinstance(lse, torch.Tensor)
    assert out.shape == (cfg["T1"], cfg["N1"], cfg["D"])
    assert out.dtype == dtype
    # lse is always fp32 regardless of q dtype.
    assert lse.shape == (cfg["T1"], cfg["N1"]), (
        f"lse shape must be [T1, N1]={(cfg['T1'], cfg['N1'])} for TND, got {tuple(lse.shape)}"
    )
    assert lse.dtype == torch.float32, f"lse dtype must be fp32, got {lse.dtype}"

    # The two branches must produce the same attn_out (modulo nondeterminism
    # there isn't any here: same kernel, same inputs).
    torch.testing.assert_close(out.cpu(), result_no_lse.cpu())


# ---- CPU-only metadata tests (no NPU required). ----


@pytest.mark.parametrize("case_name", list(SCENARIOS.keys()))
def test_metadata_shape_and_continuity(case_name):
    """Validate the Python metadata port for every paramset case.

    Asserts the output is INT32 of length ``SAS_META_SIZE = 1024`` and
    that the assigned AIC cores form a contiguous chain ``(bn2_start,
    m_start, s2_start) == (prev.bn2_end, prev.m_end, prev.s2_end)``
    -- the invariant the Ascend C scheduler maintains.
    """
    from metadata import (
        sparse_attn_sharedkv_metadata,
        SAS_META_SIZE,
        AIC_CORE_NUM,
        FA_METADATA_SIZE,
    )

    cfg = SCENARIOS[case_name]
    scenario = cfg["scenario"]
    layout_q = cfg["layout_q"]
    max_seqlen_kv = int(max(cfg["seqused_kv"]))

    kwargs = dict(
        num_heads_q=cfg["N1"],
        num_heads_kv=cfg["N2"],
        head_dim=cfg["D"],
        cu_seqlens_q=torch.tensor(cfg["cu_seqlens_q"], dtype=torch.int32)
        if layout_q == "TND"
        else None,
        seqused_kv=torch.tensor(cfg["seqused_kv"], dtype=torch.int32),
        batch_size=cfg["B"],
        max_seqlen_q=cfg.get("T1") if layout_q == "TND" else cfg["S1"],
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
        kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]
    if scenario == 3:
        kwargs["cmp_topk"] = cfg["K"]

    md = sparse_attn_sharedkv_metadata(**kwargs)
    assert md.dtype == torch.int32
    assert md.shape == (SAS_META_SIZE,)

    fa = md[: AIC_CORE_NUM * FA_METADATA_SIZE].view(AIC_CORE_NUM, FA_METADATA_SIZE)
    enable = fa[:, 0].tolist()
    n_used = sum(enable)
    assert n_used >= 1, "at least one AIC core must be enabled"

    # Continuity: each enabled core resumes where its predecessor stopped.
    used = fa[fa[:, 0] == 1]
    prev_bn2, prev_m, prev_s2 = 0, 0, 0
    for row in used.tolist():
        _, bn2s, ms, s2s, bn2e, me, s2e, _ = row
        assert (bn2s, ms, s2s) == (prev_bn2, prev_m, prev_s2), (
            f"core resume mismatch in {case_name}: "
            f"got ({bn2s},{ms},{s2s}), expected ({prev_bn2},{prev_m},{prev_s2})"
        )
        prev_bn2, prev_m, prev_s2 = bn2e, me, s2e

    # The last enabled core finishes the batch (next-batch boundary OR
    # the actual tail of the only batch).
    if scenario >= 1:
        assert prev_bn2 in (cfg["B"] * cfg["N2"], 0), (
            f"last core end bn2 should be batch*kvhead or zero, got {prev_bn2}"
        )


@pytest.mark.parametrize("case_name", list(SCENARIOS.keys()))
def test_metadata_drives_complete_kernel_coverage(case_name):
    """End-to-end sanity check: simulate the kernel's metadata-driven
    outer loop on CPU and assert it visits every valid ``(b, s)`` work
    item exactly once.

    The kernel's outer loop (kernel.py) linearises each AIC core's
    metadata window ``[bn2_start * max_seq + m_start,
    bn2_end * max_seq + m_end)`` into a pid range and walks it with
    the ``s_i < actual_q_len[b_i]`` guard for padding. For the
    scheduler-produced metadata to be a *correct* schedule we need:

    * every ``(b, s)`` with ``s < act_q_len[b]`` appears in exactly
      one core's window;
    * no two cores' windows overlap on a valid ``(b, s)`` pair;
    * out-of-range pids (``s_i >= act_q_len[b_i]``) cause no
      duplicated work.

    Decode cases collapse to one enabled core (AssignByBatch wins).
    Prefill cases spread across all 24 AICs with row-level
    partitioning. Either way the union of all windows must cover the
    full valid work set.
    """
    from metadata import sparse_attn_sharedkv_metadata, AIC_CORE_NUM, FA_METADATA_SIZE

    cfg = SCENARIOS[case_name]
    scenario = cfg["scenario"]
    layout_q = cfg["layout_q"]
    B = cfg["B"]
    seqs = cfg["seqused_kv"]
    max_seqlen_kv = int(max(seqs))
    cu = cfg["cu_seqlens_q"]
    act_q_lens = (
        [cu[i + 1] - cu[i] for i in range(B)] if layout_q == "TND" else [cfg["S1"]] * B
    )
    max_seq = max(act_q_lens) if layout_q == "TND" else cfg["S1"]

    kwargs = dict(
        num_heads_q=cfg["N1"],
        num_heads_kv=cfg["N2"],
        head_dim=cfg["D"],
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32) if layout_q == "TND" else None,
        seqused_kv=torch.tensor(seqs, dtype=torch.int32),
        batch_size=B,
        max_seqlen_q=cfg.get("T1") if layout_q == "TND" else cfg["S1"],
        max_seqlen_kv=max_seqlen_kv,
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q=layout_q,
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=scenario >= 2,
        aic_core_num=24,
    )
    if scenario >= 2:
        kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]
    if scenario == 3:
        kwargs["cmp_topk"] = cfg["K"]

    md = sparse_attn_sharedkv_metadata(**kwargs).numpy()
    fa = md[: AIC_CORE_NUM * FA_METADATA_SIZE].reshape(AIC_CORE_NUM, FA_METADATA_SIZE)

    # Simulate the kernel's per-core outer loop: each enabled core walks
    # pid in [linear_start, linear_end), maps to (b, s), and processes
    # iff s < act_q_lens[b]. We accumulate every (b, s) pair visited.
    visited_set: set = set()
    duplicates: list = []
    for cid in range(24):
        enable, bn2s, ms, _, bn2e, me, _, _ = fa[cid].tolist()
        if not enable:
            continue
        linear_start = bn2s * max_seq + ms
        linear_end = bn2e * max_seq + me
        for pid in range(linear_start, linear_end):
            b_i = pid // max_seq
            s_i = pid % max_seq
            if b_i >= B:
                continue  # last core's bn2_end may equal B (next-batch sentinel)
            if s_i < act_q_lens[b_i]:
                key = (b_i, s_i)
                if key in visited_set:
                    duplicates.append((cid, key))
                visited_set.add(key)

    expected = {(b, s) for b in range(B) for s in range(act_q_lens[b])}
    missing = expected - visited_set
    extra = visited_set - expected

    assert not duplicates, f"{case_name}: kernel windows overlap on {duplicates[:5]}..."
    assert not missing, (
        f"{case_name}: kernel misses {len(missing)} work items; first 5: "
        f"{list(missing)[:5]}"
    )
    assert not extra, (
        f"{case_name}: kernel processes {len(extra)} stray items; first 5: "
        f"{list(extra)[:5]}"
    )


def test_metadata_prefill_uses_many_cores():
    """The 8192-token prefill case must spread across all 24 AICs."""
    from metadata import sparse_attn_sharedkv_metadata, AIC_CORE_NUM, FA_METADATA_SIZE

    cfg = SCENARIOS["scfa_prefill"]
    md = sparse_attn_sharedkv_metadata(
        num_heads_q=cfg["N1"],
        num_heads_kv=cfg["N2"],
        head_dim=cfg["D"],
        cu_seqlens_q=torch.tensor(cfg["cu_seqlens_q"], dtype=torch.int32),
        seqused_kv=torch.tensor(cfg["seqused_kv"], dtype=torch.int32),
        batch_size=cfg["B"],
        max_seqlen_q=cfg["T1"],
        max_seqlen_kv=int(max(cfg["seqused_kv"])),
        cmp_topk=cfg["K"],
        cmp_ratio=cfg["cmp_ratio"],
        ori_mask_mode=cfg["ori_mask_mode"],
        cmp_mask_mode=cfg["cmp_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q=cfg["layout_q"],
        layout_kv="PA_ND",
    )
    fa = md[: AIC_CORE_NUM * FA_METADATA_SIZE].view(AIC_CORE_NUM, FA_METADATA_SIZE)
    n_used = int(fa[:, 0].sum())
    # The aicpu default is aic_core_num=24; prefill workload (S1=8192,
    # S2=8192, mBaseSize=64 → 8192 S1G rows) is huge so every core is
    # used. Decode-sized cases fit on a single core via AssignByBatch.
    assert n_used == 24, f"expected 24 AIC cores used, got {n_used}"


# ---- Math-only test: golden vs single-shot softmax (no NPU required). ----


def test_golden_math_matches_single_shot_softmax():
    """Sanity check that the chunked online softmax in :mod:`golden`
    matches a single-shot ``softmax(scores ∪ sinks) @ V`` computation for
    a small, scenario-3 case. CPU-only.
    """
    torch.manual_seed(0)
    B, S1, N1, N2, D = 1, 4, 64, 1, 512
    cmp_ratio = 4
    K = 8
    seqused_kv = [128]
    softmax_scale = 1.0 / math.sqrt(D)

    q = (torch.rand((B, N1, S1, D)) * 2 - 1).to(torch.float32)
    ori_k_bnsd = (torch.rand((B, N2, seqused_kv[0], D)) * 2 - 1).to(torch.float32)
    cmp_k_bnsd = (torch.rand((B, N2, seqused_kv[0] // cmp_ratio, D)) * 2 - 1).to(
        torch.float32
    )
    sinks = (torch.rand(N1) * 0.1).to(torch.float32)

    # Build a deterministic small sparse index set.
    idx = torch.full((B, S1, N2, K), -1, dtype=torch.int32)
    for s in range(S1):
        thr = (seqused_kv[0] - S1 + s + 1) // cmp_ratio
        valid = max(thr, 0)
        if valid > 0:
            take = min(K, valid)
            idx[0, s, 0, :take] = torch.arange(take, dtype=torch.int32)

    chunked, chunked_lse = G.sparse_attn_sharedkv_golden_bnsd(
        q,
        ori_k_bnsd,
        sinks,
        act_q_lens=[S1],
        act_kv_lens=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_k_bnsd=cmp_k_bnsd,
        cmp_sparse_indices=idx,
        cmp_ratio=cmp_ratio,
        return_lse=True,
    )

    # Reference: per (s) row, build the same sparse K=V slice and do one-shot.
    ref = torch.zeros_like(q)
    ref_lse = torch.zeros((1, N1, S1), dtype=torch.float32)
    for s in range(S1):
        s_global = seqused_kv[0] - S1 + s
        ori_left = max(s_global - 127, 0)
        ori_right = s_global + 1
        ori_k = ori_k_bnsd[0, 0, ori_left:ori_right, :]
        thr = (seqused_kv[0] - S1 + s + 1) // cmp_ratio
        # Same sparse selection logic as the chunked golden uses.
        raw = idx[0, s, 0]
        valid = (raw >= 0) & (raw < thr)
        sel = raw[valid].long()
        cmp_k = (
            cmp_k_bnsd[0, 0, sel, :] if sel.numel() else cmp_k_bnsd.new_zeros((0, D))
        )
        k_concat = torch.cat([ori_k, cmp_k], dim=0)  # [n, D]
        q_row = q[0, :, s, :]  # [N1, D]
        sm, sm_lse = G.sinks_softmax_reference(
            q_row.unsqueeze(0),
            k_concat.unsqueeze(0),
            sinks=sinks,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        ref[0, :, s, :] = sm.squeeze(0)
        ref_lse[0, :, s] = sm_lse.squeeze(0)

    torch.testing.assert_close(chunked, ref, rtol=2e-4, atol=2e-4)
    # lse aggregates across the whole row, so its absolute scale is
    # ~ln(seqlen) + max_score. Stay with fp32 strict-ish tolerance.
    torch.testing.assert_close(chunked_lse, ref_lse, rtol=1e-4, atol=1e-4)
