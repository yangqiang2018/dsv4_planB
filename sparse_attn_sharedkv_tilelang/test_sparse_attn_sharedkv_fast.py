"""Fast small-shape prefill correctness tests for the TileLang SparseAttnSharedKV port.

The full suite (``test_sparse_attn_sharedkv.py``) runs the three prefill cases
at ``S1=8192``; the CPU golden reference takes minutes at that size, which makes
the kernel-development correctness loop slow. This file runs the SAME three
prefill scenarios (swa / cfa / scfa, TND, B=1) through the SAME kernel code
paths at a small ``S1`` -- only the shape is shrunk (``S1`` / ``seqused_kv`` /
paged block counts). ``BI``, ``NI_ori`` / ``NI_cmp``, cube-direct, the paged
block-boundary split, and the cmp masking are all identical to the 8K cases, so
this is a faithful (just smaller) correctness gate -- and the golden costs ~100x
less, so these cases run by DEFAULT (they are NOT marked ``slow``).

Coverage at ``S1=1024`` (chosen by a paged-block audit, see ``_FAST_S1``):

* ori: 8 paged blocks. Positions 0..127 have a clamped (partial) window, 128..
  1023 a full 128-token window whose ``ori_left`` is non-block-aligned, so the
  paged block-boundary split (the is_subtile-dependent cube-direct path) fires
  across many block crossings, not just the first.
* scfa cmp: 2 paged blocks (``cmp_act_kv = S1/cmp_ratio = 256``). The topk
  indices spread across both, so the discrete gather's block-table indirection
  (``idx // cmp_block_size`` selecting distinct physical blocks) IS exercised --
  this is the SCFA-specific path that an S1=256 shrink would have dropped
  (cmp_act_kv=64 -> 1 block -> no inter-block indirection).
* cfa cmp: 1 block -- but cfa's ``cmp_act_kv = S1/128`` is small at every size,
  so the 8K suite is ALSO single-block here; nothing is lost vs --runslow.
* metadata: ~43 query positions per core across all cores, so the multi-core
  load-balanced scheduling is exercised (an S1=256 shrink under-fills cores).

The only thing the 8K (--runslow) suite still adds over this is block-count
*diversity* (64 ori / 16 scfa-cmp blocks vs 8 / 2 here) -- the same code paths
with more distinct blocks, not new logic. Keep --runslow as the full-scale gate.

The 8K cases in ``test_sparse_attn_sharedkv.py`` are untouched -- run them with
``--runslow`` before a release or when the tiling/shapes change. This file is
the fast inner loop; that file is the full-scale gate.

    pytest -q test_sparse_attn_sharedkv_fast.py        # fast prefill correctness

All inputs, the CPU golden, the metadata+kernel runner, and the tolerance checks
are reused verbatim from ``test_sparse_attn_sharedkv`` -- only the shapes differ.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# Make local modules importable when pytest runs from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_sparse_attn_sharedkv import (  # noqa: E402
    SCENARIOS,
    _build_case,
    _call_metadata_then_sharedkv,
    _check_lse,
    _check_result,
    requires_npu,
)

# Small S1 for the fast golden. Chosen by a paged-block coverage audit, NOT just
# "smallest that runs": S1 must be large enough that every paged-attention path
# the 8K suite hits is still hit here.
#   - scfa needs cmp_act_kv = S1/cmp_ratio > block_size2 (=128) so the topk
#     indices span >= 2 cmp blocks and the discrete gather's block-table
#     indirection is exercised. cmp_ratio=4 => S1 > 512; S1=1024 gives 2 blocks.
#   - ori boundary split + multi-core scheduling want several blocks / a full
#     core fan-out; S1=1024 gives 8 ori blocks and ~43 positions/core.
# Golden cost stays small (~0.4-0.9s/case warm), so the loop is still ~100x
# faster than 8K. Raising S1 only adds block-count diversity (covered by the
# --runslow 8K gate); lowering below 1024 silently drops the scfa multi-block
# gather -- do not.
_FAST_S1 = 1024


def _shrink(name: str, s1: int = _FAST_S1) -> dict:
    """Derive a small-S1 prefill config from an 8K one.

    Only the shape is overridden -- ``S1`` / ``T1`` / ``cu_seqlens_q`` /
    ``seqused_kv`` and the paged block counts. ``K``, ``cmp_ratio``, the block
    sizes, the masks, and ``ori_win_left`` are copied verbatim, so the kernel
    config (``NI_ori`` / ``NI_cmp`` / cube-direct / boundary-split) is the SAME
    code path as the 8K case -- just with fewer query tokens.
    """
    cfg = dict(SCENARIOS[name])
    cfg.update(S1=s1, T1=s1, cu_seqlens_q=[0, s1], seqused_kv=[s1])
    # ori paged blocks: ceil(act_kv / block_size1), plus slack.
    cfg["block_num1"] = math.ceil(s1 / cfg["block_size1"]) + 2
    # cmp paged blocks (scenarios 2/3): ceil(floor(act_kv/cmp_ratio) /
    # block_size2), plus slack. The block-table width stays floor-derived, same
    # as the 8K case -- cube-direct cfa over-reads past it for the dense cmp
    # range, but those tokens are masked out by cmp_threshold (identical to 8K).
    if cfg["scenario"] >= 2:
        cmp_seq = s1 // cfg["cmp_ratio"]
        cfg["block_num2"] = max(1, math.ceil(cmp_seq / cfg["block_size2"])) + 1
    return cfg


FAST_SCENARIOS = {
    "swa_prefill_fast": _shrink("swa_prefill"),
    "cfa_prefill_fast": _shrink("cfa_prefill"),
    "scfa_prefill_fast": _shrink("scfa_prefill"),
}


@requires_npu
@pytest.mark.parametrize("case_name", list(FAST_SCENARIOS.keys()))
# bf16 only (dtype0) for the fast loop -- it is the Ascend C reference dtype.
# fp16 is extra coverage and stays in the full 8K (--runslow) gate.
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_sparse_attn_sharedkv_fast(case_name, dtype):
    """Small-shape prefill correctness -- same code paths as the 8K suite, fast golden."""
    # Imported lazily so collection works on hosts without tilelang installed.
    from api import sparse_attn_sharedkv
    from metadata import sparse_attn_sharedkv_metadata

    cfg = FAST_SCENARIOS[case_name]
    # Data generation + golden run on CPU (default device).
    case = _build_case(cfg, dtype)
    out, lse, _ = _call_metadata_then_sharedkv(
        case, cfg, sparse_attn_sharedkv, sparse_attn_sharedkv_metadata
    )
    _check_result(out.cpu(), case["cpu_ref"])
    _check_lse(lse.cpu(), case["cpu_ref_lse"], dtype)
