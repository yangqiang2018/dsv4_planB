"""High-level Python entry point for the TileLang SparseAttnSharedKV op.

Handles the three scenarios from the original Ascend C operator and the
TND/BSND layout dispatch. The kernel works on flat ``[total_tokens,
...]`` tensors plus a per-batch ``q_prefix`` offset, so TND inputs pass
through natively and BSND inputs are a free reshape -- no host-side
layout conversion. This module only synthesises dummy inputs for the
SWA-only and CFA scenarios so the same kernel can serve all three.

Usage::

    from api import sparse_attn_sharedkv

    out = sparse_attn_sharedkv(
        q,                       # [T1, N1, D] (TND) or [B, S1, N1, D]
        ori_kv=ori_kv,           # paged: [block_num, block_size, N2, D]
        cmp_kv=cmp_kv,           # paged or None
        cmp_sparse_indices=...,  # int32, or None
        ori_block_table=...,
        cmp_block_table=...,     # or None
        cu_seqlens_q=...,        # required for TND
        seqused_kv=...,          # actual ori_kv length per batch
        sinks=...,               # [N1] fp32
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )

Pass ``return_softmax_lse=True`` to also get the per-token LogSumExp
tensor (fp32), shaped ``[T1, N1]`` for TND or ``[B, S1, N1]`` for BSND.
``lse[t, h] = max_h + ln(sum_h)`` with the sink contribution folded
in -- the standard FlashAttention-v2 reverse-pass input.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from kernel import build_sparse_attn_sharedkv, DEFAULT_BLOCK_I, DEFAULT_CORE_NUM
from metadata import sparse_attn_sharedkv_metadata, SAS_META_SIZE

# Module-level kernel cache: key is the tuple of compile-time params.
_KERNEL_CACHE: dict = {}


def _torch_to_tilelang_dtype(t: torch.dtype) -> str:
    if t == torch.bfloat16:
        return "bfloat16"
    if t == torch.float16:
        return "float16"
    raise ValueError(f"unsupported torch dtype {t}")


def _get_kernel(
    *,
    batch: int,
    max_seq: int,
    total_tokens: int,
    ori_block_num: int,
    ori_block_size: int,
    ori_table_len: int,
    cmp_block_num: int,
    cmp_block_size: int,
    cmp_table_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    topk_cmp: int,
    cmp_ratio: int,
    scenario: int,
    ori_win_left: int,
    softmax_scale: float,
    dtype: str,
    core_num: int,
):
    key = (
        batch,
        max_seq,
        total_tokens,
        ori_block_num,
        ori_block_size,
        ori_table_len,
        cmp_block_num,
        cmp_block_size,
        cmp_table_len,
        n_heads,
        n_kv_heads,
        head_dim,
        topk_cmp,
        cmp_ratio,
        scenario,
        ori_win_left,
        round(softmax_scale, 8),
        dtype,
        core_num,
    )
    func = _KERNEL_CACHE.get(key)
    if func is None:
        func = build_sparse_attn_sharedkv(
            batch=batch,
            max_seq=max_seq,
            total_tokens=total_tokens,
            ori_block_num=ori_block_num,
            ori_block_size=ori_block_size,
            ori_table_len=ori_table_len,
            cmp_block_num=cmp_block_num,
            cmp_block_size=cmp_block_size,
            cmp_table_len=cmp_table_len,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            topk_cmp=topk_cmp,
            cmp_ratio=cmp_ratio,
            scenario=scenario,
            ori_win_left=ori_win_left,
            softmax_scale=softmax_scale,
            dtype=dtype,
            core_num=core_num,
        )
        _KERNEL_CACHE[key] = func
    return func


def sparse_attn_sharedkv(
    q: torch.Tensor,
    *,
    ori_kv: torch.Tensor,
    cmp_kv: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: torch.Tensor,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
    core_num: int = DEFAULT_CORE_NUM,
    topk_cmp: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Forward pass. Returns ``attention_out`` with the same layout as ``q``.

    The function detects the scenario from the optional arguments:

    * ``cmp_kv=None, cmp_sparse_indices=None`` → SWA (scenario 1).
    * ``cmp_kv=<tensor>, cmp_sparse_indices=None`` → CFA (scenario 2).
    * ``cmp_kv=<tensor>, cmp_sparse_indices=<tensor>`` → SCFA (scenario 3).

    ``metadata`` matches the contract of the Ascend C
    ``SparseAttnSharedkv`` op: it is the output of the companion
    ``SparseAttnSharedkvMetadata`` op (see :mod:`metadata`) and carries
    the per-core FA / FD task ranges (``faMetadata[cid] = (enable,
    bn2_start, m_start, s2_start, bn2_end, m_end, s2_end, ...)``).
    The TileLang kernel consumes this tensor on-device: each AIC core
    reads its own row to know which ``(bn2, m)`` work range it owns,
    matching the Ascend C ``SparseAttnSharedkv`` cube/vector scheduling
    contract one-to-one. If ``metadata`` is omitted we build it here
    via the Python port (:func:`metadata.sparse_attn_sharedkv_metadata`)
    using the inputs already at hand.

    ``seqused_q`` is accepted for API parity but unused by the kernel.

    If ``return_softmax_lse=True`` the function returns a pair
    ``(attn_out, lse)`` where ``lse`` is fp32 with shape ``[T1, N1]``
    (TND) or ``[B, S1, N1]`` (BSND). ``lse[t, h] = max_h + ln(sum_h)``
    over all attended keys, with the per-head sink contribution folded
    in -- this is the LogSumExp value FlashAttention-v2's backward
    pass needs to rebuild the softmax probabilities. The kernel always
    computes lse internally (it is essentially free; see
    :mod:`kernel`), so the switch only controls whether the value is
    surfaced to the caller.
    """
    del seqused_q  # API-parity placeholder; kernel does not consume it
    assert ori_mask_mode == 4, "only ori_mask_mode=4 supported"
    assert cmp_mask_mode == 3 or cmp_kv is None, "only cmp_mask_mode=3 supported"
    assert ori_win_right == 0, "only ori_win_right=0 supported"
    assert layout_kv == "PA_ND", "only layout_kv=PA_ND supported"
    assert layout_q in ("TND", "BSND"), f"unsupported layout_q={layout_q!r}"

    dtype = q.dtype
    tl_dtype = _torch_to_tilelang_dtype(dtype)

    # Resolve scenario.
    if cmp_kv is None:
        scenario = 1  # SWA
    elif cmp_sparse_indices is None:
        scenario = 2  # CFA
    else:
        scenario = 3  # SCFA

    # ---- Normalize Q / indices into a flat [total_tokens, ...] view. ----
    # The kernel addresses Q / Output / cmp_indices by a flat token id
    # `q_prefix[b] + s`. TND inputs are passed through natively; BSND
    # inputs are reshaped (a free view of the same contiguous memory).
    if layout_q == "TND":
        assert cu_seqlens_q is not None, "cu_seqlens_q is required for TND"
        cu = cu_seqlens_q.to(torch.int32).cpu()
        B = cu.numel() - 1
        seq_lens = (cu[1:] - cu[:-1]).tolist()
        S_max = max(seq_lens) if seq_lens else 0
        T1, N1, D = q.shape
        total_tokens = T1
        q_flat = q
        q_prefix = cu[:-1].to(q.device)
        act_q_lens = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
        if scenario == 3:
            cmp_indices_flat = cmp_sparse_indices.to(torch.int32).to(q.device)
        else:
            cmp_indices_flat = None
    else:  # BSND
        B, S_max, N1, D = q.shape
        total_tokens = B * S_max
        q_flat = q.reshape(total_tokens, N1, D)
        q_prefix = torch.arange(B, dtype=torch.int32, device=q.device) * S_max
        seq_lens = [S_max] * B
        act_q_lens = torch.full((B,), S_max, dtype=torch.int32, device=q.device)
        if cu_seqlens_q is not None:
            cu = cu_seqlens_q.to(torch.int32).cpu()
            seq_lens = (cu[1:] - cu[:-1]).tolist()
            act_q_lens = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
        if scenario == 3:
            ci = cmp_sparse_indices.to(torch.int32)
            cmp_indices_flat = ci.reshape(total_tokens, ci.shape[2], ci.shape[3]).to(
                q.device
            )
        else:
            cmp_indices_flat = None

    seqused_kv_dev = seqused_kv.to(torch.int32).to(q.device)

    # ---- Resolve topk / scenario-specific tensors. ----
    if scenario == 3:
        N2 = cmp_indices_flat.shape[1]
        K = topk_cmp if topk_cmp is not None else cmp_indices_flat.shape[2]
        assert K == cmp_indices_flat.shape[2], (
            "topk_cmp does not match cmp_sparse_indices last dim"
        )
        cmp_indices_dev = cmp_indices_flat
    elif scenario == 2:
        N2 = cmp_kv.shape[2]
        # CFA attends to ALL compressed tokens up to the per-row causal
        # threshold (NOT a fixed top-K). K must span the largest possible
        # threshold, floor(max(seqused_kv) / cmp_ratio); topk_cmp is
        # meaningless for CFA and is intentionally ignored. The kernel
        # generates the dense [0, K) indices on-device, so cmp_indices
        # is an unused placeholder here.
        max_cmp = int(seqused_kv.max().item()) // cmp_ratio
        # Round up to a multiple of the kernel's KV tile (BI). CFA's dense
        # [0, K) indices over-cover; the extra tail (>= cmp threshold) is
        # masked out, so over-rounding only adds masked work, not error.
        _bi = DEFAULT_BLOCK_I
        K = max(_bi, ((max_cmp + _bi - 1) // _bi) * _bi)
        cmp_indices_dev = torch.zeros(
            (total_tokens, N2, K), dtype=torch.int32, device=q.device
        )
    else:
        N2 = ori_kv.shape[2]
        K = 0
        # SWA has no cmp pass; the kernel never reads cmp_indices. Pass a
        # minimal one-chunk dummy so the kernel argument stays well-typed.
        # Last dim must match the kernel's indices_shape (max(NI_cmp,1)*BI).
        cmp_indices_dev = torch.zeros(
            (total_tokens, N2, DEFAULT_BLOCK_I), dtype=torch.int32, device=q.device
        )

    if N1 != 64 or N2 != 1 or D != 512:
        raise ValueError(
            f"only N1=64, N2=1, D=512 supported (got N1={N1}, N2={N2}, D={D})"
        )

    # ---- Paged KV goes straight to the kernel. ----
    # The kernel resolves the block table on the AI Core (vector),
    # mirroring the Ascend C `DataCopyPA` path -- no host-side un-paging.
    cmp_ratio_eff = cmp_ratio if cmp_ratio is not None else 4

    ori_kv_dev = ori_kv.to(q.device)
    ori_bt_dev = ori_block_table.to(torch.int32).to(q.device)
    ori_block_num, ori_block_size = ori_kv_dev.shape[0], ori_kv_dev.shape[1]
    ori_table_len = ori_bt_dev.shape[1]

    if cmp_kv is not None:
        cmp_kv_dev = cmp_kv.to(q.device)
        cmp_bt_dev = cmp_block_table.to(torch.int32).to(q.device)
    else:
        # SWA: no cmp pass (NI_cmp == 0). A 1-block dummy paged cache
        # and block table keep the kernel signature well-typed.
        cmp_kv_dev = torch.zeros((1, 1, N2, D), dtype=dtype, device=q.device)
        cmp_bt_dev = torch.zeros((B, 1), dtype=torch.int32, device=q.device)
    cmp_block_num, cmp_block_size = cmp_kv_dev.shape[0], cmp_kv_dev.shape[1]
    cmp_table_len = cmp_bt_dev.shape[1]

    # ---- JIT-compile kernel for these compile-time params. ----
    func = _get_kernel(
        batch=int(B),
        max_seq=int(S_max),
        total_tokens=int(total_tokens),
        ori_block_num=int(ori_block_num),
        ori_block_size=int(ori_block_size),
        ori_table_len=int(ori_table_len),
        cmp_block_num=int(cmp_block_num),
        cmp_block_size=int(cmp_block_size),
        cmp_table_len=int(cmp_table_len),
        n_heads=N1,
        n_kv_heads=N2,
        head_dim=D,
        topk_cmp=K,
        cmp_ratio=cmp_ratio_eff,
        scenario=scenario,
        ori_win_left=ori_win_left,
        softmax_scale=float(softmax_scale),
        dtype=tl_dtype,
        core_num=core_num,
    )

    # ---- Sinks on device, fp32. ----
    sinks_dev = sinks.to(torch.float32).to(q.device)

    # ---- Metadata: produced by the companion scheduler op, drives
    # per-AIC-core work ranges inside the kernel. Synthesise it here
    # via the Python port if the caller did not pre-compute it (the
    # test flow does, mirroring the Ascend C reference). ----
    if metadata is None:
        max_seqlen_kv = int(seqused_kv.max().item())
        meta_kwargs = dict(
            num_heads_q=N1,
            num_heads_kv=N2,
            head_dim=D,
            cu_seqlens_q=cu_seqlens_q if layout_q == "TND" else None,
            seqused_kv=seqused_kv,
            batch_size=int(B),
            max_seqlen_q=int(S_max),
            max_seqlen_kv=max_seqlen_kv,
            ori_mask_mode=ori_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=True,
            has_cmp_kv=cmp_kv is not None,
            aic_core_num=core_num,
        )
        if scenario >= 2:
            meta_kwargs["cmp_ratio"] = cmp_ratio_eff
            meta_kwargs["cmp_mask_mode"] = cmp_mask_mode
        if scenario == 3:
            meta_kwargs["cmp_topk"] = K
        metadata = sparse_attn_sharedkv_metadata(**meta_kwargs)
    metadata_dev = metadata.to(torch.int32).to(q.device)
    assert metadata_dev.numel() == SAS_META_SIZE, (
        f"metadata must have {SAS_META_SIZE} int32 entries (got {metadata_dev.numel()})"
    )
    # The kernel expects a flat [SAS_META_SIZE] view.
    metadata_dev = metadata_dev.reshape(SAS_META_SIZE)

    # ---- Run kernel. Workspaces are auto-allocated via workspace_idx. ----
    # The kernel signature has two outputs (Output, LSE_out); jit's
    # out_idx=[11, 12] makes it return a tuple. We always receive both
    # and only forward lse when the caller asked for it.
    out_flat, lse_flat = func(
        q_flat.contiguous(),
        ori_kv_dev.contiguous(),
        ori_bt_dev.contiguous(),
        cmp_kv_dev.contiguous(),
        cmp_bt_dev.contiguous(),
        cmp_indices_dev.contiguous(),
        q_prefix.contiguous(),
        act_q_lens.contiguous(),
        seqused_kv_dev.contiguous(),
        sinks_dev.contiguous(),
        metadata_dev.contiguous(),
    )

    # ---- Restore the caller's layout. ----
    # out_flat: [total_tokens, N1, D]; lse_flat: [total_tokens, N1].
    if layout_q == "TND":
        out = out_flat  # already [T1, N1, D]
        lse = lse_flat  # [T1, N1]
    else:
        out = out_flat.reshape(B, S_max, N1, D)
        lse = lse_flat.reshape(B, S_max, N1)

    if return_softmax_lse:
        return out, lse
    return out
