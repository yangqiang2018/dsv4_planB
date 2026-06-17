"""Python golden reference for SparseAttnSharedKV.

Mirrors the math of the Ascend C kernel
(``ops-transformer/.../sparse_attn_sharedkv``) under the
``RUN_MODE=1`` flash-attention path: an online-softmax loop that walks
the sliding-window ``ori_kv`` tokens first, then the top-K sparse
``cmp_kv`` tokens, with per-head sinks folded into the initial
``(row_max, row_sum)`` state.

This file is dependency-light: ``torch`` only, no NPU. Use it to
generate ground truth for unit tests.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch

# S2 base size used by the original Ascend C kernel. Online softmax is
# mathematically tile-size invariant; using the original size keeps the
# fp16/bf16 cast roundings aligned for a tighter atol.
GOLDEN_S2_TILE = 512


def _gather_cmp_kv_tokens(
    cmp_k_bnsd: torch.Tensor,  # [B, N2, S3, D]
    sparse_indices: torch.Tensor,  # [B, S1, N2, K] int32 (BSND order)
    b: int,
    n2: int,
    s_local: int,
    act_kv: int,
    act_q: int,
    cmp_ratio: int,
    cmp_mask_mode: int = 3,
    sparse_block_size: int = 1,
) -> torch.Tensor:
    """Resolve per-q-token cmp sparse indices into a [K_real, D] tensor."""
    K = sparse_indices.shape[-1]
    cmp_act_kv = math.floor(act_kv / cmp_ratio)
    if cmp_mask_mode == 3:
        threshold = math.floor((act_kv - act_q + s_local + 1) / cmp_ratio)
    else:
        raise ValueError(f"cmp_mask_mode {cmp_mask_mode} not supported")

    valid_count = min(K, math.ceil(threshold / sparse_block_size))
    s2 = []
    for i in range(valid_count):
        topk_id = sparse_indices[b, s_local, n2, i].item()
        if topk_id == -1:
            break
        begin = topk_id * sparse_block_size
        end = begin + sparse_block_size
        if end > cmp_act_kv:
            end = cmp_act_kv
        if begin >= threshold:
            continue
        if end > threshold:
            end = threshold
        s2.extend(range(begin, end))

    if not s2:
        return cmp_k_bnsd.new_zeros((0, cmp_k_bnsd.shape[-1]))
    return cmp_k_bnsd[b, n2, s2, :]


def sparse_attn_sharedkv_golden_bnsd(
    q_bnsd: torch.Tensor,  # [B, N1, S1, D] (any of bf16/fp16/fp32)
    ori_k_bnsd: torch.Tensor,  # [B, N2, S2_ori_max, D]
    sinks: torch.Tensor,  # [N1] fp32
    *,
    act_q_lens: Sequence[int],
    act_kv_lens: Sequence[int],
    softmax_scale: float,
    cmp_k_bnsd: Optional[torch.Tensor] = None,  # [B, N2, S3_cmp_max, D]
    cmp_sparse_indices: Optional[torch.Tensor] = None,  # [B, S1, N2, K] int32 BSND
    cmp_ratio: Optional[int] = None,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    s2_tile: int = GOLDEN_S2_TILE,
    return_lse: bool = False,
):
    """Compute ``attention_out`` in BNSD layout, matching the Ascend kernel.

    If ``return_lse=True`` also returns the LogSumExp ``[B, N1, S1]``
    fp32 tensor, defined as ``lse[b, h, s] = row_max + ln(row_sum)``
    over all attended keys (sliding window + sparse cmp), with the
    per-head sink contribution folded into the initial state -- the
    same definition the Ascend C kernel uses when
    ``return_softmax_lse=True`` (see
    ``sparse_attn_sharedkv/op_kernel/arch32/
    sparse_attn_sharedkv_swa_block_vector.h`` ``ProcessLse``). Padded
    ``(b, s)`` slots (``s >= act_q_lens[b]``) stay at zero.
    """
    assert ori_win_right == 0, "only ori_win_right=0 (causal) is supported"
    assert ori_mask_mode == 4, "only ori_mask_mode=4 supported"

    B, N1, S1, D = q_bnsd.shape
    N2 = ori_k_bnsd.shape[1]
    G = N1 // N2
    dtype = q_bnsd.dtype
    out = torch.zeros_like(q_bnsd, dtype=dtype)
    lse = (
        torch.zeros((B, N1, S1), dtype=torch.float32, device=q_bnsd.device)
        if return_lse
        else None
    )

    has_cmp = cmp_k_bnsd is not None and cmp_sparse_indices is not None
    is_dense_cmp = cmp_k_bnsd is not None and cmp_sparse_indices is None

    for b in range(B):
        act_q = int(act_q_lens[b])
        act_kv = int(act_kv_lens[b])
        for n2 in range(N2):
            head_lo = n2 * G
            head_hi = (n2 + 1) * G
            sink_group = sinks[head_lo:head_hi].to(torch.float32)
            for s in range(act_q):
                # Causal s-position in kv sequence.
                s_global = act_kv - act_q + s
                q = q_bnsd[b, head_lo:head_hi, s, :].to(torch.float32)

                # Sliding window over ori_kv.
                ori_right = s_global + ori_win_right + 1  # exclusive
                ori_left = max(s_global - ori_win_left, 0)
                ori_k = ori_k_bnsd[b, n2, ori_left:ori_right, :].to(torch.float32)

                # cmp pass tokens.
                if has_cmp:
                    cmp_tokens = _gather_cmp_kv_tokens(
                        cmp_k_bnsd,
                        cmp_sparse_indices,
                        b=b,
                        n2=n2,
                        s_local=s,
                        act_kv=act_kv,
                        act_q=act_q,
                        cmp_ratio=cmp_ratio,
                        cmp_mask_mode=cmp_mask_mode,
                    ).to(torch.float32)
                elif is_dense_cmp:
                    threshold = math.floor((act_kv - act_q + s + 1) / cmp_ratio)
                    cmp_tokens = cmp_k_bnsd[b, n2, : max(threshold, 0), :].to(
                        torch.float32
                    )
                else:
                    cmp_tokens = torch.empty(
                        (0, D), dtype=torch.float32, device=q_bnsd.device
                    )

                # Online softmax state seeded from sinks. row_max starts
                # at the sink logit and row_sum at 1 = exp(sink - sink),
                # i.e. the sink is treated as a virtual KV token with a
                # zero V row. This makes the final
                # lse = row_max + ln(row_sum) naturally include the
                # sink contribution.
                row_max = sink_group.clone()  # [G]
                row_sum = torch.ones(G, dtype=torch.float32, device=q_bnsd.device)
                acc_o = torch.zeros((G, D), dtype=torch.float32, device=q_bnsd.device)

                # Walk: ori first (in tile order), then cmp.
                ori_tiles = (
                    max(1, math.ceil(ori_k.size(0) / s2_tile))
                    if ori_k.size(0) > 0
                    else 0
                )
                cmp_tiles = (
                    math.ceil(cmp_tokens.size(0) / s2_tile)
                    if cmp_tokens.size(0) > 0
                    else 0
                )

                for t in range(ori_tiles + cmp_tiles):
                    if t < ori_tiles:
                        k_tile = ori_k[t * s2_tile : (t + 1) * s2_tile, :]
                    else:
                        ct = t - ori_tiles
                        k_tile = cmp_tokens[ct * s2_tile : (ct + 1) * s2_tile, :]
                    if k_tile.size(0) == 0:
                        continue

                    score = torch.matmul(q, k_tile.T) * softmax_scale  # [G, n]
                    row_max_old = row_max.clone()
                    row_max = torch.max(row_max, score.amax(dim=1))
                    alpha = torch.exp(row_max_old - row_max)
                    p = torch.exp(score - row_max.unsqueeze(1))
                    row_sum = alpha * row_sum + p.sum(dim=1)
                    # P is materialized in dtype on the NPU before mm2.
                    p_low = p.to(dtype).to(torch.float32)
                    pv = torch.matmul(p_low, k_tile)
                    acc_o = acc_o * alpha.unsqueeze(1) + pv

                out[b, head_lo:head_hi, s, :] = (acc_o / row_sum.unsqueeze(1)).to(dtype)
                if return_lse:
                    lse[b, head_lo:head_hi, s] = row_max + torch.log(row_sum)

    if return_lse:
        return out, lse
    return out


def sinks_softmax_reference(
    q_bnsd: torch.Tensor,
    k_concat_bnsd: torch.Tensor,
    *,
    sinks: torch.Tensor,
    softmax_scale: float,
    return_lse: bool = False,
):
    """One-shot softmax reference (no chunking) for math sanity checks.

    Computes ``softmax(Q @ K^T * scale  ∪  sinks) @ V`` where ``sinks``
    is treated as a virtual extra logit per head with V row zero.

    ``k_concat_bnsd`` is the full concatenated K (=V) for this work item:
    sliding window slice + sparse cmp slice.

    If ``return_lse=True`` also returns the LogSumExp tensor (the
    leading dims match ``q_bnsd`` minus the last dim, so for input
    ``[..., G, D]`` the result is ``[..., G]``). The lse covers the
    same logit set as the softmax (scores ∪ sinks).
    """
    q = q_bnsd.to(torch.float32)
    k = k_concat_bnsd.to(torch.float32)
    score = torch.matmul(q, k.transpose(-1, -2)) * softmax_scale  # [..., G, n]
    sinks_b = sinks.to(torch.float32).view(-1, 1)
    x_concat = torch.cat([score, sinks_b.expand(score.shape[:-1] + (1,))], dim=-1)
    x_max = x_concat.amax(dim=-1, keepdim=True)
    y = torch.exp(score - x_max)
    denom = y.sum(dim=-1, keepdim=True) + torch.exp(sinks_b - x_max)
    p = y / denom
    out = torch.matmul(p.to(q_bnsd.dtype).to(torch.float32), k).to(q_bnsd.dtype)
    if return_lse:
        # lse = ln(sum exp(x_i)) = x_max + ln(denom*exp(x_max-x_max))
        # but denom here already excludes the x_max shift, so
        # lse = x_max + ln(sum y + exp(sinks - x_max)) = x_max + ln(denom).
        lse = (x_max + torch.log(denom)).squeeze(-1)
        return out, lse
    return out


def unpack_paged_kv(
    kv_pa: torch.Tensor,  # [block_num, block_size, N2, D]
    block_table: torch.Tensor,  # [B, table_len] int32, -1 ⇒ unused
    max_logical_len: int,
) -> torch.Tensor:
    """Unpack paged KV back to BNSD ``[B, N2, S, D]`` for golden math."""
    block_num, block_size, N2, D = kv_pa.shape
    B = block_table.shape[0]
    out = kv_pa.new_zeros((B, N2, max_logical_len, D))
    for b in range(B):
        for blk_i in range(block_table.shape[1]):
            phys = int(block_table[b, blk_i].item())
            if phys < 0:
                continue
            start = blk_i * block_size
            end = start + block_size
            if start >= max_logical_len:
                break
            end = min(end, max_logical_len)
            take = end - start
            out[b, :, start:end, :] = kv_pa[phys, :take, :, :].permute(1, 0, 2)
    return out


def tnd_to_bnsd_q(q_tnd: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    """Pad a TND Q tensor ``[T, N, D]`` into BNSD ``[B, N, S_max, D]``."""
    T, N, D = q_tnd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    B = len(seq_lens)
    S_max = max(seq_lens) if seq_lens else 0
    out = q_tnd.new_zeros((B, N, S_max, D))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        out[b, :, :L, :] = q_tnd[start:end, :, :].permute(1, 0, 2)
    return out


def bnsd_to_tnd_out(out_bnsd: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    """Unpad a BNSD output ``[B, N, S_max, D]`` back to TND ``[T, N, D]``."""
    B, N, _, D = out_bnsd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    T_total = sum(seq_lens)
    out = out_bnsd.new_zeros((T_total, N, D))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        out[start:end, :, :] = out_bnsd[b, :, :L, :].permute(1, 0, 2)
    return out


def bnsd_to_tnd_lse(lse_bnsd: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    """Unpad a BNSD lse ``[B, N, S_max]`` back to TND ``[T, N]``."""
    B, N, _ = lse_bnsd.shape
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    T_total = sum(seq_lens)
    out = lse_bnsd.new_zeros((T_total, N))
    for b in range(B):
        start = int(cu_seqlens_q[b].item())
        end = int(cu_seqlens_q[b + 1].item())
        L = end - start
        # lse_bnsd[b, :, :L]: [N, L] -> [L, N]
        out[start:end, :] = lse_bnsd[b, :, :L].permute(1, 0)
    return out


def gen_ori_kv_paged(
    *,
    B: int,
    N2: int,
    D: int,
    block_num: int,
    block_size: int,
    seqused_kv: Sequence[int],
    dtype: torch.dtype,
    data_range: tuple = (-10, 10),
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Generate paged ori_kv + matching ori_block_table + bnsd version."""
    rng = rng or np.random.default_rng(0)
    max_s = max(seqused_kv)
    max_blk = math.ceil(max_s / block_size)
    lo, hi = data_range

    # Dense BNSD reference, only first act_kv tokens per batch are nonzero.
    ori_k_bnsd = (torch.rand((B, N2, max_s, D)) * (hi - lo) + lo).to(dtype)

    # Build block table.
    per_batch_blocks = [math.ceil(s / block_size) for s in seqused_kv]
    total_blocks = sum(per_batch_blocks)
    if block_num < total_blocks:
        raise ValueError(f"ori block_num too small: {block_num} < {total_blocks}")

    phys_ids = rng.permutation(block_num).astype(np.int32)
    table = np.full((B, max_blk), -1, dtype=np.int32)
    cursor = 0
    for b, nb in enumerate(per_batch_blocks):
        for i in range(nb):
            table[b, i] = phys_ids[cursor]
            cursor += 1

    # Place into paged layout.
    pa = torch.zeros((block_num, block_size, N2, D), dtype=dtype)
    for b in range(B):
        for i in range(per_batch_blocks[b]):
            phys = int(table[b, i])
            s0 = i * block_size
            s1 = min(s0 + block_size, max_s)
            take = s1 - s0
            pa[phys, :take, :, :] = ori_k_bnsd[b, :, s0:s1, :].permute(1, 0, 2)

    return pa, torch.tensor(table, dtype=torch.int32), ori_k_bnsd


def gen_cmp_kv_paged(
    *,
    B: int,
    N2: int,
    D: int,
    block_num: int,
    block_size: int,
    seqused_kv: Sequence[int],
    cmp_ratio: int,
    dtype: torch.dtype,
    data_range: tuple = (-5, 10),
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Generate paged cmp_kv + matching cmp_block_table + bnsd version."""
    rng = rng or np.random.default_rng(1)
    cmp_seqs = [math.floor(s / cmp_ratio) for s in seqused_kv]
    max_cmp = max(cmp_seqs) if cmp_seqs else 0
    max_blk = math.ceil(max_cmp / block_size) if max_cmp > 0 else 1
    lo, hi = data_range

    cmp_k_bnsd = (torch.rand((B, N2, max(max_cmp, 1), D)) * (hi - lo) + lo).to(dtype)

    per_batch_blocks = [math.ceil(s / block_size) if s > 0 else 0 for s in cmp_seqs]
    total_blocks = sum(per_batch_blocks)
    if block_num < max(total_blocks, 1):
        raise ValueError(f"cmp block_num too small: {block_num} < {total_blocks}")

    phys_ids = rng.permutation(block_num).astype(np.int32)
    table = np.full((B, max(max_blk, 1)), -1, dtype=np.int32)
    cursor = 0
    for b, nb in enumerate(per_batch_blocks):
        for i in range(nb):
            table[b, i] = phys_ids[cursor]
            cursor += 1

    pa = torch.zeros((block_num, block_size, N2, D), dtype=dtype)
    for b in range(B):
        for i in range(per_batch_blocks[b]):
            phys = int(table[b, i])
            s0 = i * block_size
            s1 = min(s0 + block_size, cmp_seqs[b])
            take = s1 - s0
            if take > 0:
                pa[phys, :take, :, :] = cmp_k_bnsd[b, :, s0:s1, :].permute(1, 0, 2)

    return pa, torch.tensor(table, dtype=torch.int32), cmp_k_bnsd, cmp_seqs


def gen_cmp_sparse_indices_bsnd(
    *,
    B: int,
    S1: int,
    N2: int,
    K: int,
    seqused_kv: Sequence[int],
    cmp_ratio: int,
    cmp_mask_mode: int = 3,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pad-with-(-1) random sparse indices per (b, s, n2) tuple."""
    if cmp_mask_mode != 3:
        raise ValueError("only cmp_mask_mode=3 supported")
    cmp_sparse_indices = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    for b in range(B):
        act_kv = int(seqused_kv[b])
        for n2 in range(N2):
            for s in range(S1):
                cur_max = math.floor((act_kv - S1 + s + 1) / cmp_ratio)
                cur_max = max(0, cur_max)
                perm = (
                    torch.randperm(cur_max, generator=rng).to(torch.int32)
                    if cur_max > 0
                    else torch.empty((0,), dtype=torch.int32)
                )
                take = min(cur_max, K)
                cmp_sparse_indices[b, s, n2, :take] = perm[:take]
    return cmp_sparse_indices


def gen_cmp_sparse_indices_tnd(
    *,
    B: int,
    T1: int,
    N2: int,
    K: int,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: Sequence[int],
    cmp_ratio: int,
    cmp_mask_mode: int = 3,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if cmp_mask_mode != 3:
        raise ValueError("only cmp_mask_mode=3 supported")
    cmp_sparse_indices = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    for b in range(B):
        s_start = int(cu_seqlens_q[b].item())
        s_end = int(cu_seqlens_q[b + 1].item())
        act_q = s_end - s_start
        act_kv = int(seqused_kv[b])
        for n2 in range(N2):
            for s in range(act_q):
                cur_max = math.floor((act_kv - act_q + s + 1) / cmp_ratio)
                cur_max = max(0, cur_max)
                perm = (
                    torch.randperm(cur_max, generator=rng).to(torch.int32)
                    if cur_max > 0
                    else torch.empty((0,), dtype=torch.int32)
                )
                take = min(cur_max, K)
                cmp_sparse_indices[s_start + s, n2, :take] = perm[:take]
    return cmp_sparse_indices
