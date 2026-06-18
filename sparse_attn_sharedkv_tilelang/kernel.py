"""TileLang reimplementation of the Ascend C ``SparseAttnSharedkv`` operator.

This is a clean-room port that faithfully follows the Ascend C kernel
(`ops-transformer/.../sparse_attn_sharedkv/op_kernel`). The op fuses three
attention scenarios over a shared KV cache (N1=64 q-heads, N2=1 kv-head, D=512):

* scenario 1 -- **SWA**  : sliding-window attention over the uncompressed,
  page-attention ``ori_kv`` (win_left=127, win_right=0).
* scenario 2 -- **CFA**  : SWA + dense compressed attention over ``cmp_kv``.
* scenario 3 -- **SCFA** : SWA + sparse top-k compressed attention (the
  selected ``cmp_kv`` tokens are gathered via ``cmp_sparse_indices``).

Like the Ascend C kernel it runs in the MIX_AIC_1_2 layout: ``core_num`` AI
cores, each pairing one Cube unit (AIC) with two Vector lanes (AIV). Work is
distributed across cores by the companion metadata op (a ``[1024]`` int32
per-core ``(bn2, gS1, s2)`` range). The Cube unit runs the two matmuls
(Q@K^T, P@V) and the Vector lanes run scale / softmax / output-normalize,
overlapped through a cross-core software pipeline.

This file is built up incrementally (SWA first); see ``build_sparse_attn_sharedkv``.

SWA is written in the **structurally-faithful paged** style of the validated
``sparse_flash_attn_pa.py`` reference (``examples/sparse_flash_attention/
bench_sfa``). The per-query body keeps Cube + Vector TOGETHER inside an inner
``T.Pipelined`` over the KV tiles -- this is critical: a *flat* body inside a
plain ``T.serial`` was split by ``AUTO_CV_COMBINE`` into two SEQUENTIAL loops
(all-cube then all-vector) that overwrote the single-slot per-core GM
workspace, producing all-wrong output. Mirroring the reference's
five-phase V0/BMM1/V1/BMM2/VEC2 sequence inside ``T.Pipelined`` keeps each KV
tile's cube and vector work co-scheduled.

The synchronisation strategy also mirrors the reference EXACTLY:
``AUTO_CV_COMBINE`` + ``AUTO_CV_SYNC`` + ``MEMORY_PLANNING`` (``AUTO_SYNC`` is
*off*), plus MANUAL intra-core ``T.set_flag``/``T.wait_flag`` HardEvent pairs
(mte2/v/mte3/mte1/m/fix) inside the body. A prior full-auto attempt
(``AUTO_SYNC`` on, no manual flags) emitted 54 PIPE_ALL barriers and 0
cross-core handshakes -- the manual-flag pattern is what the auto-CV passes
were validated against.

The public ABI (kernel argument order, ``out_idx``/``workspace_idx``) is
unchanged so ``api.py`` is unchanged.

SWA-specific deltas from the paged reference (each one validated in earlier
HEAD revisions of this kernel):

* **Contiguous window, NOT sparse Indices** -- the reference reads each key id
  from ``Indices[s_i, g_i, ...]``; SWA's window keys are CONTIGUOUS:
  ``key = ori_left + i*n_base_size + r``. ``cmp_indices`` is an unused ABI
  placeholder. The per-row paged gather is kept (one ``T.copy`` per key row)
  precisely to avoid the data-dependent boundary ``if`` inside the pipelined
  body that the auto-CV pass cannot split cleanly. The causal mask reuses the
  IDENTICAL offset expression as the gather (``ori_left + i_i*n_base_size``)
  via ``createvecindex`` -- the two must never drift (see assert + comment).
* **No rope** -- our KV row is the full D=512; the reference's rope split
  (dim+rope_dim, second QK gemm, ``workspace_2``/``kv_rope_l1``) is dropped.
* **Sink seeding** -- instead of the reference's ``fill(score_max, 2^30)``
  sentinel, the flash state is seeded ONCE with the per-head SINK logit
  (running max ``m_i = sink``, running sum ``sumexp = 1`` so the implicit sink
  token contributes ``exp(sink-m_i)=1`` to the denominator). The sink is in
  UNSCALED logit space (scores get ``*softmax_scale``), matching the Ascend C
  ``swa_block_vector.h`` golden. The positive-max flash convention
  (Variant B of ``flash_attn_bhsd_auto_pipeline_h32_d512``) is used so the sink
  seed shares the score sign convention and LSE = ln(sum)+max is trivial.
* **LSE output** -- the reference emits no LSE; we write
  ``LSE_out[t,h] = ln(sumexp) + m_i`` (sink folded into both).

Per-query / per-core loop scalars are written as PLAIN assignments, exactly
like the reference (``b_i = by``); the reference uses NO ``T.alloc_var`` and
compiles + passes on this toolchain. The validated ``T.alloc_var`` idiom
(``test_tilelang_ascend_language_alloc_var``) only rebinds to COMPILE-TIME
constants -- rebinding an ``alloc_var`` to a runtime ``BufferLoad`` PrimExpr
orphans the buffer and re-inlines the GM read on every use, so it is NOT used
here.
"""

import tilelang
from tilelang import language as T

# ---------------------------------------------------------------------------
# Compile-time defaults / layout constants.
# ---------------------------------------------------------------------------
DEFAULT_CORE_NUM = 24  # AIC cores (each drives 2 AIV lanes) -- MIX_AIC_1_2.
DEFAULT_BLOCK_I = 128  # KV window width (s2 window fed across the inner tiles).

# Metadata ([1024] int32) FA-record layout, mirroring sparse_attn_sharedkv_metadata.h.
# faMetadata is [AIC_CORE_NUM][FA_METADATA_SIZE=8]; core ``cid`` reads row cid*8.
_SAS_META_SIZE = 1024
_FA_METADATA_SIZE = 8  # int32 stride per AIC core in the faMetadata block.
_FA_CORE_ENABLE_INDEX = 0
_FA_BN2_START_INDEX = 1
_FA_M_START_INDEX = 2
_FA_S2_START_INDEX = 3
_FA_BN2_END_INDEX = 4
_FA_M_END_INDEX = 5
_FA_S2_END_INDEX = 6

# Sync strategy -- EXACTLY the paged sparse_flash_attn_pa.py reference:
# AUTO_CV_COMBINE (split flat ops into Cube/Vector programs by buffer scope) +
# AUTO_CV_SYNC (layer cross-core Cube<->Vector handshakes on top of the manual
# pipe flags) + MEMORY_PLANNING. AUTO_SYNC is OFF -- intra-core pipe ordering is
# carried by the manual T.set_flag/T.wait_flag pairs inside the body. A prior
# full-auto (AUTO_SYNC on, no manual flags) produced 54 PIPE_ALL barriers and 0
# cross-core sync -- wrong.
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    # tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,  # intentionally OFF.
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Cache control: the kernel is parametrised heavily, keep JIT cache off to avoid
# stale binaries during development (matches the Ascend NPU pitfall guidance).
tilelang.disable_cache()
tilelang.cache.clear_cache()


def _check_dtypes(dtype: str) -> None:
    if dtype not in ("bfloat16", "float16"):
        raise ValueError(f"dtype must be bfloat16 or float16, got {dtype!r}")


def build_sparse_attn_sharedkv(
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
    n_heads: int = 64,
    n_kv_heads: int = 1,
    head_dim: int = 512,
    topk_cmp: int = 512,
    cmp_ratio: int = 4,
    scenario: int = 3,
    ori_win_left: int = 127,
    softmax_scale: float = 0.04419417,
    dtype: str = "bfloat16",
    block_I: int = DEFAULT_BLOCK_I,
    core_num: int = DEFAULT_CORE_NUM,
    return_prim_func: bool = False,
):
    """JIT-compile the SparseAttnSharedkv kernel for one parameter set.

    Returns a ``tilelang.jit``-wrapped ``prim_func`` whose ABI is the 11 inputs
    / 2 outputs / 5 workspaces documented in ``api.py``.

    When ``return_prim_func`` is True the *uncompiled* ``tir.PrimFunc`` is
    returned instead (no bisheng), so codegen-only dumps can be obtained via
    ``tilelang.lower`` even when the device compile would fail.
    """
    _check_dtypes(dtype)
    assert n_heads == 64, "API constraint: n_heads must be 64"
    assert n_kv_heads == 1, "API constraint: n_kv_heads must be 1"
    assert head_dim == 512, "API constraint: head_dim must be 512"
    assert ori_win_left == 127, "API constraint: ori_win_left must be 127"
    assert topk_cmp >= 0
    assert topk_cmp % block_I == 0, "topk_cmp must be a multiple of block_I"
    assert scenario in (1, 2, 3), "scenario must be 1 (SWA), 2 (CFA) or 3 (SCFA)"
    assert batch > 0 and max_seq > 0 and total_tokens > 0
    assert ori_block_num > 0 and ori_block_size > 0 and ori_table_len > 0
    assert cmp_block_num > 0 and cmp_block_size > 0 and cmp_table_len > 0
    # The SWA window spans <= ori_win_left+1 = 128 keys = exactly one BI window.
    # It is gathered PER ROW (one T.copy per key), so a key never straddles a
    # page boundary regardless of ori_block_size; the >= BI guard is kept only
    # as a sanity assertion that the window fits inside the addressed pages.
    assert ori_block_size >= block_I, "ori_block_size must be >= BI"

    gqa_group = n_heads // n_kv_heads  # 64 q-heads share 1 kv-head.
    BI = block_I  # 128 -- full sliding window width.
    D = head_dim  # 512
    accum_dtype = "float"
    indices_dtype = "int32"

    # ---- Inner KV tiling (mirror the reference's n_base_size sub-tiles). ----
    # The 128-key window is split into NI sub-tiles of n_base_size=64 keys each.
    # NI = ceil(BI / n_base_size) = 2. The inner T.Pipelined runs over these NI
    # sub-tiles, keeping cube + vector together per sub-tile. A 64-row KV tile
    # ([64,512] bf16 = 64KB) fits L0B and q_l1 [64,512] = 64KB fits L0A, so each
    # sub-tile's QK is ONE gemm into acc_s_l0c[64,64] -- no lo/hi half-split.
    n_base_size = 64
    NI = (BI + n_base_size - 1) // n_base_size  # 2
    # ori sliding window spans <= ori_win_left+1 = 128 keys = NI sub-tiles.
    # NI_cmp = number of compressed KV tiles (0 for SWA); drives indices_shape.
    NI_cmp = topk_cmp // BI

    # Head split: each of the 2 AIV lanes owns v_block=32 of the 64 q-heads.
    # m_base_size analog (cube head tile) = H_per_block = 64 (all q-heads in one
    # tile, g_block_num = 1); m_base_size_v analog (per-lane vector rows) =
    # v_block = 32. n_base_size_v analog (per-lane gather rows per sub-tile) = 32.
    H_per_block = gqa_group  # 64
    v_block = H_per_block // 2  # 32
    n_base_size_v = n_base_size // 2  # 32 -- each lane gathers its own 32 rows.
    # The per-row gather drains a UB stage to GM in vec0_copy_out_size-row
    # batches; with n_base_size_v == 32 each lane is exactly one batch.
    vec0_copy_out_size = 32
    # Pin the reference's gather-batch invariant: a single drain per lane per
    # sub-tile, so g_copy_out_time == i_i and the kv_ub_gather[2,...] ping-pong
    # is unambiguous (see the gather loop's task_id derivation).
    assert n_base_size_v == vec0_copy_out_size, (
        "gather drains in one batch per lane: keep n_base_size_v == "
        "vec0_copy_out_size so g_copy_out_time reduces to i_i"
    )
    mask_w = ((n_base_size // 8 + 31) // 32) * 32  # uint8 mask row, 32B aligned.

    # ---- Tensor shapes (the kernel ABI). ----
    q_shape = [total_tokens, n_heads, D]
    out_shape = [total_tokens, n_heads, D]
    ori_kv_shape = [ori_block_num, ori_block_size, n_kv_heads, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, n_kv_heads, D]
    ori_bt_shape = [batch, ori_table_len]
    cmp_bt_shape = [batch, cmp_table_len]
    indices_shape = [total_tokens, n_kv_heads, max(NI_cmp, 1) * BI]

    if scenario != 1:
        raise NotImplementedError(
            "Clean-room reimplementation in progress: only scenario 1 (SWA) is "
            f"available yet; scenario {scenario} (CFA/SCFA) is coming next."
        )

    @tilelang.jit(
        out_idx=[11, 12], workspace_idx=[13, 14, 15, 16, 17], pass_configs=pass_configs
    )
    def _make():
        # =================================================================
        # SWA (scenario 1): sliding-window attention only. Each query
        # position attends to <=128 uncompressed keys = NI=2 sub-tiles of 64
        # keys, processed by an inner T.Pipelined (cube + vector TOGETHER per
        # sub-tile) just like the paged sparse_flash_attn_pa.py reference.
        #
        # Each of the ``core_num`` cores walks ITS OWN metadata-assigned query
        # range [linear_start, linear_end) in a runtime-bounded T.serial loop
        # with the ``if s < act_q`` validity guard INSIDE the loop (the
        # reference's `for block_idx in T.serial(start,end): if s_i<act_q_len`
        # shape). Per query, before the inner pipeline the flash state is
        # sink-seeded ONCE; the inner pipeline runs the reference's five-phase
        # V0/BMM1/V1/BMM2/VEC2 body per sub-tile.
        #
        # Workspaces (5 GM slots, ONE per core, no ping-pong stage dim -- the
        # L1/UB/L0C tiles carry the double-buffering, matching the reference):
        #   ws_kv    [core_num, n_base_size, D]            gathered KV (cube reads -> kv_l1)
        #   ws_score [core_num, H_per_block, n_base_size]  QK scores (cube -> vector)
        #   ws_p     [core_num, H_per_block, n_base_size]  P (vector -> cube)
        #   ws_o     [core_num, H_per_block, D]            PV out (cube -> vector)
        #   ws_acc_o [core_num, n_base_size, D]            spare KV-stage slot (ABI parity)
        # =================================================================
        @T.prim_func
        def sparse_attn_sharedkv_swa(
            Q: T.Tensor(q_shape, dtype),
            ori_KV: T.Tensor(ori_kv_shape, dtype),
            ori_block_table: T.Tensor(ori_bt_shape, indices_dtype),
            cmp_KV: T.Tensor(cmp_kv_shape, dtype),
            cmp_block_table: T.Tensor(cmp_bt_shape, indices_dtype),
            cmp_indices: T.Tensor(indices_shape, indices_dtype),
            q_prefix: T.Tensor([batch], indices_dtype),
            actual_q_len: T.Tensor([batch], indices_dtype),
            actual_kv_len: T.Tensor([batch], indices_dtype),
            Sinks: T.Tensor([n_heads], accum_dtype),
            Metadata: T.Tensor([_SAS_META_SIZE], indices_dtype),
            Output: T.Tensor(out_shape, dtype),
            LSE_out: T.Tensor([total_tokens, n_heads], accum_dtype),
            # 5 workspaces, ONE slot per core (no ping-pong dim). ws_kv is the
            # gathered-KV GM slot the cube reads into kv_l1 (mirrors the
            # reference's workspace_1). ws_acc_o is a spare KV stage kept at the
            # same shape so a future 2nd-KV-stage tweak fits; it preserves the
            # workspace_idx ABI. The compiler double-buffers the L1/UB/L0C tiles
            # across pipeline stages, exactly like the paged reference.
            ws_kv: T.Tensor([core_num, n_base_size, D], dtype),
            ws_score: T.Tensor([core_num, H_per_block, n_base_size], accum_dtype),
            ws_p: T.Tensor([core_num, H_per_block, n_base_size], dtype),
            ws_o: T.Tensor([core_num, H_per_block, D], accum_dtype),
            ws_acc_o: T.Tensor([core_num, n_base_size, D], dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- L1 (cube-side) buffers. ----
                # Per 64-key sub-tile: q_l1 [64,512]=64KB fits L0A, kv_l1
                # [64,512]=64KB fits L0B. QK is one gemm into acc_s_l0c[64,64];
                # PV is one gemm into acc_o_l0c[64,512] (recomputed per sub-tile,
                # then flash-rescaled in VEC2, exactly like the reference).
                q_l1 = T.alloc_L1([H_per_block, D], dtype)
                kv_l1 = T.alloc_L1([n_base_size, D], dtype)
                acc_s_l1 = T.alloc_L1([H_per_block, n_base_size], dtype)

                acc_s_l0c = T.alloc_L0C([H_per_block, n_base_size], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                # ---- vec0 (gather) UB buffers. ----
                # mask_sel is a WHOLE buffer (not a parity slice) so
                # T.tile.select's selMask has .access_ptr (pitfall: select's
                # mask needs a whole Buffer). kv_ub_gather is the [2,...] task_id
                # ping-pong stage for the per-row gather.
                idx_int = T.alloc_ub([n_base_size], indices_dtype)
                idx_float = T.alloc_ub([n_base_size], accum_dtype)
                mask_sel = T.alloc_ub([mask_w], "uint8")
                kv_ub_gather = T.alloc_ub([2, vec0_copy_out_size, D], dtype)

                # ---- vec1 / vec2 (softmax + output) UB buffers. ----
                # Per-lane vector rows = v_block = 32. acc_s_ub holds this lane's
                # scaled scores -> P; acc_o is the running PV numerator.
                acc_s_ub = T.alloc_ub([v_block, n_base_size], accum_dtype)
                acc_s_ub_ = T.alloc_ub([v_block, n_base_size], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, n_base_size], dtype)
                m_i_2d = T.alloc_ub([v_block, n_base_size], accum_dtype)
                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                # Online-softmax running state (positive-max Variant B). These
                # are 2D COLUMN vectors [v_block, 1] -- the SAME convention as
                # BOTH validated templates (sparse_flash_attn_pa.py:
                # score_max=[m_base_size_v,1]; h32_d512: m_i/sumexp=[block_M/2,1]).
                # reduce_max/reduce_sum write into the [v_block,1] dst; broadcast
                # expands [v_block,1] -> [v_block,n_base_size]; the per-head
                # rescale indexes [h_i, 0]. A 1D [v_block] dst is NOT the shape
                # either reference validated and risks a degenerate reduce/brcb.
                m_i = T.alloc_ub([v_block, 1], accum_dtype)
                m_i_prev = T.alloc_ub([v_block, 1], accum_dtype)
                sumexp = T.alloc_ub([v_block, 1], accum_dtype)
                sumexp_i_ub = T.alloc_ub([v_block, 1], accum_dtype)
                lse_ub = T.alloc_ub([v_block, 1], accum_dtype)
                # Per-head sink logits, loaded ONCE before the loop ([v_block,1]).
                sinks_ub = T.alloc_ub([v_block, 1], accum_dtype)
                # Batched output divide: reciprocal of the running sum + its
                # per-row 8-lane broadcast (replaces a per-head div loop; this
                # batched form is the validated on-device perf path).
                recip = T.alloc_ub([v_block, 1], accum_dtype)
                recip_brd8 = T.alloc_ub([v_block, 8], accum_dtype)

                # ---- This core's metadata-assigned work range. ----
                # PLAIN scalar assignments, exactly like the reference (which
                # uses NO T.alloc_var and compiles + passes). meta_base = cid*8
                # (faMetadata stride). n_local is 0 when this core is disabled.
                meta_base = cid * _FA_METADATA_SIZE
                core_enable = Metadata[meta_base + _FA_CORE_ENABLE_INDEX]
                linear_start = (
                    Metadata[meta_base + _FA_BN2_START_INDEX] * max_seq
                    + Metadata[meta_base + _FA_M_START_INDEX]
                )
                linear_end = (
                    Metadata[meta_base + _FA_BN2_END_INDEX] * max_seq
                    + Metadata[meta_base + _FA_M_END_INDEX]
                )
                # Exact per-core query count (runtime). Looping [0, n_local) over
                # a runtime-bounded T.serial -- with the `if valid` guard INSIDE
                # -- is the validated paged sparse_flash_attn_pa.py shape (its
                # `for block_idx in T.serial(start,end): if s_i<act_q_len`).
                n_local = T.if_then_else(core_enable != 0, linear_end - linear_start, 0)

                # Load this lane's 32 sink logits once (constant per head).
                T.copy(Sinks[vid * v_block : vid * v_block + v_block], sinks_ub)

                # ---- Walk this core's queries (runtime-bounded serial loop). ----
                for j in T.serial(0, n_local):
                    # ---- Per-query scalars, PLAIN assignments (reference style).
                    # n_local==0 on disabled cores guarantees the loop body never
                    # runs there, so no extra core_enable/in_range guard is needed
                    # (pid is always in [linear_start, linear_end) by construction).
                    pid = linear_start + j
                    b = pid // max_seq
                    s = pid % max_seq
                    act_q = actual_q_len[b]
                    act_kv = actual_kv_len[b]
                    t = q_prefix[b] + s
                    s_global = act_kv - act_q + s
                    # Left window bound. The per-row gather starts AT ori_left,
                    # so the LEFT bound is enforced by construction; only the
                    # RIGHT (causal) bound needs masking below.
                    ori_left = T.if_then_else(
                        s_global < ori_win_left, 0, s_global - ori_win_left
                    )

                    # Single validity guard INSIDE the serial loop, mirroring the
                    # reference's `if s_i < act_q_len:`.
                    if s < act_q:
                        # Q into L1 once (all 64 q-heads; cube reads it for QK).
                        T.copy(Q[t, 0:H_per_block, 0:D], q_l1)

                        # ---- Flash-state init (ONCE, sink-seeded). ----
                        # Running max m_i = per-head sink logit (UNSCALED, since
                        # scores get *softmax_scale below); running sum = 1 so
                        # the implicit sink token contributes exp(sink-m_i)=1.
                        # Positive-max Variant B convention.
                        T.copy(sinks_ub, m_i)
                        T.tile.fill(sumexp, 1.0)
                        T.tile.fill(acc_o, 0.0)
                        T.pipe_barrier("v")

                        # ---- Inner pipeline over the NI=2 KV sub-tiles. ----
                        # Cube + Vector TOGETHER per sub-tile (V0/BMM1/V1/BMM2/
                        # VEC2), so AUTO_CV_COMBINE cannot split this into two
                        # sequential all-cube / all-vector loops.
                        for i_i in T.Pipelined(NI, num_stages=2):
                            # ******************** V0: per-row paged gather ********
                            # Window key for row r of sub-tile i_i:
                            #   key = ori_left + i_i*n_base_size + r   (CONTIGUOUS)
                            # The causal mask below and the gather key MUST use the
                            # IDENTICAL offset `ori_left + i_i*n_base_size` so the
                            # mask column r lines up with gathered row r. Do not
                            # let these two expressions drift.
                            T.tile.createvecindex(idx_int, ori_left + i_i * n_base_size)
                            T.copy(idx_int, idx_float)
                            T.pipe_barrier("v")
                            # Causal/right-window bound: keep keys with global
                            # index <= s_global (LE -> 1, else 0). NOTE float32
                            # compare loses integer precision above ~2^24; for the
                            # SWA window (<=128 keys, idx <= s_global) this is exact
                            # for realistic context lengths.
                            T.tile.compare(
                                mask_sel, idx_float, T.float32(s_global), "LE"
                            )

                            # Per-row gather: each lane gathers its OWN 32 rows
                            # (vid*n_base_size_v). One T.copy per key row -> no
                            # boundary `if` (the row never straddles a page). The
                            # block-table entry is resolved into a scalar FIRST,
                            # then a single-level gather (validated two-step PA).
                            #
                            # Gather-buffer reuse handshake (mirrors the reference,
                            # sparse_flash_attn_pa.py:169-211): task_id alternates
                            # per DRAIN BATCH via g_copy_out_time; the mte3->mte2
                            # round-trip keeps the NEXT batch from clobbering a
                            # kv_ub_gather stage still being drained by MTE3. With
                            # n_base_size_v==vec0_copy_out_size==32, g_copy_out_time
                            # reduces to i_i (one drain per lane per sub-tile), but
                            # the handshake is kept verbatim so it stays correct if
                            # the tile sizes ever change.
                            for bi_i in range(n_base_size_v):
                                inner_block_id = bi_i // vec0_copy_out_size  # 0
                                idx = bi_i % vec0_copy_out_size
                                g_copy_out_time = (
                                    i_i * (n_base_size_v // vec0_copy_out_size)
                                    + inner_block_id
                                )
                                task_id = g_copy_out_time % 2
                                if (
                                    g_copy_out_time > 1
                                    and bi_i % vec0_copy_out_size == 0
                                ):
                                    T.wait_flag("mte3", "mte2", task_id)

                                key = (
                                    ori_left
                                    + i_i * n_base_size
                                    + (bi_i + vid * n_base_size_v)
                                )
                                blk = key // ori_block_size
                                phys = ori_block_table[b, blk]
                                row = key % ori_block_size
                                T.copy(
                                    ori_KV[phys, row, 0, :],
                                    kv_ub_gather[task_id, idx, :],
                                )
                                # Drain this lane's 32-row batch to the GM KV slot.
                                if (bi_i + 1) % vec0_copy_out_size == 0:
                                    T.set_flag("mte2", "mte3", task_id)
                                    T.wait_flag("mte2", "mte3", task_id)
                                    T.copy(
                                        kv_ub_gather[task_id, :, :],
                                        ws_kv[
                                            cid,
                                            inner_block_id * vec0_copy_out_size
                                            + vid * n_base_size_v : (inner_block_id + 1)
                                            * vec0_copy_out_size
                                            + vid * n_base_size_v,
                                            :,
                                        ],
                                    )
                                    if (
                                        g_copy_out_time
                                        < NI * (n_base_size_v // vec0_copy_out_size) - 2
                                    ):
                                        T.set_flag("mte3", "mte2", task_id)

                            # ******************** BMM1: Q@K^T ********************
                            T.copy(ws_kv[cid, :, :], kv_l1)
                            T.set_flag("mte2", "mte1", 1)
                            T.wait_flag("mte2", "mte1", 1)
                            T.gemm_v0(
                                q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True
                            )
                            T.set_flag("m", "fix", 2)
                            T.wait_flag("m", "fix", 2)
                            T.copy(acc_s_l0c, ws_score[cid, :, :])

                            # ******************** V1: flash softmax **************
                            # This lane reads its 32 heads of the score sub-tile.
                            T.copy(
                                ws_score[
                                    cid, vid * v_block : vid * v_block + v_block, :
                                ],
                                acc_s_ub_,
                            )
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            # Causal/window mask: future keys (idx > s_global)
                            # -> -inf. Per-row select with the whole-buffer mask.
                            for h_i in T.serial(v_block):
                                T.tile.select(
                                    acc_s_ub[h_i, :],
                                    mask_sel,
                                    acc_s_ub_[h_i, :],
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
                                )
                            T.pipe_barrier("v")
                            # Positive-max flash (Variant B), the EXACT op
                            # sequence of flash_attn_bhsd_auto_pipeline_h32_d512:
                            # copy m_i->m_i_prev, scale, reduce_max into m_i,
                            # max(m_i,m_i_prev), sub+exp m_i_prev (the rescale
                            # factor alpha REUSES m_i_prev -- no separate buffer),
                            # broadcast m_i, sub, exp -> P, reduce_sum,
                            # mul sumexp by m_i_prev, add.
                            T.copy(m_i, m_i_prev)
                            T.pipe_barrier("v")
                            T.tile.mul(acc_s_ub, acc_s_ub, softmax_scale)
                            T.pipe_barrier("v")
                            T.reduce_max(acc_s_ub, m_i, dim=-1)
                            T.pipe_barrier("v")
                            # Running-max merge. dst==src0 is fine (max commutes);
                            # matches h32_d512 L108 `T.tile.max(m_i, m_i, m_i_prev)`.
                            T.tile.max(m_i, m_i, m_i_prev)
                            T.pipe_barrier("v")
                            T.tile.sub(m_i_prev, m_i_prev, m_i)
                            T.pipe_barrier("v")
                            # alpha = exp(max_prev - max), reusing m_i_prev.
                            T.tile.exp(m_i_prev, m_i_prev)
                            T.pipe_barrier("v")
                            T.tile.broadcast(m_i_2d, m_i)
                            T.pipe_barrier("v")
                            T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
                            T.pipe_barrier("v")
                            T.tile.exp(acc_s_ub, acc_s_ub)  # P = exp(scaled - max)
                            T.pipe_barrier("v")
                            T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)
                            T.pipe_barrier("v")
                            T.tile.mul(sumexp, sumexp, m_i_prev)  # rescale denom
                            T.pipe_barrier("v")
                            T.tile.add(sumexp, sumexp, sumexp_i_ub)
                            T.pipe_barrier("v")
                            T.copy(acc_s_ub, acc_s_half)
                            T.pipe_barrier("v")
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            T.copy(
                                acc_s_half,
                                ws_p[cid, vid * v_block : vid * v_block + v_block, :],
                            )

                            # ******************** BMM2: P@V *********************
                            # init=True every sub-tile is CORRECT: PV is recomputed
                            # per sub-tile into L0C, then flash-rescaled and
                            # accumulated in acc_o (UB) by VEC2 below. The flash
                            # accumulation lives in acc_o, NOT in L0C -- do NOT
                            # "fix" this to init=(i_i==0), that would double-count.
                            T.copy(ws_p[cid, :, :], acc_s_l1)
                            T.set_flag("mte2", "mte1", 3)
                            T.wait_flag("mte2", "mte1", 3)
                            T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                            T.set_flag("m", "fix", 4)
                            T.wait_flag("m", "fix", 4)
                            T.copy(acc_o_l0c, ws_o[cid, :, :])

                            # ******************** VEC2: flash output ************
                            # Rescale the running numerator by alpha (m_i_prev),
                            # then add this sub-tile's PV.
                            for h_i in range(v_block):
                                T.tile.mul(
                                    acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i, 0]
                                )
                            T.pipe_barrier("v")
                            T.set_flag("mte2", "v", 2)
                            T.wait_flag("mte2", "v", 2)
                            T.copy(
                                ws_o[cid, vid * v_block : vid * v_block + v_block, :],
                                acc_o_ub,
                            )
                            T.tile.add(acc_o, acc_o, acc_o_ub)

                        # ---- After inner loop: normalize + LSE. ----
                        # Batched normalize O[h,:] /= sumexp[h]: one reciprocal
                        # over all heads, then a per-row broadcast multiply
                        # (brcb + row_muls), instead of a per-head div loop.
                        T.tile.reciprocal(recip, sumexp)
                        T.pipe_barrier("v")
                        T.tile.brcb(
                            recip_brd8, recip[0:v_block, 0], (v_block + 7) // 8, 1, 8
                        )
                        T.pipe_barrier("v")
                        T.tile.row_muls(acc_o, acc_o, recip_brd8, v_block, D, D)
                        T.pipe_barrier("v")
                        T.copy(acc_o, acc_o_half)
                        T.set_flag("v", "mte3", 9)
                        T.wait_flag("v", "mte3", 9)
                        T.copy(
                            acc_o_half,
                            Output[t, vid * v_block : vid * v_block + v_block, :],
                        )
                        # LSE = ln(sum) + max  (sink already folded into both).
                        T.tile.ln(lse_ub, sumexp)
                        T.pipe_barrier("v")
                        T.tile.add(lse_ub, lse_ub, m_i)
                        T.pipe_barrier("v")
                        T.copy(
                            lse_ub[0:v_block, 0],
                            LSE_out[t, vid * v_block : vid * v_block + v_block],
                        )

        return sparse_attn_sharedkv_swa

    if return_prim_func:
        # ``@tilelang.jit`` keeps the original builder under ``__wrapped__``
        # (set by functools.wraps); call it to materialize the raw PrimFunc
        # without triggering the JIT compile / bisheng step.
        return _make.__wrapped__()
    return _make()
