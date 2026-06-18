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
(Q@K^T, P@V) and the Vector lanes run scale / online-softmax / output-merge,
overlapped through a cross-core software pipeline.

This file is built up incrementally (SWA first); see ``build_sparse_attn_sharedkv``.
The public ABI (kernel argument order, ``out_idx``/``workspace_idx``) matches the
original so ``api.py`` is unchanged.
"""

import tilelang
from tilelang import language as T

# ---------------------------------------------------------------------------
# Compile-time defaults / layout constants.
# ---------------------------------------------------------------------------
DEFAULT_CORE_NUM = 24  # AIC cores (each drives 2 AIV lanes) -- MIX_AIC_1_2.
DEFAULT_BLOCK_I = 128  # KV tile width (s2 base block fed to one QK/PV step).

# Metadata ([1024] int32) FA-record layout, mirroring sparse_attn_sharedkv_metadata.h.
_SAS_META_SIZE = 1024
_FA_METADATA_SIZE = 8
_FA_CORE_ENABLE_INDEX = 0
_FA_BN2_START_INDEX = 1
_FA_M_START_INDEX = 2
_FA_S2_START_INDEX = 3
_FA_BN2_END_INDEX = 4
_FA_M_END_INDEX = 5
_FA_S2_END_INDEX = 6

# Cross-core (Cube <-> Vector) handshake flags, mirroring the Ascend C
# syncC1V1 / syncV1C2 / syncC2V2 events.
_FLAG_SCORE_READY = 1  # Cube finished Q@K^T -> Vector may read scores.
_FLAG_P_READY = 2  # Vector finished softmax -> Cube may read P.
_FLAG_PV_READY = 3  # Cube finished P@V -> Vector may read O.

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
):
    """JIT-compile the SparseAttnSharedkv kernel for one parameter set.

    Returns a ``tilelang.jit``-wrapped ``prim_func`` whose ABI is the 11 inputs
    / 2 outputs / 5 workspaces documented in ``api.py``.
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
    # The SWA gather loads each 64-key half as <=2 page segments; this requires
    # the page block to hold at least one half (always true: block_size1=128).
    assert ori_block_size >= block_I // 2, "ori_block_size must be >= BI/2"

    gqa_group = n_heads // n_kv_heads  # 64 q-heads share 1 kv-head.
    BI = block_I  # 128
    BI_half = BI // 2  # 64 -- KV is loaded/gemm'd in two 64-row halves.
    D = head_dim  # 512
    accum_dtype = "float"
    indices_dtype = "int32"

    # ori sliding window spans <= ori_win_left+1 = 128 keys, i.e. exactly one BI
    # tile, so SWA is single-tile attention per query (no cross-tile flash loop).
    # NI_cmp = number of compressed KV tiles (0 for SWA); drives indices_shape.
    NI_cmp = topk_cmp // BI

    # Head split: each of the 2 AIV lanes owns v_block=32 of the 64 q-heads.
    H_per_block = gqa_group  # 64
    v_block = H_per_block // 2  # 32
    ub_len = max(32 // 4, v_block)  # >= 8 fp32 elements for per-head scalars.
    mask_w = ((BI // 8 + 31) // 32) * 32  # uint8 mask row, 32B aligned.
    MERGE_HEADS = 16  # the 32 per-lane heads are merged in 2 passes of 16.
    assert v_block % MERGE_HEADS == 0
    N_MERGE_PASS = v_block // MERGE_HEADS
    assert N_MERGE_PASS == 2

    # ---- Tensor shapes (the kernel ABI). ----
    q_shape = [total_tokens, n_heads, D]
    out_shape = [total_tokens, n_heads, D]
    ori_kv_shape = [ori_block_num, ori_block_size, n_kv_heads, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, n_kv_heads, D]
    ori_bt_shape = [batch, ori_table_len]
    cmp_bt_shape = [batch, cmp_table_len]
    indices_shape = [total_tokens, n_kv_heads, max(NI_cmp, 1) * BI]

    KB = 1024
    # Manual L0C / UB byte-offset pinning (no auto allocator on Ascend); these
    # mirror the workspace layout the original used and are correctness-critical.
    l0c_addr = {"acc_o_l0c": 0}
    ub_addr = {
        "acc_o": 0,
        "kv_ub_multi": 64 * KB,
        "acc_o_ub": 96 * KB,
        "acc_s_half": 96 * KB,
        "acc_s_ub_": 128 * KB,
        "acc_s_ub": 160 * KB,
        "m_i": 176 * KB,
        "m_i_prev": 176 * KB + 128,
        "sumexp": 176 * KB + 256,
        "sumexp_i_ub": 176 * KB + 384,
        "sinks_ub": 176 * KB + 512,
        "lse_ub": 176 * KB + 640,
        "idx_int": 176 * KB + 768,
        "idx_float": 176 * KB + 1280,
        "mask_ub": 176 * KB + 1792,
        "alpha": 176 * KB + 2048,
        "mask_sel": 176 * KB + 2304,
        "acc_o_half": 64 * KB,
    }

    if scenario != 1:
        raise NotImplementedError(
            "Clean-room reimplementation in progress: only scenario 1 (SWA) is "
            f"available yet; scenario {scenario} (CFA/SCFA) is coming next."
        )

    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15, 16, 17])
    def _make():
        # =================================================================
        # SWA (scenario 1): sliding-window attention only. Each query
        # position attends to <=128 uncompressed keys = one KV tile, so the
        # whole flash machinery collapses to a single sink-seeded softmax.
        #
        # Cube/Vector cooperate via a 3-deep cross-query software pipeline:
        #   pid0 : Cube gathers KV + Q@K^T -> ws_score ; Vector builds mask
        #   pid1 : Vector scale+softmax -> ws_p        ; Cube P@V -> ws_o
        #   pid2 : Vector normalizes O -> Output, LSE
        # which mirrors the Ascend C PreloadPipeline (MM1 / Vec1 / MM2 / Vec2).
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
            ws_kv: T.Tensor([core_num, 2, BI, D], dtype),
            ws_score: T.Tensor([core_num, 2, H_per_block, BI], accum_dtype),
            ws_p: T.Tensor([core_num, 2, H_per_block, BI], dtype),
            ws_o: T.Tensor([core_num, 2, H_per_block, D], accum_dtype),
            ws_acc_o: T.Tensor([core_num, 2, H_per_block, D], accum_dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- L1 (cube-side) buffers, double-buffered over pid parity. ----
                q_l1 = T.alloc_L1([2, H_per_block, D], dtype)
                kv_lo = T.alloc_L1([2, BI_half, D], dtype)
                kv_hi = T.alloc_L1([2, BI_half, D], dtype)
                p_lo = T.alloc_L1([H_per_block, BI_half], dtype)
                p_hi = T.alloc_L1([H_per_block, BI_half], dtype)
                # Two score accumulators so the two QK halves overlap with their
                # L0C->GM fixpipe copies; one output accumulator for P@V.
                acc_s_a = T.alloc_L0C([H_per_block, BI_half], accum_dtype)
                acc_s_b = T.alloc_L0C([H_per_block, BI_half], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                # ---- UB (vector-side) buffers. ----
                acc_o_work = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_work2 = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                m_i = T.alloc_ub([ub_len], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
                lse_ub = T.alloc_ub([ub_len], accum_dtype)
                # Per-head sink logits, loaded from GM ONCE before the loop
                # (constant per head; Ascend C's CopySinksIn is also one-shot).
                sinks_ub = T.alloc_ub([ub_len], accum_dtype)
                alpha = T.alloc_ub([2 * ub_len], accum_dtype)
                alpha_exp = T.alloc_ub([ub_len], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_ub_ = T.alloc_ub([2 * v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)
                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                softmax_tmp = T.alloc_ub([16 * KB], "uint8")
                mask_ub = T.alloc_ub([2, mask_w], "uint8")
                mask_sel = T.alloc_ub([mask_w], "uint8")
                # Per-pid saved softmax stats, reloaded at output time (pid2).
                sumexp_sv = T.alloc_ub([2, ub_len], accum_dtype)
                m_i_sv = T.alloc_ub([2, ub_len], accum_dtype)
                sumexp_rt = T.alloc_ub([ub_len], accum_dtype)
                m_i_rt = T.alloc_ub([ub_len], accum_dtype)
                # Reciprocal of the running sum + its per-row broadcast, for the
                # batched output divide (replaces the per-head div loop).
                recip = T.alloc_ub([ub_len], accum_dtype)
                recip_brd8 = T.alloc_ub([MERGE_HEADS, 8], accum_dtype)

                T.annotate_address(
                    {
                        q_l1: 0,
                        kv_lo: 128 * KB,
                        kv_hi: 256 * KB,
                        p_lo: 384 * KB,
                        p_hi: 392 * KB,
                        acc_s_a: 0,
                        acc_s_b: 64 * KB,
                        acc_o_l0c: l0c_addr["acc_o_l0c"],
                        acc_o_work: ub_addr["acc_o"],
                        acc_o_work2: ub_addr["acc_o"] + 32 * KB,
                        acc_s_ub: ub_addr["acc_s_ub"],
                        acc_s_ub_: ub_addr["acc_s_ub_"],
                        acc_s_half: ub_addr["acc_s_half"],
                        m_i: ub_addr["m_i"],
                        m_i_prev: ub_addr["m_i_prev"],
                        sumexp: ub_addr["sumexp"],
                        sumexp_i_ub: ub_addr["sumexp_i_ub"],
                        lse_ub: ub_addr["lse_ub"],
                        sinks_ub: ub_addr["sinks_ub"],
                        idx_int: ub_addr["idx_int"],
                        idx_float: ub_addr["idx_float"],
                        alpha: ub_addr["alpha"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_sel: ub_addr["mask_sel"],
                        acc_o_half: ub_addr["acc_o_half"],
                        softmax_tmp: ub_addr["kv_ub_multi"],
                        alpha_exp: ub_addr["kv_ub_multi"] + 16 * KB + 512,
                        sumexp_sv: ub_addr["mask_sel"] + 32,
                        m_i_sv: ub_addr["mask_sel"] + 32 + 256,
                        sumexp_rt: ub_addr["mask_sel"] + 32 + 512,
                        m_i_rt: ub_addr["mask_sel"] + 32 + 640,
                        recip: ub_addr["mask_sel"] + 32 + 768,
                        recip_brd8: ub_addr["mask_sel"] + 32 + 896,
                    }
                )

                # ---- This core's metadata-assigned work range. ----
                # Materialize the loop-invariant scalars ONCE (alloc_var): as plain
                # PrimExprs they are re-inlined (re-read from GM) on every use, which
                # the msprof scalar-pipe profile flagged as the dominant cost.
                meta_base = cid * _FA_METADATA_SIZE
                core_enable = T.alloc_var(indices_dtype, init=0)
                core_enable = Metadata[meta_base + _FA_CORE_ENABLE_INDEX]
                linear_start = T.alloc_var(indices_dtype, init=0)
                linear_start = (
                    Metadata[meta_base + _FA_BN2_START_INDEX] * max_seq
                    + Metadata[meta_base + _FA_M_START_INDEX]
                )
                linear_end = T.alloc_var(indices_dtype, init=0)
                linear_end = (
                    Metadata[meta_base + _FA_BN2_END_INDEX] * max_seq
                    + Metadata[meta_base + _FA_M_END_INDEX]
                )
                total_work = batch * max_seq

                # ============================ CUBE ============================
                with T.Scope("C"):
                    # Per-iteration scalars materialized once (alloc_var): otherwise
                    # each is re-inlined on every use (the gather offsets ~20x/query),
                    # which msprof showed is the cube's scalar-pipe bottleneck.
                    valid0 = T.alloc_var("bool", init=False)
                    valid1 = T.alloc_var("bool", init=False)
                    valid2 = T.alloc_var("bool", init=False)
                    b0 = T.alloc_var(indices_dtype, init=0)
                    act_q0 = T.alloc_var(indices_dtype, init=0)
                    act_kv0 = T.alloc_var(indices_dtype, init=0)
                    s_global0 = T.alloc_var(indices_dtype, init=0)
                    ori_left0 = T.alloc_var(indices_dtype, init=0)
                    t0 = T.alloc_var(indices_dtype, init=0)
                    bidx_lo = T.alloc_var(indices_dtype, init=0)
                    rowc_lo = T.alloc_var(indices_dtype, init=0)
                    n_lo = T.alloc_var(indices_dtype, init=0)
                    bidx_hi = T.alloc_var(indices_dtype, init=0)
                    rowc_hi = T.alloc_var(indices_dtype, init=0)
                    n_hi = T.alloc_var(indices_dtype, init=0)
                    T.set_flag("fix", "m", 0)
                    T.set_flag("fix", "m", 1)
                    for g in T.serial(total_work + 2):
                        # pid0 = current task (QK), pid1 = task-1 (PV).
                        pid0 = linear_start + g
                        in_range0 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid0 < linear_end, pid0 >= linear_start, False
                            ),
                            False,
                        )
                        b0 = T.if_then_else(in_range0, pid0 // max_seq, 0)
                        s0 = pid0 % max_seq
                        act_q0 = actual_q_len[b0]
                        act_kv0 = actual_kv_len[b0]
                        valid0 = T.if_then_else(in_range0, s0 < act_q0, False)
                        t0 = q_prefix[b0] + s0
                        s_global0 = act_kv0 - act_q0 + s0
                        ori_left0 = T.if_then_else(
                            s_global0 - ori_win_left < 0, 0, s_global0 - ori_win_left
                        )

                        pid1 = linear_start + g - 1
                        in_range1 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid1 < linear_end, pid1 >= linear_start, False
                            ),
                            False,
                        )
                        valid1 = T.if_then_else(
                            in_range1,
                            (pid1 % max_seq)
                            < actual_q_len[
                                T.if_then_else(in_range1, pid1 // max_seq, 0)
                            ],
                            False,
                        )

                        pid2 = linear_start + g - 2
                        in_range2 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid2 < linear_end, pid2 >= linear_start, False
                            ),
                            False,
                        )
                        valid2 = T.if_then_else(
                            in_range2,
                            (pid2 % max_seq)
                            < actual_q_len[
                                T.if_then_else(in_range2, pid2 // max_seq, 0)
                            ],
                            False,
                        )

                        if valid2:
                            # Release KV double-buffer slot consumed two iters ago.
                            T.wait_flag("m", "mte2", 0)

                        if valid0:
                            pa = g % 2
                            T.copy(Q[t0, 0:n_heads, 0:D], q_l1[pa, :, :])
                            # --- Paged sliding-window KV gather into kv_lo/kv_hi. ---
                            # Load BI=128 keys starting at ori_left0; ori_block_size
                            # (128) >= BI_half so each 64-row half spans at most two
                            # page segments (mirrors Ascend C DataCopyPA fast path).
                            bidx_lo = ori_left0 // ori_block_size
                            rowc_lo = ori_left0 % ori_block_size
                            n_lo = ori_block_size - rowc_lo
                            if n_lo >= BI_half:
                                T.copy(
                                    ori_KV[
                                        ori_block_table[b0, bidx_lo],
                                        rowc_lo : rowc_lo + BI_half,
                                        0,
                                        :,
                                    ],
                                    kv_lo[pa, 0:BI_half, :],
                                )
                            else:
                                T.copy(
                                    ori_KV[
                                        ori_block_table[b0, bidx_lo],
                                        rowc_lo : rowc_lo + n_lo,
                                        0,
                                        :,
                                    ],
                                    kv_lo[pa, 0:n_lo, :],
                                )
                                T.copy(
                                    ori_KV[
                                        ori_block_table[b0, bidx_lo + 1],
                                        0 : BI_half - n_lo,
                                        0,
                                        :,
                                    ],
                                    kv_lo[pa, n_lo:BI_half, :],
                                )
                            g0_hi = ori_left0 + BI_half
                            bidx_hi = g0_hi // ori_block_size
                            rowc_hi = g0_hi % ori_block_size
                            n_hi = ori_block_size - rowc_hi
                            if n_hi >= BI_half:
                                T.copy(
                                    ori_KV[
                                        ori_block_table[b0, bidx_hi],
                                        rowc_hi : rowc_hi + BI_half,
                                        0,
                                        :,
                                    ],
                                    kv_hi[pa, 0:BI_half, :],
                                )
                            else:
                                T.copy(
                                    ori_KV[
                                        ori_block_table[b0, bidx_hi],
                                        rowc_hi : rowc_hi + n_hi,
                                        0,
                                        :,
                                    ],
                                    kv_hi[pa, 0:n_hi, :],
                                )
                                T.copy(
                                    ori_KV[
                                        ori_block_table[b0, bidx_hi + 1],
                                        0 : BI_half - n_hi,
                                        0,
                                        :,
                                    ],
                                    kv_hi[pa, n_hi:BI_half, :],
                                )
                            T.set_flag("mte2", "m", 3)
                            T.wait_flag("mte2", "m", 3)
                            # --- Q@K^T over the two 64-key halves. ---
                            T.wait_flag("fix", "m", 0)
                            T.gemm_v0(
                                q_l1[pa, :, :],
                                kv_lo[pa, :, :],
                                acc_s_a,
                                transpose_B=True,
                                init=True,
                            )
                            T.set_flag("m", "fix", 0)
                            T.wait_flag("fix", "m", 1)
                            T.gemm_v0(
                                q_l1[pa, :, :],
                                kv_hi[pa, :, :],
                                acc_s_b,
                                transpose_B=True,
                                init=True,
                            )
                            T.set_flag("m", "fix", 2)
                            # Drain L0C scores to the GM workspace (two halves).
                            T.wait_flag("m", "fix", 0)
                            T.copy(acc_s_a, ws_score[cid, pa, 0:H_per_block, 0:BI_half])
                            T.set_flag("fix", "m", 0)
                            T.wait_flag("m", "fix", 2)
                            T.copy(
                                acc_s_b, ws_score[cid, pa, 0:H_per_block, BI_half:BI]
                            )
                            T.set_flag("fix", "m", 1)
                            T.set_cross_flag("FIX", _FLAG_SCORE_READY)

                        if valid1:
                            pb = (g - 1) % 2
                            # --- P@V: read P (softmax output) from the vector side. ---
                            T.wait_cross_flag(_FLAG_P_READY)
                            T.copy(ws_p[cid, pb, 0:H_per_block, 0:BI_half], p_lo)
                            T.copy(ws_p[cid, pb, 0:H_per_block, BI_half:BI], p_hi)
                            T.set_flag("mte2", "m", 0)
                            T.wait_flag("mte2", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.wait_flag("fix", "m", 1)
                            T.gemm_v0(p_lo, kv_lo[pb, :, :], acc_o_l0c, init=True)
                            T.gemm_v0(p_hi, kv_hi[pb, :, :], acc_o_l0c, init=False)
                            T.set_flag("m", "mte2", 0)
                            T.set_flag("m", "fix", 1)
                            T.wait_flag("m", "fix", 1)
                            T.copy(acc_o_l0c, ws_o[cid, pb, 0:H_per_block, 0:D])
                            T.set_flag("fix", "m", 0)
                            T.set_flag("fix", "m", 1)
                            T.set_cross_flag("FIX", _FLAG_PV_READY)
                    T.wait_flag("fix", "m", 0)
                    T.wait_flag("fix", "m", 1)

                # =========================== VECTOR ===========================
                with T.Scope("V"):
                    # Load this lane's 32 sink logits once (constant per head).
                    T.copy(Sinks[vid * v_block : vid * v_block + v_block], sinks_ub)
                    T.set_flag("mte2", "v", 4)
                    T.wait_flag("mte2", "v", 4)
                    T.set_flag("v", "mte2", 0)
                    T.set_flag("v", "mte2", 1)
                    # Per-iteration scalars materialized once (alloc_var).
                    valid0 = T.alloc_var("bool", init=False)
                    valid1 = T.alloc_var("bool", init=False)
                    valid2 = T.alloc_var("bool", init=False)
                    b0 = T.alloc_var(indices_dtype, init=0)
                    act_q0 = T.alloc_var(indices_dtype, init=0)
                    act_kv0 = T.alloc_var(indices_dtype, init=0)
                    s_global0 = T.alloc_var(indices_dtype, init=0)
                    ori_left0 = T.alloc_var(indices_dtype, init=0)
                    s_global1 = T.alloc_var(indices_dtype, init=0)
                    t2 = T.alloc_var(indices_dtype, init=0)
                    for g in T.serial(total_work + 2):
                        pid0 = linear_start + g
                        in_range0 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid0 < linear_end, pid0 >= linear_start, False
                            ),
                            False,
                        )
                        b0 = T.if_then_else(in_range0, pid0 // max_seq, 0)
                        s0 = pid0 % max_seq
                        act_q0 = actual_q_len[b0]
                        act_kv0 = actual_kv_len[b0]
                        valid0 = T.if_then_else(in_range0, s0 < act_q0, False)
                        s_global0 = act_kv0 - act_q0 + s0
                        ori_right0 = s_global0
                        ori_left0 = T.if_then_else(
                            s_global0 - ori_win_left < 0, 0, s_global0 - ori_win_left
                        )

                        pid1 = linear_start + g - 1
                        in_range1 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid1 < linear_end, pid1 >= linear_start, False
                            ),
                            False,
                        )
                        b1 = T.if_then_else(in_range1, pid1 // max_seq, 0)
                        s1 = pid1 % max_seq
                        valid1 = T.if_then_else(in_range1, s1 < actual_q_len[b1], False)
                        s_global1 = actual_kv_len[b1] - actual_q_len[b1] + s1

                        pid2 = linear_start + g - 2
                        in_range2 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid2 < linear_end, pid2 >= linear_start, False
                            ),
                            False,
                        )
                        b2 = T.if_then_else(in_range2, pid2 // max_seq, 0)
                        s2 = pid2 % max_seq
                        valid2 = T.if_then_else(in_range2, s2 < actual_q_len[b2], False)
                        t2 = q_prefix[b2] + s2

                        if valid0:
                            # Build the causal/window mask ONLY for partial windows
                            # (s_global < win_left): there the loaded 128 keys span
                            # [0, 127] but only [0, s_global] are valid, so future
                            # keys must be masked. A full window loads exactly its
                            # valid keys, so masking is skipped (the common case).
                            if s_global0 < ori_win_left:
                                T.tile.createvecindex(idx_int, ori_left0)
                                T.copy(idx_int, idx_float)
                                T.pipe_barrier("v")
                                T.tile.compare(
                                    mask_ub[g % 2, :],
                                    idx_float,
                                    T.float32(ori_right0),
                                    "LE",
                                )
                                T.pipe_barrier("v")
                            T.wait_cross_flag(_FLAG_SCORE_READY)
                            T.wait_flag("v", "mte2", g % 2)
                            T.copy(
                                ws_score[
                                    cid,
                                    g % 2,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                                acc_s_ub_[
                                    (g % 2) * v_block : (g % 2) * v_block + v_block, :
                                ],
                            )
                            T.set_flag("mte2", "v", g % 2)

                        if valid1:
                            pv1 = (g - 1) % 2
                            # Sink-seeded online softmax (single tile): seed running
                            # max with per-head sink logit, sum with exp(0)=1.
                            T.copy(sinks_ub, m_i)
                            T.tile.fill(sumexp, 1.0)
                            T.pipe_barrier("v")
                            T.wait_flag("mte2", "v", pv1)
                            if s_global1 < ori_win_left:
                                # Partial window: mask future keys to -inf per head.
                                T.copy(mask_ub[pv1, :], mask_sel)
                                for h_i in T.serial(v_block):
                                    T.tile.select(
                                        acc_s_ub[h_i, :],
                                        mask_sel,
                                        acc_s_ub_[pv1 * v_block + h_i, :],
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                            else:
                                # Full window: every loaded key is valid -> one
                                # batched move (no per-head select).
                                T.copy(
                                    acc_s_ub_[
                                        pv1 * v_block : pv1 * v_block + v_block, :
                                    ],
                                    acc_s_ub,
                                )
                            T.set_flag("v", "mte2", pv1)
                            T.copy(m_i, m_i_prev)
                            T.tile.mul(acc_s_ub, acc_s_ub, softmax_scale)
                            T.copy(sumexp, sumexp_i_ub)
                            T.tile.softmax_flashv2(
                                acc_s_ub,
                                sumexp,
                                m_i,
                                alpha_exp,
                                sumexp_i_ub,
                                m_i_prev,
                                softmax_tmp,
                                v_block,
                                BI,
                                BI,
                            )
                            T.copy(
                                alpha_exp, alpha[pv1 * ub_len : pv1 * ub_len + ub_len]
                            )
                            T.copy(sumexp, sumexp_sv[pv1, :])
                            T.copy(m_i, m_i_sv[pv1, :])
                            T.copy(acc_s_ub, acc_s_half)
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(
                                acc_s_half,
                                ws_p[
                                    cid, pv1, vid * v_block : vid * v_block + v_block, :
                                ],
                            )
                            # The cross-flag is MTE3-tied, so it already fences the
                            # ws_p copy above -- no PIPE_ALL barrier needed.
                            T.set_cross_flag("MTE3", _FLAG_P_READY)

                        if valid2:
                            pv2 = (g - 2) % 2
                            # Normalize O = (P@V) / sumexp, per head, and write out.
                            # PV_READY (a cross-core flag) already fences the cube's
                            # ws_o write; no PIPE_ALL barrier needed after the wait.
                            T.wait_cross_flag(_FLAG_PV_READY)
                            T.copy(sumexp_sv[pv2, :], sumexp_rt)
                            T.copy(m_i_sv[pv2, :], m_i_rt)
                            T.pipe_barrier("v")
                            T.copy(
                                ws_o[
                                    cid,
                                    pv2,
                                    vid * v_block : vid * v_block + MERGE_HEADS,
                                    :,
                                ],
                                acc_o_work,
                            )
                            T.set_flag("mte2", "v", 2)
                            # Batched normalize O[h,:] /= sumexp[h]: one reciprocal
                            # over all heads, then a per-row broadcast multiply
                            # (brcb + row_muls) -- replaces the per-head div loop.
                            T.tile.reciprocal(recip, sumexp_rt)
                            T.pipe_barrier("v")
                            T.wait_flag("mte2", "v", 2)
                            T.tile.brcb(
                                recip_brd8,
                                recip[0:MERGE_HEADS],
                                (MERGE_HEADS + 7) // 8,
                                1,
                                8,
                            )
                            T.pipe_barrier("v")
                            T.tile.row_muls(
                                acc_o_work, acc_o_work, recip_brd8, MERGE_HEADS, D, D
                            )
                            T.pipe_barrier("v")
                            T.copy(acc_o_work, acc_o_half[0:MERGE_HEADS, :])
                            T.copy(
                                ws_o[
                                    cid,
                                    pv2,
                                    vid * v_block + MERGE_HEADS : vid * v_block
                                    + 2 * MERGE_HEADS,
                                    :,
                                ],
                                acc_o_work2,
                            )
                            T.set_flag("mte2", "v", 3)
                            T.wait_flag("mte2", "v", 3)
                            T.tile.brcb(
                                recip_brd8,
                                recip[MERGE_HEADS : 2 * MERGE_HEADS],
                                (MERGE_HEADS + 7) // 8,
                                1,
                                8,
                            )
                            T.pipe_barrier("v")
                            T.tile.row_muls(
                                acc_o_work2, acc_o_work2, recip_brd8, MERGE_HEADS, D, D
                            )
                            T.pipe_barrier("v")
                            T.copy(
                                acc_o_work2,
                                acc_o_half[MERGE_HEADS : 2 * MERGE_HEADS, :],
                            )
                            T.set_flag("v", "mte3", 1)
                            T.wait_flag("v", "mte3", 1)
                            T.copy(
                                acc_o_half,
                                Output[t2, vid * v_block : vid * v_block + v_block, :],
                            )
                            # LSE = ln(sum) + max  (sink already folded into both).
                            T.tile.ln(lse_ub, sumexp_rt)
                            T.pipe_barrier("v")
                            T.tile.add(lse_ub, lse_ub, m_i_rt)
                            # Add (V) writes lse_ub; the GM store is MTE3 -> fence
                            # with a V_MTE3 event, not a PIPE_ALL barrier.
                            T.set_flag("v", "mte3", 2)
                            T.wait_flag("v", "mte3", 2)
                            T.copy(
                                lse_ub,
                                LSE_out[t2, vid * v_block : vid * v_block + v_block],
                            )
                    T.wait_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 1)

        return sparse_attn_sharedkv_swa

    return _make()
