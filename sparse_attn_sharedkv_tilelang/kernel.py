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

    if scenario == 3:
        raise NotImplementedError(
            "Clean-room reimplementation in progress: scenarios 1 (SWA) and 2 "
            "(CFA) are available; scenario 3 (SCFA) is coming next."
        )

    # CFA (scenario 2) = SWA ori pass + a DENSE compressed pass over cmp_kv,
    # merged by ONE running flash softmax (faithful to swa_*.h templateMode ==
    # CFA_TEMPLATE; CFA is compiled from the SAME SparseAttnSharedkvSwa class,
    # NOT scfa_*.h). Per query: 1 ori tile (sliding window) + NI_cmp dense cmp
    # tiles. cmp tile t covers compressed keys [t*BI, t*BI+BI); the per-query
    # causal threshold (valid compressed-key count) is (s_global + 1)//cmp_ratio
    # = floor((act_kv - act_q + s + 1)/cmp_ratio), matching cmpMaskMode=3. The
    # over-read tail (>= threshold) is masked. NI_cmp = K/BI is compile-time
    # (=1 for the cfa test: cmp_ratio=128, seqused~8192 -> K=128). All CFA codegen
    # below is gated `if scenario >= 2:` so scenario 1 stays byte-identical.
    is_cfa = scenario >= 2
    NI_cmp_eff = max(NI_cmp, 1)  # workspace tile dim (>=1 so SWA allocs a dummy)

    # CFA requires cmp_block_size to be a multiple of BI, so each BI_half KV chunk
    # of a dense cmp tile lies entirely within one page block (one DataCopy per
    # half, no segment split). The per-tile page index/row are computed in-body as
    # TIR expressions of the loop var (a Python list indexed by the TIR loop Var
    # fails: "list indices must be ... not Var").
    if is_cfa:
        assert cmp_block_size % BI == 0, (
            "CFA: cmp_block_size must be a multiple of BI (BI-aligned cmp paging)"
        )

    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15, 16, 17, 18, 19, 20])
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
            # CFA dense compressed scores (scenario>=2); SWA allocates a [.,.,1,.,.]
            # dummy and never reads it (no codegen -> SWA byte-identical).
            ws_score_cmp: T.Tensor(
                [core_num, 2, NI_cmp_eff, H_per_block, BI], accum_dtype
            ),
            # CFA compressed P (softmax probs) and compressed P@V output. SWA
            # allocates [.,.,.,1]/[.,.,.,1] dummies and never reads them.
            ws_p_cmp: T.Tensor([core_num, 2, H_per_block, BI], dtype),
            ws_o_cmp: T.Tensor([core_num, 2, H_per_block, D], accum_dtype),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- L1 (cube-side) buffers, double-buffered over pid parity. ----
                q_l1 = T.alloc_L1([2, H_per_block, D], dtype)
                kv_lo = T.alloc_L1([2, BI_half, D], dtype)
                kv_hi = T.alloc_L1([2, BI_half, D], dtype)
                p_lo = T.alloc_L1([H_per_block, BI_half], dtype)
                p_hi = T.alloc_L1([H_per_block, BI_half], dtype)
                # CFA: one reused cmp-KV L1 half (lo then hi, serially), 64KB in
                # the L1 headroom after the ori buffers; cmp QK reuses the ori
                # acc_s_a/acc_s_b L0C tiles (free after the ori fixpipe). Faithful
                # to swa_block_cube.h ComputeMm1 isOri=false (cmp_KV via
                # cmp_block_table). Declared UNCONDITIONALLY (like Ascend C always
                # declares the cmp buffers) -- TVMScript scopes names to the `if`
                # frame they are defined in, so an `if is_cfa:` alloc would be
                # invisible later; SWA never touches it (NI_cmp==0) -> DCE'd.
                cmp_kv_l1 = T.alloc_L1([BI_half, D], dtype)
                # CFA cmp PV reads P_cmp into its OWN L1 halves (not the ori p_lo/
                # p_hi) so there is no ori-PV<->cmp-PV p-buffer reuse baton. 8KB each
                # in the L1 headroom after cmp_kv_l1.
                p_cmp_lo = T.alloc_L1([H_per_block, BI_half], dtype)
                p_cmp_hi = T.alloc_L1([H_per_block, BI_half], dtype)
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
                # CFA: per-pid saved cmp-tile flash rescale alpha = exp(m_ori -
                # m_cmp), applied to the ori P@V at output-merge time (valid2).
                alpha_cmp_sv = T.alloc_ub([2, ub_len], accum_dtype)
                alpha_cmp_rt = T.alloc_ub([ub_len], accum_dtype)

                # Address map. cmp_kv_l1 (64KB in the L1 headroom after p_hi) is
                # always included; for SWA it is unused and DCE'd. A single literal
                # passed directly to annotate_address -- a dict assigned to a local
                # is intercepted as a LetStmt, a `**`-unpack is rejected by the
                # parser, and an `if is_cfa:` branch scopes any names to its frame.
                T.annotate_address(
                    {
                        q_l1: 0,
                        kv_lo: 128 * KB,
                        kv_hi: 256 * KB,
                        p_lo: 384 * KB,
                        p_hi: 392 * KB,
                        cmp_kv_l1: 400 * KB,
                        p_cmp_lo: 464 * KB,
                        p_cmp_hi: 472 * KB,
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
                        # softmax_tmp + alpha_exp moved OUT of the kv_ub_multi
                        # region (64KB) into the free [104KB,128KB) UB gap
                        # (acc_s_half ends @104KB, acc_s_ub_ starts @128KB).
                        # acc_o_half[0:16] (bf16, bytes [64KB,80KB)) used to ALIAS
                        # softmax_tmp byte-for-byte -> the NEXT query's valid1
                        # softmax_flashv2 (V) raced THIS query's Output DMA (MTE3
                        # reading acc_o_half[0:16]), corrupting the first 16 heads
                        # of every vector lane. Data-independent: even thr==0
                        # tokens (cmp a no-op) failed. SWA hid it (one
                        # softmax_flashv2 + short valid2 -> the DMA drained first);
                        # CFA's 2nd flash + longer cmp-merge valid2 made the
                        # cross-iteration race fire. Ascend C keeps these in
                        # DISTINCT TBufs (outputBuff1 vs softmaxTmpUb); un-aliasing
                        # restores that separation (no extra UB, no serialization).
                        softmax_tmp: 104 * KB,
                        alpha_exp: 104 * KB + 16 * KB + 512,
                        sumexp_sv: ub_addr["mask_sel"] + 32,
                        m_i_sv: ub_addr["mask_sel"] + 32 + 256,
                        sumexp_rt: ub_addr["mask_sel"] + 32 + 512,
                        m_i_rt: ub_addr["mask_sel"] + 32 + 640,
                        recip: ub_addr["mask_sel"] + 32 + 768,
                        recip_brd8: ub_addr["mask_sel"] + 32 + 896,
                        alpha_cmp_sv: ub_addr["mask_sel"] + 32 + 1408,
                        alpha_cmp_rt: ub_addr["mask_sel"] + 32 + 1664,
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
                # Loop over THIS core's assigned tasks only -- Ascend C's ProcessBalance
                # walks the core's [linear_start, linear_end) gloop, not the global
                # index space. Bounding by total_work (= batch*max_seq) made every one
                # of the 24 cores spin batch*max_seq iterations while doing real work on
                # only ~1/core_num of them; the wasted iterations still re-ran the whole
                # per-iteration validity/address scalar header + 3 metadata GM reads,
                # which is what pinned aic_scalar_ratio=0.63 / aiv_scalar_ratio=0.29.
                # +2 drains the 3-stage cross-query pipeline tail.
                num_local = T.alloc_var(indices_dtype, init=0)
                num_local = T.if_then_else(
                    core_enable != 0, linear_end - linear_start, 0
                )

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
                    # Seed the cmp-KV L1 slot as free (M->MTE2, id 6) so the first
                    # cmp gather's wait passes. Gated by a range() (compile-time
                    # unroll: emitted only when NI_cmp>0) rather than `if is_cfa:`,
                    # which TVMScript turns into a runtime/​scoped IfFrame.
                    for _ in range(1 if NI_cmp > 0 else 0):
                        T.set_flag("m", "mte2", 6)
                    for g in T.serial(num_local + 2):
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
                            # PRELOAD overlap: issue Q + the two KV-half gathers,
                            # each with its OWN mte2->m flag, then gemm the LO half as
                            # soon as Q + kv_lo are in while kv_hi is still streaming
                            # GM->L1 -- the GM->L1 || Mmad overlap of Ascend C
                            # ComputeMm1, at half granularity. All inside the QK block,
                            # so QK still precedes PV (no cross-core deadlock).
                            T.copy(Q[t0, 0:n_heads, 0:D], q_l1[pa, :, :])
                            T.set_flag("mte2", "m", 3)
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
                            T.set_flag("mte2", "m", 4)
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
                            T.set_flag("mte2", "m", 5)
                            # --- Q@K^T LO half + Fixpipe, faithful ComputeMm1:
                            # the gemm's last Mmad sets unitFlag 0b11 and the
                            # immediately-following Fixpipe (unit_flag=0b11)
                            # consumes it, so NO explicit m->fix/fix->m barrier
                            # between Mmad and Fixpipe. kv_lo (4) + Q (3) ready;
                            # kv_hi (5) keeps streaming on MTE2 meanwhile. ---
                            T.wait_flag("mte2", "m", 3)
                            T.wait_flag("mte2", "m", 4)
                            T.wait_flag("fix", "m", 0)
                            T.gemm_v0(
                                q_l1[pa, :, :],
                                kv_lo[pa, :, :],
                                acc_s_a,
                                transpose_B=True,
                                init=True,
                                unit_flag=True,
                            )
                            T.fixpipe(
                                acc_s_a,
                                ws_score[cid, pa, 0:H_per_block, 0:BI_half],
                                unit_flag=0b11,
                            )
                            T.set_flag("fix", "m", 0)
                            # --- Q@K^T HI half + Fixpipe (kv_hi finished loading
                            # during the LO gemm/fixpipe above). ---
                            T.wait_flag("mte2", "m", 5)
                            T.wait_flag("fix", "m", 1)
                            T.gemm_v0(
                                q_l1[pa, :, :],
                                kv_hi[pa, :, :],
                                acc_s_b,
                                transpose_B=True,
                                init=True,
                                unit_flag=True,
                            )
                            T.fixpipe(
                                acc_s_b,
                                ws_score[cid, pa, 0:H_per_block, BI_half:BI],
                                unit_flag=0b11,
                            )
                            T.set_flag("fix", "m", 1)

                            # ---- CFA dense compressed QK: Q @ cmp_kv^T over NI_cmp
                            # dense tiles [t*BI, t*BI+BI). Mirrors the ori paged
                            # gather but from cmp_KV via cmp_block_table; reuses
                            # acc_s_a/acc_s_b (free after the ori fixpipe) and one
                            # reused cmp KV L1 half. T.serial(NI_cmp) is a TIR For
                            # (empty for SWA where NI_cmp==0). Page index/row are
                            # TIR expressions of the loop var (cmp_block_size % BI
                            # == 0 -> each half is within one block). Writes
                            # ws_score_cmp; vector merge + cmp PV come next. ----
                            for tcmp in T.serial(NI_cmp):
                                # -- lo half [t*BI, t*BI+BI_half) (one block) --
                                T.wait_flag("m", "mte2", 6)
                                T.copy(
                                    cmp_KV[
                                        cmp_block_table[
                                            b0, (tcmp * BI) // cmp_block_size
                                        ],
                                        (tcmp * BI) % cmp_block_size : (tcmp * BI)
                                        % cmp_block_size
                                        + BI_half,
                                        0,
                                        :,
                                    ],
                                    cmp_kv_l1[0:BI_half, :],
                                )
                                T.set_flag("mte2", "m", 6)
                                T.wait_flag("mte2", "m", 6)
                                T.wait_flag("fix", "m", 0)
                                T.gemm_v0(
                                    q_l1[pa, :, :],
                                    cmp_kv_l1,
                                    acc_s_a,
                                    transpose_B=True,
                                    init=True,
                                    unit_flag=True,
                                )
                                T.fixpipe(
                                    acc_s_a,
                                    ws_score_cmp[
                                        cid, pa, tcmp, 0:H_per_block, 0:BI_half
                                    ],
                                    unit_flag=0b11,
                                )
                                T.set_flag("fix", "m", 0)
                                T.set_flag("m", "mte2", 6)
                                # -- hi half [t*BI+BI_half, t*BI+BI) (one block) --
                                T.wait_flag("m", "mte2", 6)
                                T.copy(
                                    cmp_KV[
                                        cmp_block_table[
                                            b0,
                                            (tcmp * BI + BI_half) // cmp_block_size,
                                        ],
                                        (tcmp * BI + BI_half) % cmp_block_size : (
                                            tcmp * BI + BI_half
                                        )
                                        % cmp_block_size
                                        + BI_half,
                                        0,
                                        :,
                                    ],
                                    cmp_kv_l1[0:BI_half, :],
                                )
                                T.set_flag("mte2", "m", 6)
                                T.wait_flag("mte2", "m", 6)
                                T.wait_flag("fix", "m", 1)
                                T.gemm_v0(
                                    q_l1[pa, :, :],
                                    cmp_kv_l1,
                                    acc_s_b,
                                    transpose_B=True,
                                    init=True,
                                    unit_flag=True,
                                )
                                T.fixpipe(
                                    acc_s_b,
                                    ws_score_cmp[
                                        cid, pa, tcmp, 0:H_per_block, BI_half:BI
                                    ],
                                    unit_flag=0b11,
                                )
                                T.set_flag("fix", "m", 1)
                                T.set_flag("m", "mte2", 6)
                            # Signal the vector ONLY after BOTH ori and cmp QK are
                            # written (CFA vector reads ws_score AND ws_score_cmp).
                            # SWA's cmp loop is empty so this sits right after the
                            # ori QK, unchanged.
                            T.set_cross_flag("FIX", _FLAG_SCORE_READY)

                        if valid1:
                            pb = (g - 1) % 2
                            # query g-1's batch, for the CFA cmp PV re-gather.
                            b1 = T.if_then_else(in_range1, pid1 // max_seq, 0)
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
                            # ---- CFA cmp PV: P_cmp @ V_cmp -> acc_o_l0c (reused
                            # after the ori PV copy-out) -> ws_o_cmp. Re-gathers cmp
                            # KV (query g-1, batch b1) into cmp_kv_l1 (lo then hi, the
                            # SAME m/mte2(6) baton as the cmp QK) and reads P_cmp into
                            # its own p_cmp_lo/p_cmp_hi (mte2/m 7 fence). Faithful to
                            # swa_block_cube.h ComputeMm2 isOri=false. ----
                            for tcmp in T.serial(NI_cmp):
                                # -- lo half --
                                T.wait_flag("m", "mte2", 6)
                                T.copy(
                                    cmp_KV[
                                        cmp_block_table[
                                            b1, (tcmp * BI) // cmp_block_size
                                        ],
                                        (tcmp * BI) % cmp_block_size : (tcmp * BI)
                                        % cmp_block_size
                                        + BI_half,
                                        0,
                                        :,
                                    ],
                                    cmp_kv_l1[0:BI_half, :],
                                )
                                T.set_flag("mte2", "m", 6)
                                T.copy(
                                    ws_p_cmp[cid, pb, 0:H_per_block, 0:BI_half],
                                    p_cmp_lo,
                                )
                                T.set_flag("mte2", "m", 7)
                                T.wait_flag("mte2", "m", 6)
                                T.wait_flag("mte2", "m", 7)
                                T.wait_flag("fix", "m", 0)
                                T.wait_flag("fix", "m", 1)
                                T.gemm_v0(p_cmp_lo, cmp_kv_l1, acc_o_l0c, init=True)
                                T.set_flag("m", "mte2", 6)
                                # -- hi half --
                                T.wait_flag("m", "mte2", 6)
                                T.copy(
                                    cmp_KV[
                                        cmp_block_table[
                                            b1,
                                            (tcmp * BI + BI_half) // cmp_block_size,
                                        ],
                                        (tcmp * BI + BI_half) % cmp_block_size : (
                                            tcmp * BI + BI_half
                                        )
                                        % cmp_block_size
                                        + BI_half,
                                        0,
                                        :,
                                    ],
                                    cmp_kv_l1[0:BI_half, :],
                                )
                                T.set_flag("mte2", "m", 6)
                                T.copy(
                                    ws_p_cmp[cid, pb, 0:H_per_block, BI_half:BI],
                                    p_cmp_hi,
                                )
                                T.set_flag("mte2", "m", 7)
                                T.wait_flag("mte2", "m", 6)
                                T.wait_flag("mte2", "m", 7)
                                T.gemm_v0(p_cmp_hi, cmp_kv_l1, acc_o_l0c, init=False)
                                T.set_flag("m", "mte2", 6)
                                T.set_flag("m", "fix", 1)
                                T.wait_flag("m", "fix", 1)
                                T.copy(acc_o_l0c, ws_o_cmp[cid, pb, 0:H_per_block, 0:D])
                                T.set_flag("fix", "m", 0)
                                T.set_flag("fix", "m", 1)
                            # PV_READY after BOTH ori and cmp PV are written.
                            T.set_cross_flag("FIX", _FLAG_PV_READY)
                    T.wait_flag("fix", "m", 0)
                    T.wait_flag("fix", "m", 1)
                    # Drain the cmp-KV slot flag (M->MTE2 id 6): the loop seeds it
                    # once and each valid0 iter nets one extra set, leaving one
                    # un-consumed set per launch. Event IDs are HW state that
                    # persists across kernel launches, so a leftover set makes the
                    # next launch's first wait match the stale token -> deadlock in
                    # a repeated-call (warmup) loop. Consume it here to balance.
                    for _ in range(1 if NI_cmp > 0 else 0):
                        T.wait_flag("m", "mte2", 6)

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
                    for g in T.serial(num_local + 2):
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

                        # NOTE: the score-prefetch stage (formerly `if valid0`)
                        # was MOVED to the END of this iteration (after valid2).
                        # It waits SCORE_READY(g) from the *current* cube QK(g);
                        # keeping it first stalled the whole vector iteration
                        # (incl. the ready softmax(g-1)/output(g-2)) behind the
                        # cross-core wait. Doing the ready work first and the
                        # prefetch last matches Ascend C's PreloadPipeline
                        # (process past queries; load the current one last) and
                        # removes the front-of-iteration bubble.
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
                            # ori V is now done reading acc_s_ub -> let the cmp MTE2
                            # load reuse it (V->MTE2 fence, CFA only).
                            for _ in range(1 if NI_cmp > 0 else 0):
                                T.set_flag("v", "mte2", 6)
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(
                                acc_s_half,
                                ws_p[
                                    cid, pv1, vid * v_block : vid * v_block + v_block, :
                                ],
                            )
                            # ori MTE3 is now done reading acc_s_half -> let the cmp V
                            # reuse it (MTE3->V fence, CFA only). Without these two
                            # fences the cmp clobbers acc_s_ub/acc_s_half while ori's
                            # P->ws_p is still in flight -> ori P corrupted.
                            for _ in range(1 if NI_cmp > 0 else 0):
                                T.set_flag("mte3", "v", 6)
                            # ---- CFA: second flash over the dense cmp tile, merged
                            # into the running (m_i, sumexp) (Ascend C
                            # SoftmaxFlashV2Compute, isFirstSInnerLoop=false). Reuses
                            # acc_s_ub (P_ori already copied to ws_p) + acc_s_half.
                            # Mask mode 3: valid cmp cols = (s_global1+1)//cmp_ratio,
                            # mask the rest to -inf (running max stays finite -> no
                            # NaN even when thr==0). Overwrites sumexp_sv/m_i_sv[pv1]
                            # with the post-cmp running state and saves the cmp
                            # rescale alpha = exp(m_ori - m_cmp) for the valid2 merge.
                            # range() gate so SWA (NI_cmp==0) emits nothing. ----
                            for _ in range(1 if NI_cmp > 0 else 0):
                                # Wait for ori to finish with acc_s_ub (V read) and
                                # acc_s_half (MTE3 read) before the cmp reuses them.
                                T.wait_flag("v", "mte2", 6)
                                T.wait_flag("mte3", "v", 6)
                                T.copy(
                                    ws_score_cmp[
                                        cid,
                                        pv1,
                                        0,
                                        vid * v_block : vid * v_block + v_block,
                                        :,
                                    ],
                                    acc_s_ub,
                                )
                                # Fence the GM->UB DMA (MTE2) before reading acc_s_ub
                                # (pipe_barrier("v") would NOT drain MTE2).
                                T.set_flag("mte2", "v", 5)
                                T.wait_flag("mte2", "v", 5)
                                T.tile.createvecindex(idx_int, 0)
                                T.copy(idx_int, idx_float)
                                T.pipe_barrier("v")
                                T.tile.compare(
                                    mask_sel,
                                    idx_float,
                                    T.float32((s_global1 + 1) // cmp_ratio),
                                    "LT",
                                )
                                T.pipe_barrier("v")
                                for h_i in T.serial(v_block):
                                    T.tile.select(
                                        acc_s_ub[h_i, :],
                                        mask_sel,
                                        acc_s_ub[h_i, :],
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
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
                                # CFA cmp-tile flash rescale alpha = exp(m_ori -
                                # m_cmp), computed EXPLICITLY from the running maxes
                                # rather than softmax_flashv2's per-row exp_out. The
                                # fused primitive's exp_out is the ONE softmax output
                                # SWA never consumes (the ori flash discards alpha;
                                # only new_sum/new_max are validated) -> a per-row
                                # exp_out error at deal_row_count=32 is invisible in
                                # SWA and in Ascend C (which always vec-splits to 16
                                # rows). m_i_prev still holds the post-ori max (the
                                # primitive reads in_max read-only); m_i is post-cmp.
                                T.tile.sub(alpha_exp, m_i_prev, m_i)
                                T.pipe_barrier("v")
                                T.tile.exp(alpha_exp, alpha_exp)
                                T.pipe_barrier("v")
                                T.copy(alpha_exp, alpha_cmp_sv[pv1, :])
                                T.copy(sumexp, sumexp_sv[pv1, :])
                                T.copy(m_i, m_i_sv[pv1, :])
                                T.copy(acc_s_ub, acc_s_half)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    acc_s_half,
                                    ws_p_cmp[
                                        cid,
                                        pv1,
                                        vid * v_block : vid * v_block + v_block,
                                        :,
                                    ],
                                )
                            # P_READY only after BOTH ori P (ws_p) and cmp P
                            # (ws_p_cmp) are written; MTE3-tied so it fences them.
                            T.set_cross_flag("MTE3", _FLAG_P_READY)

                        if valid2:
                            pv2 = (g - 2) % 2
                            # Normalize O = (P@V) / sumexp, per head, and write out.
                            # PV_READY (a cross-core flag) already fences the cube's
                            # ws_o write; no PIPE_ALL barrier needed after the wait.
                            T.wait_cross_flag(_FLAG_PV_READY)
                            T.copy(sumexp_sv[pv2, :], sumexp_rt)
                            T.copy(m_i_sv[pv2, :], m_i_rt)
                            # CFA: load the cmp rescale alpha = exp(m_ori - m_cmp)
                            # for the O = alpha*O_ori + O_cmp merge below.
                            for _ in range(1 if NI_cmp > 0 else 0):
                                T.copy(alpha_cmp_sv[pv2, :], alpha_cmp_rt)
                            T.pipe_barrier("v")
                            T.copy(
                                ws_o[
                                    cid,
                                    pv2,
                                    vid * v_block + MERGE_HEADS : vid * v_block
                                    + 2 * MERGE_HEADS,
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
                            # CFA pass-1 merge: acc_o_work = alpha*O_ori + O_cmp
                            # (acc_o_work2 is the cmp temp; freed before pass 2 needs
                            # it for O_ori). The recip-normalize below then divides by
                            # the post-cmp sumexp.
                            for _ in range(1 if NI_cmp > 0 else 0):
                                T.copy(
                                    ws_o_cmp[
                                        cid,
                                        pv2,
                                        vid * v_block + MERGE_HEADS : vid * v_block
                                        + 2 * MERGE_HEADS,
                                        :,
                                    ],
                                    acc_o_work2,
                                )
                                T.set_flag("mte2", "v", 6)
                                T.wait_flag("mte2", "v", 6)
                                T.tile.brcb(
                                    recip_brd8,
                                    alpha_cmp_rt[MERGE_HEADS : 2 * MERGE_HEADS],
                                    (MERGE_HEADS + 7) // 8,
                                    1,
                                    8,
                                )
                                T.pipe_barrier("v")
                                T.tile.row_muls(
                                    acc_o_work,
                                    acc_o_work,
                                    recip_brd8,
                                    MERGE_HEADS,
                                    D,
                                    D,
                                )
                                T.pipe_barrier("v")
                                T.tile.add(acc_o_work, acc_o_work, acc_o_work2)
                                T.pipe_barrier("v")
                                # acc_o_work2 (cmp temp) now free -> pass-2 reloads it
                                # as O_ori (V->MTE2 fence, R1).
                                T.set_flag("v", "mte2", 8)
                            T.tile.brcb(
                                recip_brd8,
                                recip[MERGE_HEADS : 2 * MERGE_HEADS],
                                (MERGE_HEADS + 7) // 8,
                                1,
                                8,
                            )
                            T.pipe_barrier("v")
                            T.tile.row_muls(
                                acc_o_work, acc_o_work, recip_brd8, MERGE_HEADS, D, D
                            )
                            T.pipe_barrier("v")
                            T.copy(
                                acc_o_work, acc_o_half[MERGE_HEADS : 2 * MERGE_HEADS, :]
                            )
                            for _ in range(1 if NI_cmp > 0 else 0):
                                # acc_o_work now free -> pass-2 reuses it as the cmp
                                # temp (V->MTE2 fence, R2); and wait for pass-1's read
                                # of acc_o_work2 before reloading it as O_ori (R1).
                                T.set_flag("v", "mte2", 9)
                                T.wait_flag("v", "mte2", 8)
                            T.copy(
                                ws_o[
                                    cid,
                                    pv2,
                                    vid * v_block : vid * v_block + MERGE_HEADS,
                                    :,
                                ],
                                acc_o_work2,
                            )
                            T.set_flag("mte2", "v", 3)
                            T.wait_flag("mte2", "v", 3)
                            # CFA pass-2 merge: acc_o_work2 = alpha*O_ori + O_cmp
                            # (acc_o_work, free after pass 1's copy, is the cmp temp).
                            for _ in range(1 if NI_cmp > 0 else 0):
                                T.wait_flag("v", "mte2", 9)
                                T.copy(
                                    ws_o_cmp[
                                        cid,
                                        pv2,
                                        vid * v_block : vid * v_block + MERGE_HEADS,
                                        :,
                                    ],
                                    acc_o_work,
                                )
                                T.set_flag("mte2", "v", 7)
                                T.wait_flag("mte2", "v", 7)
                                T.tile.brcb(
                                    recip_brd8,
                                    alpha_cmp_rt[0:MERGE_HEADS],
                                    (MERGE_HEADS + 7) // 8,
                                    1,
                                    8,
                                )
                                T.pipe_barrier("v")
                                T.tile.row_muls(
                                    acc_o_work2,
                                    acc_o_work2,
                                    recip_brd8,
                                    MERGE_HEADS,
                                    D,
                                    D,
                                )
                                T.pipe_barrier("v")
                                T.tile.add(acc_o_work2, acc_o_work2, acc_o_work)
                                T.pipe_barrier("v")
                            T.tile.brcb(
                                recip_brd8,
                                recip[0:MERGE_HEADS],
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
                                acc_o_half[0:MERGE_HEADS, :],
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

                        # --- Score prefetch for the NEXT query (moved here from
                        # the front of the iteration). Waits SCORE_READY(g) from
                        # the current cube QK(g); by now the vector has already
                        # done softmax(g-1)/output(g-2), so cube QK(g) is done
                        # and this no longer stalls. Loads score(g) into
                        # acc_s_ub_[g%2] for next iteration's softmax, and builds
                        # the partial-window mask into mask_ub[g%2]. ---
                        if valid0:
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
                    T.wait_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 1)

        return sparse_attn_sharedkv_swa

    if return_prim_func:
        # ``@tilelang.jit`` keeps the original builder under ``__wrapped__``
        # (set by functools.wraps); call it to materialize the raw PrimFunc
        # without triggering the JIT compile / bisheng step.
        return _make.__wrapped__()
    return _make()
