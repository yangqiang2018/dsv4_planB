import tilelang
from tilelang import language as T
from tvm import ir as tvm_ir
from tvm import tir as tvm_tir


def _sub_tile2(buf, row0, rows, col0, cols):
    return tvm_tir.BufferRegion(
        buf,
        [
            tvm_ir.Range.from_min_extent(row0, rows),
            tvm_ir.Range.from_min_extent(col0, cols),
        ],
    )


def _sub_tile(buf, row0, rows, cols):
    return tvm_tir.BufferRegion(
        buf,
        [
            tvm_ir.Range.from_min_extent(row0, rows),
            tvm_ir.Range.from_min_extent(0, cols),
        ],
    )


tilelang.disable_cache()
tilelang.cache.clear_cache()

DEFAULT_CORE_NUM = 24

DEFAULT_BLOCK_I = 128

_SAS_META_SIZE = 1024
_FA_METADATA_SIZE = 8
_FA_CORE_ENABLE_INDEX = 0
_FA_BN2_START_INDEX = 1
_FA_M_START_INDEX = 2
_FA_S2_START_INDEX = 3
_FA_BN2_END_INDEX = 4
_FA_M_END_INDEX = 5
_FA_S2_END_INDEX = 6

_FLAG_KV_READY = 0
_FLAG_SCORE_READY = 1
_FLAG_P_READY = 2
_FLAG_PV_READY = 3
_FLAG_ITER_DONE = 4


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

    gqa_group = n_heads // n_kv_heads
    BI = block_I
    D = head_dim
    accum_dtype = "float"
    indices_dtype = "int32"

    ori_window_max = ori_win_left + 1
    NI_ori = (ori_window_max + BI - 1) // BI
    NI_cmp = topk_cmp // BI
    NI_total = NI_ori + NI_cmp
    BI_half = BI // 2
    is_cfa = scenario == 2
    cube_direct = (NI_cmp == 0) or is_cfa

    H_per_block = gqa_group
    v_block = H_per_block // 2
    ub_len = max(32 // 4, v_block)
    mask_w = ((BI // 8 + 31) // 32) * 32
    GATHER_ROWS = 16
    assert (BI // 2) % GATHER_ROWS == 0, "BI//2 must be a multiple of GATHER_ROWS"
    N_GATHER_PASS = (BI // 2) // GATHER_ROWS
    MERGE_HEADS = 16
    assert v_block % MERGE_HEADS == 0, "v_block must be a multiple of MERGE_HEADS"
    N_MERGE_PASS = v_block // MERGE_HEADS
    assert N_MERGE_PASS == 2, "Q4 cube_direct merge unroll assumes N_MERGE_PASS == 2"

    q_shape = [total_tokens, n_heads, D]
    out_shape = [total_tokens, n_heads, D]
    ori_kv_shape = [ori_block_num, ori_block_size, n_kv_heads, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, n_kv_heads, D]
    ori_bt_shape = [batch, ori_table_len]
    cmp_bt_shape = [batch, cmp_table_len]
    indices_shape = [total_tokens, n_kv_heads, max(NI_cmp, 1) * BI]

    KB = 1024
    l1_addr = {
        "q_l1": 0,
        "kv_lo": 64 * KB,
        "kv_hi": 192 * KB,
        "p_lo": 320 * KB,
        "p_hi": 328 * KB,
    }
    l0c_addr = {"acc_s_l0c": 0, "acc_o_l0c": 0}
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
        "mask_ub_2": 176 * KB + 1856,
        "alpha": 176 * KB + 2048,
        "mask_sel": 176 * KB + 2304,
        "acc_o_half": 64 * KB,
    }

    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15, 16, 17])
    def _make():
        @T.prim_func
        def sparse_attn_sharedkv(
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
                q_l1 = T.alloc_L1([H_per_block, D], dtype)
                kv_lo = T.alloc_L1([2, BI_half, D], dtype)
                kv_hi = T.alloc_L1([2, BI_half, D], dtype)
                p_lo = T.alloc_L1([H_per_block, BI_half], dtype)
                p_hi = T.alloc_L1([H_per_block, BI_half], dtype)
                acc_s_l0c = T.alloc_L0C([H_per_block, BI_half], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_work = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_work2 = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_ub = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                m_i = T.alloc_ub([ub_len], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
                sinks_ub = T.alloc_ub([ub_len], accum_dtype)
                lse_ub = T.alloc_ub([ub_len], accum_dtype)
                alpha = T.alloc_ub([2 * ub_len], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_ub_ = T.alloc_ub([2 * v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)
                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                kv_ub_multi = T.alloc_ub([2 * GATHER_ROWS, D], dtype)
                softmax_tmp = T.alloc_ub([16 * KB], "uint8")
                alpha_exp = T.alloc_ub([ub_len], accum_dtype)
                alpha_brd8 = T.alloc_ub([MERGE_HEADS, 8], accum_dtype)
                mask_ub = T.alloc_ub([2, mask_w], "uint8")
                mask_ub_2 = T.alloc_ub([2, mask_w], "uint8")
                mask_sel = T.alloc_ub([mask_w], "uint8")

                T.annotate_address(
                    {
                        q_l1: l1_addr["q_l1"],
                        kv_lo: l1_addr["kv_lo"],
                        kv_hi: l1_addr["kv_hi"],
                        p_lo: l1_addr["p_lo"],
                        p_hi: l1_addr["p_hi"],
                        acc_s_l0c: l0c_addr["acc_s_l0c"],
                        acc_o_l0c: l0c_addr["acc_o_l0c"],
                        acc_o: ub_addr["acc_o"],
                        acc_s_ub: ub_addr["acc_s_ub"],
                        acc_s_ub_: ub_addr["acc_s_ub_"],
                        acc_s_half: ub_addr["acc_s_half"],
                        m_i: ub_addr["m_i"],
                        m_i_prev: ub_addr["m_i_prev"],
                        sumexp: ub_addr["sumexp"],
                        sumexp_i_ub: ub_addr["sumexp_i_ub"],
                        sinks_ub: ub_addr["sinks_ub"],
                        lse_ub: ub_addr["lse_ub"],
                        idx_int: ub_addr["idx_int"],
                        idx_float: ub_addr["idx_float"],
                        alpha: ub_addr["alpha"],
                        kv_ub_multi: ub_addr["kv_ub_multi"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_ub_2: ub_addr["mask_ub_2"],
                        mask_sel: ub_addr["mask_sel"],
                        acc_o_ub: ub_addr["acc_o_ub"],
                        acc_o_half: ub_addr["acc_o_half"],
                    }
                )
                if cube_direct:
                    T.annotate_address(
                        {
                            softmax_tmp: ub_addr["kv_ub_multi"],
                            alpha_brd8: ub_addr["kv_ub_multi"] + 16 * KB,
                            alpha_exp: ub_addr["kv_ub_multi"] + 16 * KB + 512,
                            acc_o_work: ub_addr["acc_o"],
                            acc_o_work2: ub_addr["acc_o"] + 32 * KB,
                        }
                    )

                meta_base = cid * _FA_METADATA_SIZE
                core_enable = Metadata[meta_base + _FA_CORE_ENABLE_INDEX]
                bn2_start = Metadata[meta_base + _FA_BN2_START_INDEX]
                m_start = Metadata[meta_base + _FA_M_START_INDEX]
                bn2_end = Metadata[meta_base + _FA_BN2_END_INDEX]
                m_end = Metadata[meta_base + _FA_M_END_INDEX]
                linear_start = bn2_start * max_seq + m_start
                linear_end = bn2_end * max_seq + m_end

                total_work = batch * max_seq
                for slot in T.serial(total_work):
                    pid = linear_start + slot
                    if core_enable != 0 and pid < linear_end:
                        b_i = pid // max_seq
                        s_i = pid % max_seq
                        act_q = actual_q_len[b_i]
                        act_kv = actual_kv_len[b_i]
                        if s_i < act_q:
                            t_i = q_prefix[b_i] + s_i
                            s_global = act_kv - act_q + s_i
                            ori_right = s_global
                            ori_left_raw = s_global - ori_win_left
                            ori_left = T.if_then_else(ori_left_raw < 0, 0, ori_left_raw)
                            cmp_threshold = (s_global + 1) // cmp_ratio

                            with T.Scope("C"):
                                T.copy(Q[t_i, 0:n_heads, 0:D], q_l1)
                                T.barrier_all()
                                for t in range(NI_total + 1):
                                    if t < NI_total:
                                        pa = t % 2
                                        if cube_direct and t < NI_ori:
                                            for gp in range(BI_half // GATHER_ROWS):
                                                g0 = (
                                                    ori_left + t * BI + gp * GATHER_ROWS
                                                )
                                                bidx = g0 // ori_block_size
                                                rowc = g0 % ori_block_size
                                                if ori_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = ori_block_size - rowc
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            for gp in range(BI_half // GATHER_ROWS):
                                                g0 = (
                                                    ori_left
                                                    + t * BI
                                                    + BI_half
                                                    + gp * GATHER_ROWS
                                                )
                                                bidx = g0 // ori_block_size
                                                rowc = g0 % ori_block_size
                                                if ori_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = ori_block_size - rowc
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        ori_KV[
                                                            ori_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            T.barrier_all()
                                        elif cube_direct and is_cfa:
                                            for gp in range(BI_half // GATHER_ROWS):
                                                gc0 = (
                                                    t - NI_ori
                                                ) * BI + gp * GATHER_ROWS
                                                bidx = gc0 // cmp_block_size
                                                rowc = gc0 % cmp_block_size
                                                if cmp_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = cmp_block_size - rowc
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_lo[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            for gp in range(BI_half // GATHER_ROWS):
                                                gc0 = (
                                                    (t - NI_ori) * BI
                                                    + BI_half
                                                    + gp * GATHER_ROWS
                                                )
                                                bidx = gc0 // cmp_block_size
                                                rowc = gc0 % cmp_block_size
                                                if cmp_block_size - rowc >= GATHER_ROWS:
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + GATHER_ROWS,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : (gp + 1)
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                else:
                                                    n0 = cmp_block_size - rowc
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[b_i, bidx],
                                                            rowc : rowc + n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS : gp
                                                            * GATHER_ROWS
                                                            + n0,
                                                            :,
                                                        ],
                                                    )
                                                    T.copy(
                                                        cmp_KV[
                                                            cmp_block_table[
                                                                b_i, bidx + 1
                                                            ],
                                                            0 : GATHER_ROWS - n0,
                                                            0,
                                                            :,
                                                        ],
                                                        kv_hi[
                                                            pa,
                                                            gp * GATHER_ROWS + n0 : (
                                                                gp + 1
                                                            )
                                                            * GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                            T.barrier_all()
                                        else:
                                            T.wait_cross_flag(_FLAG_KV_READY)
                                            T.barrier_all()
                                            T.copy(
                                                ws_kv[cid, pa, 0:BI_half, 0:D],
                                                kv_lo[pa, :, :],
                                            )
                                            T.barrier_all()
                                            T.copy(
                                                ws_kv[cid, pa, BI_half:BI, 0:D],
                                                kv_hi[pa, :, :],
                                            )
                                            T.barrier_all()
                                        T.gemm_v0(
                                            q_l1,
                                            kv_lo[pa, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                        )
                                        T.set_flag("m", "fix", 0)
                                        T.wait_flag("m", "fix", 0)
                                        T.copy(
                                            acc_s_l0c,
                                            ws_score[cid, pa, 0:H_per_block, 0:BI_half],
                                        )
                                        T.set_flag("fix", "m", 1)
                                        T.wait_flag("fix", "m", 1)
                                        T.gemm_v0(
                                            q_l1,
                                            kv_hi[pa, :, :],
                                            acc_s_l0c,
                                            transpose_B=True,
                                            init=True,
                                        )
                                        T.set_flag("m", "fix", 2)
                                        T.wait_flag("m", "fix", 2)
                                        T.copy(
                                            acc_s_l0c,
                                            ws_score[
                                                cid, pa, 0:H_per_block, BI_half:BI
                                            ],
                                        )
                                        T.barrier_all()
                                        T.set_cross_flag("FIX", _FLAG_SCORE_READY)
                                    if t >= 1:
                                        pb = (t - 1) % 2
                                        T.wait_cross_flag(_FLAG_P_READY)
                                        T.barrier_all()
                                        T.copy(
                                            ws_p[cid, pb, 0:H_per_block, 0:BI_half],
                                            p_lo,
                                        )
                                        T.copy(
                                            ws_p[cid, pb, 0:H_per_block, BI_half:BI],
                                            p_hi,
                                        )
                                        T.set_flag("mte2", "m", 0)
                                        T.wait_flag("mte2", "m", 0)
                                        T.gemm_v0(
                                            p_lo, kv_lo[pb, :, :], acc_o_l0c, init=True
                                        )
                                        T.barrier_all()
                                        T.gemm_v0(
                                            p_hi, kv_hi[pb, :, :], acc_o_l0c, init=False
                                        )
                                        T.set_flag("m", "fix", 1)
                                        T.wait_flag("m", "fix", 1)
                                        T.copy(
                                            acc_o_l0c,
                                            ws_o[cid, pb, 0:H_per_block, 0:D],
                                        )
                                        T.barrier_all()
                                        T.set_cross_flag("FIX", _FLAG_PV_READY)

                            with T.Scope("V"):
                                T.copy(
                                    Sinks[vid * v_block : vid * v_block + v_block],
                                    m_i,
                                )
                                if cube_direct:
                                    for hp in range(N_MERGE_PASS):
                                        hb = hp * MERGE_HEADS
                                        T.tile.fill(acc_o_work, 0.0)
                                        T.barrier_all()
                                        T.copy(
                                            acc_o_work,
                                            ws_acc_o[
                                                cid,
                                                slot % 2,
                                                vid * v_block + hb : vid * v_block
                                                + hb
                                                + MERGE_HEADS,
                                                :,
                                            ],
                                        )
                                    T.barrier_all()
                                else:
                                    T.tile.fill(acc_o, 0.0)
                                T.tile.fill(sumexp, 1.0)
                                T.barrier_all()

                                T.set_flag("v", "mte2", 0)
                                T.set_flag("v", "mte2", 1)
                                for t in range(NI_total + 2):
                                    if t < NI_total:
                                        c0 = t
                                        pv0 = t % 2
                                        is_ori = c0 < NI_ori

                                        if is_ori:
                                            chunk_start = ori_left + c0 * BI
                                            T.tile.createvecindex(
                                                idx_int,
                                                chunk_start,
                                            )
                                            T.copy(idx_int, idx_float)
                                            T.barrier_all()
                                            T.tile.compare(
                                                mask_ub[pv0, :],
                                                idx_float,
                                                T.float32(ori_right),
                                                "LE",
                                            )
                                            T.barrier_all()
                                            if not cube_direct:
                                                for gp in range(N_GATHER_PASS):
                                                    pp = gp % 2
                                                    gh = pp * GATHER_ROWS
                                                    kv_row0 = (
                                                        vid * (BI // 2)
                                                        + gp * GATHER_ROWS
                                                    )
                                                    if gp >= 2:
                                                        T.wait_flag("mte3", "mte2", pp)
                                                    for r in range(GATHER_ROWS):
                                                        g_idx = (
                                                            chunk_start + kv_row0 + r
                                                        )
                                                        ori_blk = ori_block_table[
                                                            b_i,
                                                            g_idx // ori_block_size,
                                                        ]
                                                        ori_row = g_idx % ori_block_size
                                                        T.copy(
                                                            ori_KV[
                                                                ori_blk,
                                                                ori_row,
                                                                0,
                                                                :,
                                                            ],
                                                            kv_ub_multi[gh + r, :],
                                                        )
                                                    T.set_flag("mte2", "mte3", pp)
                                                    T.wait_flag("mte2", "mte3", pp)
                                                    T.copy(
                                                        kv_ub_multi[
                                                            gh : gh + GATHER_ROWS, :
                                                        ],
                                                        ws_kv[
                                                            cid,
                                                            pv0,
                                                            kv_row0 : kv_row0
                                                            + GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                    T.set_flag("mte3", "mte2", pp)
                                        else:
                                            if is_cfa:
                                                T.tile.createvecindex(
                                                    idx_int,
                                                    (c0 - NI_ori) * BI,
                                                )
                                            else:
                                                cmp_off = (c0 - NI_ori) * BI
                                                T.copy(
                                                    cmp_indices[
                                                        t_i,
                                                        0,
                                                        cmp_off : cmp_off + BI,
                                                    ],
                                                    idx_int,
                                                )
                                                T.barrier_all()
                                            T.copy(idx_int, idx_float)
                                            T.barrier_all()
                                            T.tile.compare(
                                                mask_ub[pv0, :],
                                                idx_float,
                                                T.float32(-0.5),
                                                "GT",
                                            )
                                            T.tile.compare(
                                                mask_ub_2[pv0, :],
                                                idx_float,
                                                T.float32(cmp_threshold),
                                                "LT",
                                            )
                                            T.barrier_all()
                                            T.tile.bitwise_and(
                                                mask_ub[pv0, :],
                                                mask_ub[pv0, :],
                                                mask_ub_2[pv0, :],
                                            )
                                            T.barrier_all()
                                            if not cube_direct:
                                                for gp in range(N_GATHER_PASS):
                                                    pp = gp % 2
                                                    gh = pp * GATHER_ROWS
                                                    kv_row0 = (
                                                        vid * (BI // 2)
                                                        + gp * GATHER_ROWS
                                                    )
                                                    if gp >= 2:
                                                        T.wait_flag("mte3", "mte2", pp)
                                                    for r in range(GATHER_ROWS):
                                                        cmp_idx = idx_int[kv_row0 + r]
                                                        safe_idx = T.if_then_else(
                                                            cmp_idx < 0, 0, cmp_idx
                                                        )
                                                        cmp_blk = cmp_block_table[
                                                            b_i,
                                                            safe_idx // cmp_block_size,
                                                        ]
                                                        cmp_row = (
                                                            safe_idx % cmp_block_size
                                                        )
                                                        T.copy(
                                                            cmp_KV[
                                                                cmp_blk, cmp_row, 0, :
                                                            ],
                                                            kv_ub_multi[gh + r, :],
                                                        )
                                                    T.set_flag("mte2", "mte3", pp)
                                                    T.wait_flag("mte2", "mte3", pp)
                                                    T.copy(
                                                        kv_ub_multi[
                                                            gh : gh + GATHER_ROWS, :
                                                        ],
                                                        ws_kv[
                                                            cid,
                                                            pv0,
                                                            kv_row0 : kv_row0
                                                            + GATHER_ROWS,
                                                            :,
                                                        ],
                                                    )
                                                    T.set_flag("mte3", "mte2", pp)
                                        if not cube_direct:
                                            T.wait_flag("mte3", "mte2", 0)
                                            T.wait_flag("mte3", "mte2", 1)
                                            T.set_cross_flag("MTE3", _FLAG_KV_READY)
                                    if t == 0:
                                        T.wait_cross_flag(_FLAG_SCORE_READY)
                                        T.wait_flag("v", "mte2", 0)
                                        T.copy(
                                            ws_score[
                                                cid,
                                                0,
                                                vid * v_block : vid * v_block + v_block,
                                                :,
                                            ],
                                            acc_s_ub_[0:v_block, :],
                                        )
                                        T.set_flag("mte2", "v", 0)
                                    if t >= 1:
                                        if t <= NI_total:
                                            pv1 = (t - 1) % 2
                                            T.wait_flag("mte2", "v", pv1)
                                            T.copy(mask_ub[pv1, :], mask_sel)
                                            for h_i in T.serial(v_block):
                                                T.tile.select(
                                                    acc_s_ub[h_i, :],
                                                    mask_sel,
                                                    acc_s_ub_[pv1 * v_block + h_i, :],
                                                    -T.infinity(accum_dtype),
                                                    "VSEL_TENSOR_SCALAR_MODE",
                                                )
                                            T.set_flag("v", "mte2", pv1)
                                            T.copy(m_i, m_i_prev)
                                            T.tile.mul(
                                                acc_s_ub,
                                                acc_s_ub,
                                                softmax_scale,
                                            )

                                            if cube_direct:
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
                                                    alpha_exp,
                                                    alpha[
                                                        pv1 * ub_len : pv1 * ub_len
                                                        + ub_len
                                                    ],
                                                )
                                            else:
                                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                                T.tile.max(m_i, m_i_prev, m_i)
                                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                                T.tile.exp(m_i_prev, m_i_prev)
                                                T.copy(
                                                    m_i_prev,
                                                    alpha[
                                                        pv1 * ub_len : pv1 * ub_len
                                                        + ub_len
                                                    ],
                                                )
                                                for h_i in range(v_block):
                                                    T.tile.sub(
                                                        acc_s_ub[h_i, :],
                                                        acc_s_ub[h_i, :],
                                                        m_i[h_i],
                                                    )
                                                T.tile.exp(acc_s_ub, acc_s_ub)
                                                T.reduce_sum(
                                                    acc_s_ub,
                                                    sumexp_i_ub,
                                                    dim=-1,
                                                )
                                                T.tile.mul(sumexp, sumexp, m_i_prev)
                                                T.tile.add(
                                                    sumexp,
                                                    sumexp,
                                                    sumexp_i_ub,
                                                )

                                            if t < NI_total:
                                                T.wait_cross_flag(_FLAG_SCORE_READY)
                                                T.wait_flag("v", "mte2", t % 2)
                                                T.copy(
                                                    ws_score[
                                                        cid,
                                                        t % 2,
                                                        vid * v_block : vid * v_block
                                                        + v_block,
                                                        :,
                                                    ],
                                                    acc_s_ub_[
                                                        (t % 2) * v_block : (t % 2)
                                                        * v_block
                                                        + v_block,
                                                        :,
                                                    ],
                                                )
                                                T.set_flag("mte2", "v", t % 2)

                                            T.copy(acc_s_ub, acc_s_half)
                                            T.set_flag("v", "mte3", 0)
                                            T.wait_flag("v", "mte3", 0)
                                            T.copy(
                                                acc_s_half,
                                                ws_p[
                                                    cid,
                                                    pv1,
                                                    vid * v_block : vid * v_block
                                                    + v_block,
                                                    :,
                                                ],
                                            )
                                            T.barrier_all()
                                            T.set_cross_flag("MTE3", _FLAG_P_READY)
                                    if t >= 2:
                                        pv2 = (t - 2) % 2
                                        T.wait_cross_flag(_FLAG_PV_READY)
                                        T.barrier_all()
                                        if cube_direct:
                                            T.copy(
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block : vid * v_block
                                                    + MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_work,
                                            )
                                            T.copy(
                                                ws_o[
                                                    cid,
                                                    pv2,
                                                    vid * v_block : vid * v_block
                                                    + MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_ub,
                                            )
                                            T.set_flag("mte2", "v", 2)
                                            T.wait_flag("mte2", "v", 2)
                                            T.tile.brcb(
                                                alpha_brd8,
                                                alpha[
                                                    pv2 * ub_len : pv2 * ub_len
                                                    + MERGE_HEADS
                                                ],
                                                (MERGE_HEADS + 7) // 8,
                                                1,
                                                8,
                                            )
                                            T.pipe_barrier("v")
                                            T.tile.row_muls(
                                                acc_o_work,
                                                acc_o_work,
                                                alpha_brd8,
                                                MERGE_HEADS,
                                                D,
                                                D,
                                            )
                                            T.pipe_barrier("v")
                                            for h_i in range(MERGE_HEADS):
                                                T.tile.add(
                                                    acc_o_work[h_i, :],
                                                    acc_o_work[h_i, :],
                                                    acc_o_ub[h_i, :],
                                                )
                                            T.set_flag("v", "mte2", 2)
                                            T.set_flag("v", "mte3", 1)
                                            T.wait_flag("v", "mte3", 1)
                                            T.copy(
                                                acc_o_work,
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block : vid * v_block
                                                    + MERGE_HEADS,
                                                    :,
                                                ],
                                            )
                                            T.copy(
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block + MERGE_HEADS : vid
                                                    * v_block
                                                    + 2 * MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_work2,
                                            )
                                            T.wait_flag("v", "mte2", 2)
                                            T.copy(
                                                ws_o[
                                                    cid,
                                                    pv2,
                                                    vid * v_block + MERGE_HEADS : vid
                                                    * v_block
                                                    + 2 * MERGE_HEADS,
                                                    :,
                                                ],
                                                acc_o_ub,
                                            )
                                            T.set_flag("mte2", "v", 3)
                                            T.wait_flag("mte2", "v", 3)
                                            T.tile.brcb(
                                                alpha_brd8,
                                                alpha[
                                                    pv2 * ub_len + MERGE_HEADS : pv2
                                                    * ub_len
                                                    + 2 * MERGE_HEADS
                                                ],
                                                (MERGE_HEADS + 7) // 8,
                                                1,
                                                8,
                                            )
                                            T.pipe_barrier("v")
                                            T.tile.row_muls(
                                                acc_o_work2,
                                                acc_o_work2,
                                                alpha_brd8,
                                                MERGE_HEADS,
                                                D,
                                                D,
                                            )
                                            T.pipe_barrier("v")
                                            for h_i in range(MERGE_HEADS):
                                                T.tile.add(
                                                    acc_o_work2[h_i, :],
                                                    acc_o_work2[h_i, :],
                                                    acc_o_ub[h_i, :],
                                                )
                                            T.set_flag("v", "mte3", 2)
                                            T.wait_flag("v", "mte3", 2)
                                            T.copy(
                                                acc_o_work2,
                                                ws_acc_o[
                                                    cid,
                                                    slot % 2,
                                                    vid * v_block + MERGE_HEADS : vid
                                                    * v_block
                                                    + 2 * MERGE_HEADS,
                                                    :,
                                                ],
                                            )
                                        else:
                                            for mp in range(N_MERGE_PASS):
                                                hbase = mp * MERGE_HEADS
                                                T.copy(
                                                    ws_o[
                                                        cid,
                                                        pv2,
                                                        vid * v_block + hbase : vid
                                                        * v_block
                                                        + hbase
                                                        + MERGE_HEADS,
                                                        :,
                                                    ],
                                                    acc_o_ub,
                                                )
                                                T.barrier_all()
                                                for h_i in range(MERGE_HEADS):
                                                    T.barrier_all()
                                                    T.tile.mul(
                                                        acc_o[hbase + h_i, :],
                                                        acc_o[hbase + h_i, :],
                                                        alpha[
                                                            pv2 * ub_len + hbase + h_i
                                                        ],
                                                    )
                                                    T.barrier_all()
                                                    T.tile.add(
                                                        acc_o[hbase + h_i, :],
                                                        acc_o[hbase + h_i, :],
                                                        acc_o_ub[h_i, :],
                                                    )
                                                    T.barrier_all()

                                T.wait_flag("v", "mte2", 0)
                                T.wait_flag("v", "mte2", 1)
                                if cube_direct:
                                    T.barrier_all()
                                    T.copy(
                                        ws_acc_o[
                                            cid,
                                            slot % 2,
                                            vid * v_block : vid * v_block + MERGE_HEADS,
                                            :,
                                        ],
                                        acc_o_work,
                                    )
                                    T.set_flag("mte2", "v", 2)
                                    T.wait_flag("mte2", "v", 2)
                                    for h_i in range(MERGE_HEADS):
                                        T.tile.div(
                                            acc_o_work[h_i, :],
                                            acc_o_work[h_i, :],
                                            sumexp[h_i],
                                        )
                                    T.pipe_barrier("v")
                                    T.copy(
                                        acc_o_work,
                                        acc_o_half[0:MERGE_HEADS, :],
                                    )
                                    T.copy(
                                        ws_acc_o[
                                            cid,
                                            slot % 2,
                                            vid * v_block + MERGE_HEADS : vid * v_block
                                            + 2 * MERGE_HEADS,
                                            :,
                                        ],
                                        acc_o_work2,
                                    )
                                    T.set_flag("mte2", "v", 3)
                                    T.wait_flag("mte2", "v", 3)
                                    for h_i in range(MERGE_HEADS):
                                        T.tile.div(
                                            acc_o_work2[h_i, :],
                                            acc_o_work2[h_i, :],
                                            sumexp[MERGE_HEADS + h_i],
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
                                        Output[
                                            t_i,
                                            vid * v_block : vid * v_block + v_block,
                                            :,
                                        ],
                                    )
                                else:
                                    for h_i in range(v_block):
                                        T.barrier_all()
                                        T.tile.div(
                                            acc_o[h_i, :],
                                            acc_o[h_i, :],
                                            sumexp[h_i],
                                        )
                                        T.barrier_all()
                                    T.copy(acc_o, acc_o_half)
                                    T.barrier_all()
                                    T.copy(
                                        acc_o_half,
                                        Output[
                                            t_i,
                                            vid * v_block : vid * v_block + v_block,
                                            :,
                                        ],
                                    )

                                T.tile.ln(lse_ub, sumexp)
                                T.barrier_all()
                                T.tile.add(lse_ub, lse_ub, m_i)
                                T.barrier_all()
                                T.copy(
                                    lse_ub,
                                    LSE_out[
                                        t_i,
                                        vid * v_block : vid * v_block + v_block,
                                    ],
                                )

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
                q_l1 = T.alloc_L1([2, H_per_block, D], dtype)
                kv_lo = T.alloc_L1([2, BI_half, D], dtype)
                kv_hi = T.alloc_L1([2, BI_half, D], dtype)
                p_lo = T.alloc_L1([H_per_block, BI_half], dtype)
                p_hi = T.alloc_L1([H_per_block, BI_half], dtype)
                acc_s_a = T.alloc_L0C([H_per_block, BI_half], accum_dtype)
                acc_s_b = T.alloc_L0C([H_per_block, BI_half], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_work = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_work2 = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_ub = T.alloc_ub([MERGE_HEADS, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                m_i = T.alloc_ub([ub_len], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                sumexp_i_ub = T.alloc_ub([ub_len], accum_dtype)
                sinks_ub = T.alloc_ub([ub_len], accum_dtype)
                lse_ub = T.alloc_ub([ub_len], accum_dtype)
                alpha = T.alloc_ub([2 * ub_len], accum_dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_ub_ = T.alloc_ub([2 * v_block, BI], accum_dtype)
                acc_s_half = T.alloc_ub([v_block, BI], dtype)
                idx_int = T.alloc_ub([BI], indices_dtype)
                idx_float = T.alloc_ub([BI], accum_dtype)
                kv_ub_multi = T.alloc_ub([2 * GATHER_ROWS, D], dtype)
                softmax_tmp = T.alloc_ub([16 * KB], "uint8")
                alpha_exp = T.alloc_ub([ub_len], accum_dtype)
                alpha_brd8 = T.alloc_ub([MERGE_HEADS, 8], accum_dtype)
                mask_ub = T.alloc_ub([2, mask_w], "uint8")
                mask_sel = T.alloc_ub([mask_w], "uint8")
                sumexp_sv = T.alloc_ub([2, ub_len], accum_dtype)
                m_i_sv = T.alloc_ub([2, ub_len], accum_dtype)
                sumexp_rt = T.alloc_ub([ub_len], accum_dtype)
                m_i_rt = T.alloc_ub([ub_len], accum_dtype)

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
                        acc_o: ub_addr["acc_o"],
                        acc_s_ub: ub_addr["acc_s_ub"],
                        acc_s_ub_: ub_addr["acc_s_ub_"],
                        acc_s_half: ub_addr["acc_s_half"],
                        m_i: ub_addr["m_i"],
                        m_i_prev: ub_addr["m_i_prev"],
                        sumexp: ub_addr["sumexp"],
                        sumexp_i_ub: ub_addr["sumexp_i_ub"],
                        sinks_ub: ub_addr["sinks_ub"],
                        lse_ub: ub_addr["lse_ub"],
                        idx_int: ub_addr["idx_int"],
                        idx_float: ub_addr["idx_float"],
                        alpha: ub_addr["alpha"],
                        kv_ub_multi: ub_addr["kv_ub_multi"],
                        mask_ub: ub_addr["mask_ub"],
                        mask_sel: ub_addr["mask_sel"],
                        acc_o_ub: ub_addr["acc_o_ub"],
                        acc_o_half: ub_addr["acc_o_half"],
                        softmax_tmp: ub_addr["kv_ub_multi"],
                        alpha_brd8: ub_addr["kv_ub_multi"] + 16 * KB,
                        alpha_exp: ub_addr["kv_ub_multi"] + 16 * KB + 512,
                        acc_o_work: ub_addr["acc_o"],
                        acc_o_work2: ub_addr["acc_o"] + 32 * KB,
                        sumexp_sv: ub_addr["mask_sel"] + 32,
                        m_i_sv: ub_addr["mask_sel"] + 32 + 256,
                        sumexp_rt: ub_addr["mask_sel"] + 32 + 512,
                        m_i_rt: ub_addr["mask_sel"] + 32 + 640,
                    }
                )

                meta_base = cid * _FA_METADATA_SIZE
                core_enable = Metadata[meta_base + _FA_CORE_ENABLE_INDEX]
                bn2_start = Metadata[meta_base + _FA_BN2_START_INDEX]
                m_start = Metadata[meta_base + _FA_M_START_INDEX]
                bn2_end = Metadata[meta_base + _FA_BN2_END_INDEX]
                m_end = Metadata[meta_base + _FA_M_END_INDEX]
                linear_start = bn2_start * max_seq + m_start
                linear_end = bn2_end * max_seq + m_end
                total_work = batch * max_seq

                with T.Scope("C"):
                    T.set_flag("fix", "m", 0)
                    T.set_flag("fix", "m", 1)
                    for g in T.serial(total_work + 2):
                        pid0 = linear_start + g
                        in_range0 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid0 < linear_end, pid0 >= linear_start, False
                            ),
                            False,
                        )
                        b0_safe = T.if_then_else(in_range0, pid0 // max_seq, 0)
                        s0 = pid0 % max_seq
                        act_q0 = actual_q_len[b0_safe]
                        act_kv0 = actual_kv_len[b0_safe]
                        valid0 = T.if_then_else(in_range0, s0 < act_q0, False)
                        t0 = q_prefix[b0_safe] + s0
                        s_global0 = act_kv0 - act_q0 + s0
                        ori_left0_raw = s_global0 - ori_win_left
                        ori_left0 = T.if_then_else(ori_left0_raw < 0, 0, ori_left0_raw)
                        pid1 = linear_start + g - 1
                        in_range1 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid1 < linear_end, pid1 >= linear_start, False
                            ),
                            False,
                        )
                        b1_safe = T.if_then_else(in_range1, pid1 // max_seq, 0)
                        s1 = pid1 % max_seq
                        act_q1 = actual_q_len[b1_safe]
                        valid1 = T.if_then_else(in_range1, s1 < act_q1, False)
                        pid2 = linear_start + g - 2
                        in_range2 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid2 < linear_end, pid2 >= linear_start, False
                            ),
                            False,
                        )
                        b2_safe = T.if_then_else(in_range2, pid2 // max_seq, 0)
                        s2 = pid2 % max_seq
                        act_q2 = actual_q_len[b2_safe]
                        valid2 = T.if_then_else(in_range2, s2 < act_q2, False)

                        if valid2:
                            T.wait_flag("m", "mte2", 0)

                        if valid0:
                            T.copy(Q[t0, 0:n_heads, 0:D], q_l1[g % 2, :, :])
                            pa = g % 2
                            if ori_block_size >= BI_half:
                                bidx_lo = ori_left0 // ori_block_size
                                rowc_lo = ori_left0 % ori_block_size
                                n_lo = ori_block_size - rowc_lo
                                if n_lo >= BI_half:
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx_lo],
                                            rowc_lo : rowc_lo + BI_half,
                                            0,
                                            :,
                                        ],
                                        kv_lo[pa, 0:BI_half, :],
                                    )
                                else:
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx_lo],
                                            rowc_lo : rowc_lo + n_lo,
                                            0,
                                            :,
                                        ],
                                        kv_lo[pa, 0:n_lo, :],
                                    )
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx_lo + 1],
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
                                            ori_block_table[b0_safe, bidx_hi],
                                            rowc_hi : rowc_hi + BI_half,
                                            0,
                                            :,
                                        ],
                                        kv_hi[pa, 0:BI_half, :],
                                    )
                                else:
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx_hi],
                                            rowc_hi : rowc_hi + n_hi,
                                            0,
                                            :,
                                        ],
                                        kv_hi[pa, 0:n_hi, :],
                                    )
                                    T.copy(
                                        ori_KV[
                                            ori_block_table[b0_safe, bidx_hi + 1],
                                            0 : BI_half - n_hi,
                                            0,
                                            :,
                                        ],
                                        kv_hi[pa, n_hi:BI_half, :],
                                    )
                            else:
                                for gp in range(BI_half // GATHER_ROWS):
                                    g0 = ori_left0 + gp * GATHER_ROWS
                                    bidx = g0 // ori_block_size
                                    rowc = g0 % ori_block_size
                                    if ori_block_size - rowc >= GATHER_ROWS:
                                        T.copy(
                                            ori_KV[
                                                ori_block_table[b0_safe, bidx],
                                                rowc : rowc + GATHER_ROWS,
                                                0,
                                                :,
                                            ],
                                            kv_lo[
                                                pa,
                                                gp * GATHER_ROWS : (gp + 1)
                                                * GATHER_ROWS,
                                                :,
                                            ],
                                        )
                                    else:
                                        n0 = ori_block_size - rowc
                                        T.copy(
                                            ori_KV[
                                                ori_block_table[b0_safe, bidx],
                                                rowc : rowc + n0,
                                                0,
                                                :,
                                            ],
                                            kv_lo[
                                                pa,
                                                gp * GATHER_ROWS : gp * GATHER_ROWS
                                                + n0,
                                                :,
                                            ],
                                        )
                                        T.copy(
                                            ori_KV[
                                                ori_block_table[b0_safe, bidx + 1],
                                                0 : GATHER_ROWS - n0,
                                                0,
                                                :,
                                            ],
                                            kv_lo[
                                                pa,
                                                gp * GATHER_ROWS + n0 : (gp + 1)
                                                * GATHER_ROWS,
                                                :,
                                            ],
                                        )
                                for gp in range(BI_half // GATHER_ROWS):
                                    g0 = ori_left0 + BI_half + gp * GATHER_ROWS
                                    bidx = g0 // ori_block_size
                                    rowc = g0 % ori_block_size
                                    if ori_block_size - rowc >= GATHER_ROWS:
                                        T.copy(
                                            ori_KV[
                                                ori_block_table[b0_safe, bidx],
                                                rowc : rowc + GATHER_ROWS,
                                                0,
                                                :,
                                            ],
                                            kv_hi[
                                                pa,
                                                gp * GATHER_ROWS : (gp + 1)
                                                * GATHER_ROWS,
                                                :,
                                            ],
                                        )
                                    else:
                                        n0 = ori_block_size - rowc
                                        T.copy(
                                            ori_KV[
                                                ori_block_table[b0_safe, bidx],
                                                rowc : rowc + n0,
                                                0,
                                                :,
                                            ],
                                            kv_hi[
                                                pa,
                                                gp * GATHER_ROWS : gp * GATHER_ROWS
                                                + n0,
                                                :,
                                            ],
                                        )
                                        T.copy(
                                            ori_KV[
                                                ori_block_table[b0_safe, bidx + 1],
                                                0 : GATHER_ROWS - n0,
                                                0,
                                                :,
                                            ],
                                            kv_hi[
                                                pa,
                                                gp * GATHER_ROWS + n0 : (gp + 1)
                                                * GATHER_ROWS,
                                                :,
                                            ],
                                        )
                            T.set_flag("mte2", "m", 3)
                            T.wait_flag("mte2", "m", 3)
                            T.wait_flag("fix", "m", 0)
                            T.gemm_v0(
                                q_l1[g % 2, :, :],
                                kv_lo[pa, :, :],
                                acc_s_a,
                                transpose_B=True,
                                init=True,
                            )
                            T.set_flag("m", "fix", 0)
                            T.wait_flag("fix", "m", 1)
                            T.gemm_v0(
                                q_l1[g % 2, :, :],
                                kv_hi[pa, :, :],
                                acc_s_b,
                                transpose_B=True,
                                init=True,
                            )
                            T.set_flag("m", "fix", 2)
                            T.wait_flag("m", "fix", 0)
                            T.copy(
                                acc_s_a,
                                ws_score[cid, pa, 0:H_per_block, 0:BI_half],
                            )
                            T.set_flag("fix", "m", 0)
                            T.wait_flag("m", "fix", 2)
                            T.copy(
                                acc_s_b,
                                ws_score[cid, pa, 0:H_per_block, BI_half:BI],
                            )
                            T.set_flag("fix", "m", 1)
                            T.set_cross_flag("FIX", _FLAG_SCORE_READY)
                        if valid1:
                            pb = (g - 1) % 2
                            T.wait_cross_flag(_FLAG_P_READY)
                            T.copy(
                                ws_p[cid, pb, 0:H_per_block, 0:BI_half],
                                p_lo,
                            )
                            T.copy(
                                ws_p[cid, pb, 0:H_per_block, BI_half:BI],
                                p_hi,
                            )
                            T.set_flag("mte2", "m", 0)
                            T.wait_flag("mte2", "m", 0)
                            T.wait_flag("fix", "m", 0)
                            T.wait_flag("fix", "m", 1)
                            T.gemm_v0(p_lo, kv_lo[pb, :, :], acc_o_l0c, init=True)
                            T.gemm_v0(p_hi, kv_hi[pb, :, :], acc_o_l0c, init=False)
                            T.set_flag("m", "mte2", 0)
                            T.set_flag("m", "fix", 1)
                            T.wait_flag("m", "fix", 1)
                            T.copy(
                                acc_o_l0c,
                                ws_o[cid, pb, 0:H_per_block, 0:D],
                            )
                            T.set_flag("fix", "m", 0)
                            T.set_flag("fix", "m", 1)
                            T.set_cross_flag("FIX", _FLAG_PV_READY)
                    T.wait_flag("fix", "m", 0)
                    T.wait_flag("fix", "m", 1)

                with T.Scope("V"):
                    T.set_flag("v", "mte2", 0)
                    T.set_flag("v", "mte2", 1)
                    for g in T.serial(total_work + 2):
                        pid0 = linear_start + g
                        in_range0 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid0 < linear_end, pid0 >= linear_start, False
                            ),
                            False,
                        )
                        b0_safe = T.if_then_else(in_range0, pid0 // max_seq, 0)
                        s0 = pid0 % max_seq
                        act_q0 = actual_q_len[b0_safe]
                        act_kv0 = actual_kv_len[b0_safe]
                        valid0 = T.if_then_else(in_range0, s0 < act_q0, False)
                        s_global0 = act_kv0 - act_q0 + s0
                        ori_right0 = s_global0
                        ori_left0_raw = s_global0 - ori_win_left
                        ori_left0 = T.if_then_else(ori_left0_raw < 0, 0, ori_left0_raw)
                        pid1 = linear_start + g - 1
                        in_range1 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid1 < linear_end, pid1 >= linear_start, False
                            ),
                            False,
                        )
                        b1_safe = T.if_then_else(in_range1, pid1 // max_seq, 0)
                        s1 = pid1 % max_seq
                        act_q1 = actual_q_len[b1_safe]
                        valid1 = T.if_then_else(in_range1, s1 < act_q1, False)
                        pid2 = linear_start + g - 2
                        in_range2 = T.if_then_else(
                            core_enable != 0,
                            T.if_then_else(
                                pid2 < linear_end, pid2 >= linear_start, False
                            ),
                            False,
                        )
                        b2_safe = T.if_then_else(in_range2, pid2 // max_seq, 0)
                        s2 = pid2 % max_seq
                        act_q2 = actual_q_len[b2_safe]
                        valid2 = T.if_then_else(in_range2, s2 < act_q2, False)
                        t2 = q_prefix[b2_safe] + s2

                        if valid0:
                            pv0 = g % 2
                            chunk_start = ori_left0
                            T.tile.createvecindex(idx_int, chunk_start)
                            T.copy(idx_int, idx_float)
                            T.barrier_all()
                            T.tile.compare(
                                mask_ub[pv0, :],
                                idx_float,
                                T.float32(ori_right0),
                                "LE",
                            )
                            T.barrier_all()
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
                                    (g % 2) * v_block : (g % 2) * v_block + v_block,
                                    :,
                                ],
                            )
                            T.set_flag("mte2", "v", g % 2)

                        if valid1:
                            pv1 = (g - 1) % 2
                            T.copy(
                                Sinks[vid * v_block : vid * v_block + v_block],
                                m_i,
                            )
                            T.tile.fill(sumexp, 1.0)
                            T.barrier_all()
                            T.wait_flag("mte2", "v", pv1)
                            T.copy(mask_ub[pv1, :], mask_sel)
                            for h_i in T.serial(v_block):
                                T.tile.select(
                                    acc_s_ub[h_i, :],
                                    mask_sel,
                                    acc_s_ub_[pv1 * v_block + h_i, :],
                                    -T.infinity(accum_dtype),
                                    "VSEL_TENSOR_SCALAR_MODE",
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
                                alpha_exp,
                                alpha[pv1 * ub_len : pv1 * ub_len + ub_len],
                            )
                            T.copy(sumexp, sumexp_sv[pv1, :])
                            T.copy(m_i, m_i_sv[pv1, :])
                            T.copy(acc_s_ub, acc_s_half)
                            T.set_flag("v", "mte3", 0)
                            T.wait_flag("v", "mte3", 0)
                            T.copy(
                                acc_s_half,
                                ws_p[
                                    cid,
                                    pv1,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                            )
                            T.barrier_all()
                            T.set_cross_flag("MTE3", _FLAG_P_READY)

                        if valid2:
                            pv2 = (g - 2) % 2
                            T.wait_cross_flag(_FLAG_PV_READY)
                            T.barrier_all()
                            T.copy(sumexp_sv[pv2, :], sumexp_rt)
                            T.copy(m_i_sv[pv2, :], m_i_rt)
                            T.barrier_all()
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
                            T.wait_flag("mte2", "v", 2)
                            for h_i in range(MERGE_HEADS):
                                T.tile.div(
                                    acc_o_work[h_i, :],
                                    acc_o_work[h_i, :],
                                    sumexp_rt[h_i],
                                )
                            T.pipe_barrier("v")
                            T.copy(
                                acc_o_work,
                                acc_o_half[0:MERGE_HEADS, :],
                            )
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
                            for h_i in range(MERGE_HEADS):
                                T.tile.div(
                                    acc_o_work2[h_i, :],
                                    acc_o_work2[h_i, :],
                                    sumexp_rt[MERGE_HEADS + h_i],
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
                                Output[
                                    t2,
                                    vid * v_block : vid * v_block + v_block,
                                    :,
                                ],
                            )
                            T.tile.ln(lse_ub, sumexp_rt)
                            T.barrier_all()
                            T.tile.add(lse_ub, lse_ub, m_i_rt)
                            T.barrier_all()
                            T.copy(
                                lse_ub,
                                LSE_out[
                                    t2,
                                    vid * v_block : vid * v_block + v_block,
                                ],
                            )
                    T.wait_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 1)

        return sparse_attn_sharedkv_swa if NI_total == 1 else sparse_attn_sharedkv

    return _make()
