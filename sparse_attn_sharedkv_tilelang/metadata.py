"""Python port of the SparseAttnSharedkvMetadata aicpu kernel.

Mirrors the Ascend C implementation at
``ops-transformer/experimental/attention/sparse_attn_sharedkv_metadata``.
The aicpu kernel runs the cross-core load-balancing scheduler that
assigns work items to AIC cores and records the FlashDecode reduction
tasks for the AIV cores. Its output is a flat INT32 tensor of length
``SAS_META_SIZE = 1024`` whose layout is::

    faMetadata[AIC_CORE_NUM=36][FA_METADATA_SIZE=8]   # 288 int32
    fdMetadata[AIV_CORE_NUM=72][FD_METADATA_SIZE=8]   # 576 int32

The TileLang ``sparse_attn_sharedkv`` kernel does its own ``T.Kernel``
dispatch and does not consume ``metadata`` for scheduling. We
nevertheless port the scheduler so that the TileLang test flow
calls ``metadata`` + ``sharedkv`` together, matching the Ascend C
test flow (``ops-transformer/.../sparse_attn_sharedkv/tests/pytest/
batch/sparse_attn_sharedkv_process.py``).

Algorithm faithfulness: this is a direct port of
``sparse_attn_sharedkv_metadata_aicpu.cpp``. The class layout, method
names, and per-block cost formulas track the C++ source closely so
divergences are easy to audit. ``supportFd`` defaults to ``False``
(the C++ source never flips it), which short-circuits the
block-level reassignment / FD bookkeeping paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


# ---- Constants mirrored from sparse_attn_sharedkv_metadata.h ----

AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
SAS_META_SIZE = 1024

FA_METADATA_SIZE = 8
FD_METADATA_SIZE = 8

FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7

FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6


# BlockType enum
WIN_NORMAL_BLOCK = 0
WIN_TAIL_BLOCK = 1
CMP_NORMAL_BLOCK = 2
CMP_TAIL_BLOCK = 3
BLOCK_MAX_TYPE = 4


class SparseMode:
    DEFAULT_MASK = 0
    ALL_MASK = 1
    LEFT_UP_CAUSAL = 2
    RIGHT_DOWN_CAUSAL = 3
    BAND = 4
    SPARSE_BUTT = 5


FA_TOLERANCE_RATIO = 2
INT64_MAX = (1 << 63) - 1


def _clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def _is_within_tolerance(limit: int, tolerance: int, value: int) -> bool:
    return limit + tolerance >= value


# ---- Internal dataclasses (mirror the aicpu SplitInfo / CostInfo /
# BatchCache / S1GCache / CoreCache / AssignContext / SplitResult). ----


@dataclass
class SplitInfo:
    s1g_base_num: List[int]
    s2_base_num: List[int]
    s1g_tail_size: List[int]
    s2_tail_size: List[int]
    is_kv_seq_all_zero: bool = True

    @classmethod
    def make(cls, batch_size: int) -> "SplitInfo":
        return cls(
            s1g_base_num=[0] * batch_size,
            s2_base_num=[0] * batch_size,
            s1g_tail_size=[0] * batch_size,
            s2_tail_size=[0] * batch_size,
        )


@dataclass
class CostInfo:
    bn2_cost_of_each_batch: List[int]
    bn2_block_of_each_batch: List[int]
    bn2_last_block_cost_of_each_batch: List[int]
    total_block_num: int = 0
    total_cost: int = 0
    max_s1g_cost: int = 0

    @classmethod
    def make(cls, batch_size: int) -> "CostInfo":
        return cls(
            bn2_cost_of_each_batch=[0] * batch_size,
            bn2_block_of_each_batch=[0] * batch_size,
            bn2_last_block_cost_of_each_batch=[0] * batch_size,
        )


@dataclass
class SplitContext:
    split_info: SplitInfo
    cost_info: CostInfo

    @classmethod
    def make(cls, batch_size: int) -> "SplitContext":
        return cls(
            split_info=SplitInfo.make(batch_size),
            cost_info=CostInfo.make(batch_size),
        )


@dataclass
class BatchCache:
    b_idx: int = 0
    s1_size: int = 0
    s2_size: int = 0
    pre_token_left_up: int = 0
    next_token_left_up: int = 0


@dataclass
class S1GCache:
    b_idx: int = 0
    s1g_idx: int = 0
    s2_start: int = 0
    s2_end: int = 0
    win_s2_start: int = 0
    win_s2_end: int = 0
    cmp_s2_start: int = 0
    cmp_s2_end: int = 0
    s1g_cost: int = 0
    s1g_last_block_cost: int = 0
    s1g_block: int = 0
    s1g_normal_block_cost: int = 0
    win_s1g_block: int = 0
    win_s1g_cost: int = 0
    win_s1g_last_block_cost: int = 0
    win_s1g_normal_block_cost: int = 0
    cmp_s1g_block: int = 0
    cmp_s1g_cost: int = 0
    cmp_s1g_last_block_cost: int = 0
    cmp_s1g_normal_block_cost: int = 0
    cmp_s2_tail_size: int = 0
    win_s2_tail_size: int = 0


@dataclass
class CoreCache:
    cost_limit: int = 0
    cost: int = 0
    block: int = 0


@dataclass
class AssignContext:
    cur_b_idx: int = 0
    cur_bn2_idx: int = 0
    cur_s1g_idx: int = 0
    cur_s2_idx: int = 0
    cur_core_idx: int = 0
    unassigned_cost: int = 0
    used_core_num: int = 0
    cur_kv_split_part: int = 1
    cur_fd_data_num: int = 1
    bn2_cost: int = 0
    bn2_block: int = 0
    is_finished: bool = False
    batch_cache: BatchCache = field(default_factory=BatchCache)
    s1g_cache: S1GCache = field(default_factory=S1GCache)
    core_cache: CoreCache = field(default_factory=CoreCache)


@dataclass
class FlashDecodeResult:
    fd_used_vec_num: int = 0
    fd_bn2_idx: List[int] = field(default_factory=list)
    fd_m_idx: List[int] = field(default_factory=list)
    fd_workspace_idx: List[int] = field(default_factory=list)
    fd_s2_split_num: List[int] = field(default_factory=list)
    fd_m_size: List[int] = field(default_factory=list)
    fd_idx: List[int] = field(default_factory=list)
    fd_m_start: List[int] = field(default_factory=list)
    fd_m_num: List[int] = field(default_factory=list)

    @classmethod
    def make(cls, aic_num: int, aiv_num: int) -> "FlashDecodeResult":
        return cls(
            fd_bn2_idx=[0] * aic_num,
            fd_m_idx=[0] * aic_num,
            fd_workspace_idx=[0] * aic_num,
            fd_s2_split_num=[0] * aic_num,
            fd_m_size=[0] * aic_num,
            fd_idx=[0] * aiv_num,
            fd_m_start=[0] * aiv_num,
            fd_m_num=[0] * aiv_num,
        )


@dataclass
class SplitResult:
    used_core_num: int = 0
    bn2_end: List[int] = field(default_factory=list)
    g_s1_end: List[int] = field(default_factory=list)
    s2_end: List[int] = field(default_factory=list)
    first_fd_data_workspace_idx: List[int] = field(default_factory=list)
    max_cost: int = 0
    num_of_fd_head: int = 0
    max_s2_split_num: int = 0
    fd_res: FlashDecodeResult = field(default_factory=FlashDecodeResult)

    @classmethod
    def make(cls, aic_num: int, aiv_num: int) -> "SplitResult":
        return cls(
            bn2_end=[0] * aic_num,
            g_s1_end=[0] * aic_num,
            s2_end=[0] * aic_num,
            first_fd_data_workspace_idx=[0] * aic_num,
            fd_res=FlashDecodeResult.make(aic_num, aiv_num),
        )


# ---- The scheduler proper. ----


class _MetadataScheduler:
    """Direct port of ``SparseAttnSharedkvMetadataCpuKernel``."""

    def __init__(
        self,
        *,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
        cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_kv: Optional[torch.Tensor] = None,
        batch_size: int = 0,
        max_seqlen_q: int = 0,
        max_seqlen_kv: int = 0,
        ori_topk: int = 0,
        cmp_topk: int = 0,
        cmp_ratio: int = -1,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = "BSND",
        layout_kv: str = "PA_ND",
        has_ori_kv: bool = True,
        has_cmp_kv: bool = True,
        aic_core_num: int = 24,
        aiv_core_num: int = 48,
    ):
        self.queryHeadNum = num_heads_q
        self.kvHeadNum = num_heads_kv
        self.headDim = head_dim
        self.cu_seqlens_q = self._as_int_list(cu_seqlens_q)
        self.cu_seqlens_ori_kv = self._as_int_list(cu_seqlens_ori_kv)
        self.cu_seqlens_cmp_kv = self._as_int_list(cu_seqlens_cmp_kv)
        self.seqused_q = self._as_int_list(seqused_q)
        self.seqused_kv = self._as_int_list(seqused_kv)
        self.batchSize = batch_size
        self.querySeqSize = max_seqlen_q
        self.kvSeqSize = max_seqlen_kv
        self.oriTopK = ori_topk
        self.cmpTopK = cmp_topk
        self.cmpRatio = cmp_ratio
        self.oriMaskMode = ori_mask_mode
        self.cmpMaskMode = cmp_mask_mode
        self.winLeft = ori_win_left
        self.winRight = ori_win_right
        self.layoutQuery = layout_q
        self.layoutKv = layout_kv
        self.hasOriKv = has_ori_kv
        self.hasCmpKv = has_cmp_kv
        self.aicCoreNum = aic_core_num
        self.aivCoreNum = aiv_core_num

        # Derived attributes filled by ParamsInit().
        self.preToken = 0
        self.nextToken = 0
        self.groupSize = 0
        self.mBaseSize = 0
        self.s2BaseSize = 512
        self.isS1G = True
        self.isCFA = False
        self.isSCFA = False
        # The aicpu source declares supportFd and never flips it; keep
        # the same default so AssignByBlock / FD bookkeeping are skipped.
        self.supportFd = False
        self.attentionMode = 1
        self.typeCost: List[List[int]] = [
            [0] * BLOCK_MAX_TYPE for _ in range(BLOCK_MAX_TYPE)
        ]

    @staticmethod
    def _as_int_list(t) -> List[int]:
        if t is None:
            return []
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                return []
            return t.detach().cpu().to(torch.int64).tolist()
        if isinstance(t, (list, tuple)):
            return [int(x) for x in t]
        return []

    # ---- Prepare / param checks ----

    def _get_query_batch_size(self) -> int:
        if self.seqused_q:
            return len(self.seqused_q)
        if self.layoutQuery == "TND" and self.cu_seqlens_q:
            return len(self.cu_seqlens_q) - 1
        return self.batchSize

    def _get_kv_batch_size(self) -> int:
        if self.seqused_kv:
            return len(self.seqused_kv)
        if self.layoutKv == "TND" and self.cu_seqlens_ori_kv:
            return len(self.cu_seqlens_ori_kv) - 1
        return self.batchSize

    def _params_init(self) -> None:
        self.batchSize = self._get_query_batch_size()

        mode = self.oriMaskMode
        if mode == SparseMode.DEFAULT_MASK:
            self.preToken = INT64_MAX
            self.nextToken = INT64_MAX
            self.attentionMode = 0
        elif mode == SparseMode.RIGHT_DOWN_CAUSAL:
            self.preToken = INT64_MAX
            self.nextToken = 0
            self.attentionMode = 1
        else:  # BAND (mode == 4) and anything else
            self.preToken = self.winLeft if self.winLeft > -1 else INT64_MAX
            self.nextToken = 0
            self.attentionMode = 1

        self.isS1G = self.layoutQuery in ("BSND", "BSH", "TND")
        if self.kvHeadNum == 0:
            self.groupSize = 0
        else:
            self.groupSize = self.queryHeadNum // self.kvHeadNum

        if self.hasCmpKv:
            if self.cmpTopK > 0:
                self.isSCFA = True
            else:
                self.isCFA = True

        self.mBaseSize = self.groupSize
        self.s2BaseSize = 512

    # ---- Utilities ----

    def _get_s1_seq_size(self, b_idx: int) -> int:
        if self.seqused_q:
            return int(self.seqused_q[b_idx])
        if self.layoutQuery == "TND" and self.cu_seqlens_q:
            return int(self.cu_seqlens_q[b_idx + 1] - self.cu_seqlens_q[b_idx])
        return int(self.querySeqSize)

    def _get_s2_seq_size(self, b_idx: int) -> int:
        if self.seqused_kv:
            return int(self.seqused_kv[b_idx])
        if self.layoutKv == "TND" and self.cu_seqlens_ori_kv:
            return int(
                self.cu_seqlens_ori_kv[b_idx + 1] - self.cu_seqlens_ori_kv[b_idx]
            )
        return int(self.kvSeqSize)

    def _calc_pre_token_left_up(self, s1_size: int, s2_size: int) -> int:
        if self.oriMaskMode == SparseMode.BAND:
            return s1_size - s2_size + self.preToken
        return self.preToken

    def _calc_next_token_left_up(self, s1_size: int, s2_size: int) -> int:
        m = self.oriMaskMode
        if m in (
            SparseMode.DEFAULT_MASK,
            SparseMode.ALL_MASK,
            SparseMode.LEFT_UP_CAUSAL,
        ):
            return self.nextToken
        if m == SparseMode.RIGHT_DOWN_CAUSAL:
            return s2_size - s1_size
        if m == SparseMode.BAND:
            return s2_size - s1_size + self.nextToken
        return self.nextToken

    @staticmethod
    def _win_calc_cost(basic_m: int, basic_s2: int) -> int:
        win_align_m = (basic_m + 15) >> 4  # ceil-div by 16
        win_align_s2 = (basic_s2 + 63) >> 6  # ceil-div by 64
        return 6 * win_align_m + 10 * win_align_s2

    @staticmethod
    def _cmp_calc_cost(basic_m: int, basic_s2: int) -> int:
        cmp_align_m = (basic_m + 15) >> 4
        cmp_align_s2 = (basic_s2 + 63) >> 6
        return 6 * cmp_align_m + 10 * cmp_align_s2

    def _calc_cost_table(
        self,
        s1_normal: int,
        s2_normal: int,
        s1g_tail: int,
        win_s2_tail: int,
        cmp_s2_tail: int,
    ) -> None:
        tc = self.typeCost
        tc[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] = self._win_calc_cost(
            s1_normal, s2_normal
        )
        tc[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] = (
            0 if s1g_tail == 0 else self._win_calc_cost(s1g_tail, s2_normal)
        )
        tc[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] = (
            0 if win_s2_tail == 0 else self._win_calc_cost(s1_normal, win_s2_tail)
        )
        tc[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] = (
            0
            if (s1g_tail == 0 or win_s2_tail == 0)
            else self._win_calc_cost(s1g_tail, win_s2_tail)
        )
        if self.hasCmpKv:
            tc[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] = self._cmp_calc_cost(
                s1_normal, s2_normal
            )
            tc[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] = (
                0 if s1g_tail == 0 else self._cmp_calc_cost(s1g_tail, s2_normal)
            )
            tc[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] = (
                0 if cmp_s2_tail == 0 else self._cmp_calc_cost(s1_normal, cmp_s2_tail)
            )
            tc[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] = (
                0
                if (s1g_tail == 0 or cmp_s2_tail == 0)
                else self._cmp_calc_cost(s1g_tail, cmp_s2_tail)
            )

    def _calc_s2_token_range(self, s1g_idx: int, bc: BatchCache) -> Tuple[int, int]:
        if bc.s1_size == 0 or bc.s2_size == 0:
            return 0, 0
        if not self.attentionMode:
            return 0, bc.s2_size - 1

        s1g_first = s1g_idx * self.mBaseSize
        s1g_last = min(s1g_first + self.mBaseSize, bc.s1_size * self.groupSize) - 1

        if self.isS1G:
            s1_first = s1g_first // self.groupSize
            s1_last = s1g_last // self.groupSize
        else:
            if s1g_first // bc.s1_size == s1g_last // bc.s1_size:
                s1_first = s1g_first % bc.s1_size
                s1_last = s1g_last % bc.s1_size
            else:
                s1_first = 0
                s1_last = bc.s1_size

        s2_first = s1_first - bc.pre_token_left_up
        s2_last = s1_last + bc.next_token_left_up
        return s2_first, s2_last

    # ---- Cache calculation ----

    def _calc_batch_cache(self, b_idx: int, ctx: SplitContext, bc: BatchCache) -> None:
        bc.b_idx = b_idx
        bc.s1_size = self._get_s1_seq_size(b_idx)
        bc.s2_size = self._get_s2_seq_size(b_idx)
        bc.pre_token_left_up = self._calc_pre_token_left_up(bc.s1_size, bc.s2_size)
        bc.next_token_left_up = self._calc_next_token_left_up(bc.s1_size, bc.s2_size)

    def _calc_block_range_and_tail_size(
        self, ori_s2_range: Tuple[int, int], bc: BatchCache, sc: S1GCache
    ) -> None:
        ori_s2_first, ori_s2_last = ori_s2_range
        if ori_s2_first >= bc.s2_size or ori_s2_last < 0 or ori_s2_last < ori_s2_first:
            ori_s2_first = 0
            ori_s2_last = 0
            sc.win_s2_start = 0
            sc.win_s2_end = 0
            sc.win_s2_tail_size = 0
        else:
            ori_s2_first = _clip(ori_s2_first, 0, bc.s2_size - 1)
            ori_s2_last = _clip(ori_s2_last, 0, bc.s2_size - 1)
            sc.win_s2_start = 0
            sc.win_s2_end = (ori_s2_last - ori_s2_first) // self.s2BaseSize + 1
            sc.win_s2_tail_size = (ori_s2_last - ori_s2_first + 1) % self.s2BaseSize

        sc.cmp_s2_start = sc.win_s2_end
        if self.hasCmpKv and self.cmpRatio > 0:
            cmp_s2_last_token_size = (ori_s2_last + 1) // self.cmpRatio
        else:
            cmp_s2_last_token_size = 0
        act_cmp_s2_last_token_size = 0
        if self.isCFA:
            act_cmp_s2_last_token_size = cmp_s2_last_token_size
        elif self.isSCFA:
            act_cmp_s2_last_token_size = min(cmp_s2_last_token_size, self.cmpTopK)
        if act_cmp_s2_last_token_size == 0:
            sc.cmp_s2_end = sc.cmp_s2_start
        else:
            sc.cmp_s2_end = (
                sc.cmp_s2_start
                + (act_cmp_s2_last_token_size - 1) // self.s2BaseSize
                + 1
            )
        sc.cmp_s2_tail_size = act_cmp_s2_last_token_size % self.s2BaseSize

    def _calc_win_s1g_cache(self, sc: S1GCache, si: SplitInfo) -> None:
        tc = self.typeCost
        if sc.win_s2_start >= sc.win_s2_end:
            sc.win_s1g_block = 0
            sc.win_s1g_cost = 0
            sc.win_s1g_last_block_cost = 0
            sc.win_s1g_normal_block_cost = 0
            return
        sc.win_s1g_block = sc.win_s2_end - sc.win_s2_start
        cur_win_tail = 1 if sc.win_s2_tail_size != 0 else 0
        cur_win_normal = sc.win_s1g_block - cur_win_tail
        if (
            sc.s1g_idx == (si.s1g_base_num[sc.b_idx] - 1)
            and si.s1g_tail_size[sc.b_idx] != 0
        ):
            sc.win_s1g_cost = (
                tc[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] * cur_win_normal
                + tc[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] * cur_win_tail
            )
            sc.win_s1g_last_block_cost = (
                tc[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK]
                if cur_win_tail > 0
                else tc[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK]
            )
            sc.win_s1g_normal_block_cost = tc[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK]
        else:
            sc.win_s1g_cost = (
                tc[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] * cur_win_normal
                + tc[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] * cur_win_tail
            )
            sc.win_s1g_last_block_cost = (
                tc[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK]
                if cur_win_tail > 0
                else tc[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK]
            )
            sc.win_s1g_normal_block_cost = tc[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK]

    def _calc_cmp_s1g_cache(self, sc: S1GCache, si: SplitInfo) -> None:
        tc = self.typeCost
        if sc.cmp_s2_start >= sc.cmp_s2_end:
            sc.cmp_s1g_block = 0
            sc.cmp_s1g_cost = 0
            sc.cmp_s1g_last_block_cost = 0
            sc.cmp_s1g_normal_block_cost = 0
            return
        sc.cmp_s1g_block = sc.cmp_s2_end - sc.cmp_s2_start
        cur_cmp_tail = 1 if sc.cmp_s2_tail_size != 0 else 0
        cur_cmp_normal = sc.cmp_s1g_block - cur_cmp_tail
        if (
            sc.s1g_idx == (si.s1g_base_num[sc.b_idx] - 1)
            and si.s1g_tail_size[sc.b_idx] != 0
        ):
            sc.cmp_s1g_cost = (
                tc[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] * cur_cmp_normal
                + tc[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] * cur_cmp_tail
            )
            sc.cmp_s1g_last_block_cost = (
                tc[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK]
                if cur_cmp_tail > 0
                else tc[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK]
            )
            sc.cmp_s1g_normal_block_cost = tc[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK]
        else:
            sc.cmp_s1g_cost = (
                tc[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] * cur_cmp_normal
                + tc[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] * cur_cmp_tail
            )
            sc.cmp_s1g_last_block_cost = (
                tc[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK]
                if cur_cmp_tail > 0
                else tc[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK]
            )
            sc.cmp_s1g_normal_block_cost = tc[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK]

    def _gather_win_and_cmp(self, sc: S1GCache) -> None:
        sc.s2_start = sc.win_s2_start if sc.win_s1g_block > 0 else sc.cmp_s2_start
        if sc.cmp_s1g_block > 0:
            sc.s1g_last_block_cost = sc.cmp_s1g_last_block_cost
            sc.s2_end = sc.cmp_s2_end
        else:
            sc.s1g_last_block_cost = sc.win_s1g_last_block_cost
            sc.s2_end = sc.win_s2_end
        sc.s1g_block = sc.win_s1g_block + sc.cmp_s1g_block
        sc.s1g_cost = sc.win_s1g_cost + sc.cmp_s1g_cost

    def _calc_s1g_cache(
        self,
        s1g_idx: int,
        ctx: SplitContext,
        bc: BatchCache,
        sc: S1GCache,
    ) -> None:
        si = ctx.split_info
        if si.s1g_base_num[bc.b_idx] == 0:
            sc.s1g_cost = 0
            sc.s1g_last_block_cost = 0
            sc.win_s1g_normal_block_cost = 0
            sc.win_s1g_last_block_cost = 0
            sc.cmp_s1g_normal_block_cost = 0
            sc.cmp_s1g_last_block_cost = 0
            sc.s1g_block = 0
            sc.s2_start = 0
            sc.cmp_s2_start = 0
            sc.s2_end = 0
            return
        sc.b_idx = bc.b_idx
        sc.s1g_idx = s1g_idx
        ori_s2_range = self._calc_s2_token_range(s1g_idx, bc)
        self._calc_block_range_and_tail_size(ori_s2_range, bc, sc)
        self._calc_cost_table(
            self.mBaseSize,
            self.s2BaseSize,
            si.s1g_tail_size[sc.b_idx],
            sc.win_s2_tail_size,
            sc.cmp_s2_tail_size,
        )
        self._calc_win_s1g_cache(sc, si)
        self._calc_cmp_s1g_cache(sc, si)
        self._gather_win_and_cmp(sc)

    # ---- Preprocess ----

    def _calc_split_info(self, ctx: SplitContext) -> None:
        si = ctx.split_info
        for b_idx in range(self.batchSize):
            s1_size = self._get_s1_seq_size(b_idx)
            s2_size = self._get_s2_seq_size(b_idx)
            si.s1g_base_num[b_idx] = (
                s1_size * self.groupSize + self.mBaseSize - 1
            ) // self.mBaseSize
            si.s1g_tail_size[b_idx] = (s1_size * self.groupSize) % self.mBaseSize
            si.s2_base_num[b_idx] = (s2_size + self.s2BaseSize - 1) // self.s2BaseSize
            si.s2_tail_size[b_idx] = s2_size % self.s2BaseSize
            if si.s1g_base_num[b_idx] != 0 and si.s2_base_num[b_idx] != 0:
                si.is_kv_seq_all_zero = False

    def _calc_batch_cost(self, b_idx: int, ctx: SplitContext, ci: CostInfo) -> None:
        si = ctx.split_info
        ci.bn2_cost_of_each_batch[b_idx] = 0
        ci.bn2_block_of_each_batch[b_idx] = 0
        ci.bn2_last_block_cost_of_each_batch[b_idx] = 0
        if self._get_s1_seq_size(b_idx) == 0 or self._get_s2_seq_size(b_idx) == 0:
            return
        bc = BatchCache()
        sc = S1GCache()
        self._calc_batch_cache(b_idx, ctx, bc)
        for s1g_idx in range(si.s1g_base_num[b_idx]):
            self._calc_s1g_cache(s1g_idx, ctx, bc, sc)
            ci.bn2_cost_of_each_batch[b_idx] += sc.s1g_cost
            ci.bn2_block_of_each_batch[b_idx] += sc.s1g_block
            if sc.s1g_cost > ci.max_s1g_cost:
                ci.max_s1g_cost = sc.s1g_cost
            if sc.s1g_block > 0:
                ci.bn2_last_block_cost_of_each_batch[b_idx] = sc.s1g_last_block_cost

    def _calc_cost_info(self, ctx: SplitContext) -> None:
        si = ctx.split_info
        ci = ctx.cost_info
        if si.is_kv_seq_all_zero:
            ci.total_cost = 0
            ci.total_block_num = 0
            return
        for b_idx in range(self.batchSize):
            self._calc_batch_cost(b_idx, ctx, ci)
            ci.total_cost += ci.bn2_cost_of_each_batch[b_idx] * self.kvHeadNum
            ci.total_block_num += ci.bn2_block_of_each_batch[b_idx] * self.kvHeadNum

    # ---- Assign ----

    def _assign_by_batch(self, ctx: SplitContext, ac: AssignContext) -> None:
        if ac.is_finished:
            return
        ci = ctx.cost_info
        while ac.bn2_cost == 0 or _is_within_tolerance(
            ac.core_cache.cost_limit,
            ci.bn2_last_block_cost_of_each_batch[ac.cur_b_idx] // FA_TOLERANCE_RATIO,
            ac.core_cache.cost + ac.bn2_cost,
        ):
            ac.core_cache.cost += ac.bn2_cost
            ac.core_cache.block += ac.bn2_block
            ac.cur_bn2_idx += 1

            if ac.cur_bn2_idx == self.batchSize * self.kvHeadNum:
                ac.cur_s1g_idx = 0
                ac.cur_s2_idx = 0
                ac.is_finished = True
                return

            if ac.cur_bn2_idx // self.kvHeadNum != ac.cur_b_idx:
                ac.cur_b_idx = ac.cur_bn2_idx // self.kvHeadNum
                self._calc_batch_cache(ac.cur_b_idx, ctx, ac.batch_cache)

            ac.bn2_cost = ci.bn2_cost_of_each_batch[ac.cur_b_idx]
            ac.bn2_block = ci.bn2_block_of_each_batch[ac.cur_b_idx]
            ac.cur_s1g_idx = 0
            self._calc_s1g_cache(ac.cur_s1g_idx, ctx, ac.batch_cache, ac.s1g_cache)
            ac.cur_s2_idx = ac.s1g_cache.s2_start

    def _assign_by_row(self, ctx: SplitContext, ac: AssignContext) -> None:
        if ac.is_finished:
            return
        while _is_within_tolerance(
            ac.core_cache.cost_limit,
            ac.s1g_cache.s1g_last_block_cost // FA_TOLERANCE_RATIO,
            ac.core_cache.cost + ac.s1g_cache.s1g_cost,
        ):
            ac.core_cache.cost += ac.s1g_cache.s1g_cost
            ac.core_cache.block += ac.s1g_cache.s1g_block
            ac.bn2_cost = max(0, ac.bn2_cost - ac.s1g_cache.s1g_cost)
            ac.bn2_block = max(0, ac.bn2_block - ac.s1g_cache.s1g_block)
            while True:
                ac.cur_s1g_idx += 1
                self._calc_s1g_cache(ac.cur_s1g_idx, ctx, ac.batch_cache, ac.s1g_cache)
                if ac.s1g_cache.s1g_block != 0:
                    break
            ac.cur_s2_idx = ac.s1g_cache.s2_start

    def _calc_cur_block_cost(self, ac: AssignContext) -> int:
        if ac.cur_s2_idx < ac.s1g_cache.cmp_s2_start:
            cur = ac.s1g_cache.win_s1g_normal_block_cost
            if ac.cur_s2_idx == (ac.s1g_cache.cmp_s2_start - 1):
                cur = ac.s1g_cache.win_s1g_last_block_cost
        else:
            cur = ac.s1g_cache.cmp_s1g_normal_block_cost
            if ac.cur_s2_idx == (ac.s1g_cache.s2_end - 1):
                cur = ac.s1g_cache.cmp_s1g_last_block_cost
        return cur

    def _assign_by_block(self, ctx: SplitContext, ac: AssignContext) -> None:
        if ac.is_finished or not self.supportFd:
            return
        cur_cost = self._calc_cur_block_cost(ac)
        while _is_within_tolerance(
            ac.core_cache.cost_limit,
            cur_cost // FA_TOLERANCE_RATIO,
            ac.core_cache.cost + cur_cost,
        ):
            ac.core_cache.cost += cur_cost
            ac.core_cache.block += 1
            ac.cur_s2_idx += 1
            ac.bn2_cost -= cur_cost
            ac.s1g_cache.s1g_cost -= cur_cost
            ac.bn2_block -= 1
            ac.s1g_cache.s1g_block -= 1
            cur_cost = self._calc_cur_block_cost(ac)

    def _is_need_record_fd(self, ac: AssignContext, sr: SplitResult) -> bool:
        if ac.cur_core_idx == 0:
            return False
        if ac.cur_kv_split_part <= 1:
            return False
        if (
            ac.cur_bn2_idx == sr.bn2_end[ac.cur_core_idx - 1]
            and ac.cur_s1g_idx == sr.g_s1_end[ac.cur_core_idx - 1]
        ):
            return False
        return True

    def _record_fd_info(
        self, ctx: SplitContext, ac: AssignContext, sr: SplitResult
    ) -> None:
        si = ctx.split_info
        split_b_idx = sr.bn2_end[ac.cur_core_idx - 1] // self.kvHeadNum
        split_s1g_idx = sr.g_s1_end[ac.cur_core_idx - 1]
        s1_size = self._get_s1_seq_size(split_b_idx)
        cur_fd_s1g_size = (
            s1_size * self.groupSize - split_s1g_idx * self.mBaseSize
            if split_s1g_idx == si.s1g_base_num[split_b_idx] - 1
            else self.mBaseSize
        )
        sr.max_s2_split_num = max(sr.max_s2_split_num, ac.cur_kv_split_part)
        sr.fd_res.fd_bn2_idx[sr.num_of_fd_head] = sr.bn2_end[ac.cur_core_idx - 1]
        sr.fd_res.fd_m_idx[sr.num_of_fd_head] = sr.g_s1_end[ac.cur_core_idx - 1]
        sr.fd_res.fd_workspace_idx[sr.num_of_fd_head] = (
            ac.cur_fd_data_num - ac.cur_kv_split_part
        )
        sr.fd_res.fd_s2_split_num[sr.num_of_fd_head] = ac.cur_kv_split_part
        sr.fd_res.fd_m_size[sr.num_of_fd_head] = cur_fd_s1g_size
        sr.num_of_fd_head += 1

    def _update_cursor(self, ctx: SplitContext, ac: AssignContext) -> None:
        si = ctx.split_info
        ci = ctx.cost_info
        update_s1g = False
        update_batch = False
        if ac.cur_s2_idx >= ac.s1g_cache.s2_end:
            ac.cur_s2_idx = 0
            ac.cur_s1g_idx += 1
            update_s1g = True
        if ac.cur_s1g_idx >= si.s1g_base_num[ac.cur_b_idx]:
            ac.cur_s1g_idx = 0
            ac.cur_bn2_idx += 1
        if ac.cur_bn2_idx == self.batchSize * self.kvHeadNum:
            ac.cur_s1g_idx = 0
            ac.cur_s2_idx = 0
            ac.is_finished = True
            return
        if ac.cur_bn2_idx // self.kvHeadNum != ac.cur_b_idx:
            ac.cur_b_idx = ac.cur_bn2_idx // self.kvHeadNum
            ac.cur_s1g_idx = 0
            update_batch = True
            update_s1g = True
        if update_batch:
            self._calc_batch_cache(ac.cur_b_idx, ctx, ac.batch_cache)
            ac.bn2_cost = ci.bn2_cost_of_each_batch[ac.cur_b_idx]
            ac.bn2_block = ci.bn2_block_of_each_batch[ac.cur_b_idx]
        if update_s1g:
            self._calc_s1g_cache(ac.cur_s1g_idx, ctx, ac.batch_cache, ac.s1g_cache)
            ac.cur_s2_idx = ac.s1g_cache.win_s2_start if self.supportFd else 0

    def _force_assign(self, ctx: SplitContext, ac: AssignContext) -> None:
        if ac.is_finished:
            return
        cur_cost = self._calc_cur_block_cost(ac)
        ac.core_cache.cost += cur_cost
        ac.core_cache.block += 1
        ac.cur_s2_idx += 1
        ac.bn2_cost -= cur_cost
        ac.bn2_block -= 1
        ac.s1g_cache.s1g_cost -= cur_cost
        ac.s1g_cache.s1g_block -= 1
        self._update_cursor(ctx, ac)

    def _assign_blocks_to_core(
        self, ctx: SplitContext, ac: AssignContext, sr: SplitResult
    ) -> None:
        ci = ctx.cost_info
        avg_cost = ac.unassigned_cost // max(1, self.aicCoreNum - ac.cur_core_idx)
        ac.core_cache = CoreCache()
        if not self.supportFd:
            ac.core_cache.cost_limit = max(avg_cost, ci.max_s1g_cost)
        else:
            ac.core_cache.cost_limit = avg_cost
        self._assign_by_batch(ctx, ac)
        self._assign_by_row(ctx, ac)
        self._assign_by_block(ctx, ac)
        if ac.core_cache.block == 0 and self.supportFd:
            self._force_assign(ctx, ac)
        sr.bn2_end[ac.cur_core_idx] = ac.cur_bn2_idx
        sr.g_s1_end[ac.cur_core_idx] = ac.cur_s1g_idx
        sr.s2_end[ac.cur_core_idx] = ac.cur_s2_idx
        sr.max_cost = max(sr.max_cost, ac.core_cache.cost)
        ac.unassigned_cost -= ac.core_cache.cost
        if self._is_need_record_fd(ac, sr):
            self._record_fd_info(ctx, ac, sr)
            ac.cur_kv_split_part = 1
        if ac.s1g_cache.s2_start < ac.cur_s2_idx <= ac.s1g_cache.s2_end:
            ac.cur_kv_split_part += 1

    def _calc_split_plan(
        self, cost_limit: int, ctx: SplitContext, sr: SplitResult
    ) -> None:
        ci = ctx.cost_info
        if self.aicCoreNum == 0:
            return
        sr.max_cost = 0
        sr.used_core_num = 0
        ac = AssignContext()
        ac.cur_b_idx = 0
        ac.cur_s1g_idx = 0
        ac.unassigned_cost = ci.total_cost
        ac.bn2_cost = ci.bn2_cost_of_each_batch[ac.cur_b_idx]
        ac.bn2_block = ci.bn2_block_of_each_batch[ac.cur_b_idx]
        self._calc_batch_cache(ac.cur_b_idx, ctx, ac.batch_cache)
        self._calc_s1g_cache(ac.cur_s1g_idx, ctx, ac.batch_cache, ac.s1g_cache)
        ac.cur_s2_idx = ac.s1g_cache.s2_start

        for i in range(self.aicCoreNum):
            if sr.max_cost > cost_limit:
                return
            if ac.is_finished or ac.unassigned_cost <= 0:
                break
            ac.cur_core_idx = i
            self._assign_blocks_to_core(ctx, ac, sr)
        sr.used_core_num = ac.cur_core_idx + 1

    def _split_fd(self, sr: SplitResult) -> None:
        total_fd_load = 0
        for i in range(sr.num_of_fd_head):
            total_fd_load += sr.fd_res.fd_s2_split_num[i] * sr.fd_res.fd_m_size[i]
        if self.aivCoreNum == 0:
            return
        average_load = total_fd_load // self.aivCoreNum
        if average_load == 0:
            return
        cur_core_index = 0
        for i in range(sr.num_of_fd_head):
            cur_fd_vec_num = (
                sr.fd_res.fd_s2_split_num[i] * sr.fd_res.fd_m_size[i] // average_load
            )
            if cur_fd_vec_num == 0:
                continue
            cur_ave_m_size = sr.fd_res.fd_m_size[i] // cur_fd_vec_num
            for vid in range(cur_fd_vec_num):
                if cur_core_index >= len(sr.fd_res.fd_idx):
                    break
                sr.fd_res.fd_idx[cur_core_index] = i
                sr.fd_res.fd_m_start[cur_core_index] = vid * cur_ave_m_size
                if vid < cur_fd_vec_num - 1:
                    sr.fd_res.fd_m_num[cur_core_index] = cur_ave_m_size
                else:
                    sr.fd_res.fd_m_num[cur_core_index] = (
                        sr.fd_res.fd_m_size[i] - vid * cur_ave_m_size
                    )
                cur_core_index += 1
        sr.fd_res.fd_used_vec_num = cur_core_index

    def _balance_schedule(self, sr: SplitResult) -> bool:
        ctx = SplitContext.make(self.batchSize)
        self._calc_split_info(ctx)
        if ctx.split_info.is_kv_seq_all_zero:
            sr.used_core_num = 1
            sr.bn2_end[0] = self.batchSize * self.kvHeadNum
            sr.g_s1_end[0] = 0
            sr.s2_end[0] = 0
            return True
        self._calc_cost_info(ctx)
        sr.max_cost = INT64_MAX
        sr.used_core_num = 1
        self._calc_split_plan(sr.max_cost, ctx, sr)
        if sr.num_of_fd_head > 0:
            self._split_fd(sr)
        sr.used_core_num = max(sr.used_core_num, 1)
        return True

    def _gen_metadata(self, sr: SplitResult) -> torch.Tensor:
        # Output buffer: SAS_META_SIZE int32 entries representing the
        # SasMetaData struct (faMetadata first, then fdMetadata).
        out = torch.zeros(SAS_META_SIZE, dtype=torch.int32)
        fa_base = 0
        for i in range(AIC_CORE_NUM):
            row = fa_base + i * FA_METADATA_SIZE
            if i >= sr.used_core_num or i >= self.aicCoreNum:
                out[row + FA_CORE_ENABLE_INDEX] = 0
                continue
            out[row + FA_CORE_ENABLE_INDEX] = 1
            out[row + FA_BN2_START_INDEX] = 0 if i == 0 else sr.bn2_end[i - 1]
            out[row + FA_M_START_INDEX] = 0 if i == 0 else sr.g_s1_end[i - 1]
            out[row + FA_S2_START_INDEX] = 0 if i == 0 else sr.s2_end[i - 1]
            out[row + FA_BN2_END_INDEX] = sr.bn2_end[i]
            out[row + FA_M_END_INDEX] = sr.g_s1_end[i]
            out[row + FA_S2_END_INDEX] = sr.s2_end[i]
            out[row + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = (
                sr.first_fd_data_workspace_idx[i]
            )

        fd_base = AIC_CORE_NUM * FA_METADATA_SIZE
        for i in range(AIV_CORE_NUM):
            row = fd_base + i * FD_METADATA_SIZE
            if i >= sr.fd_res.fd_used_vec_num or i >= self.aivCoreNum:
                out[row + FD_CORE_ENABLE_INDEX] = 0
                continue
            out[row + FD_CORE_ENABLE_INDEX] = 1
            cur_fd_idx = sr.fd_res.fd_idx[i]
            out[row + FD_BN2_IDX_INDEX] = sr.fd_res.fd_bn2_idx[cur_fd_idx]
            out[row + FD_M_IDX_INDEX] = sr.fd_res.fd_m_idx[cur_fd_idx]
            out[row + FD_WORKSPACE_IDX_INDEX] = sr.fd_res.fd_workspace_idx[cur_fd_idx]
            out[row + FD_WORKSPACE_NUM_INDEX] = sr.fd_res.fd_s2_split_num[cur_fd_idx]
            out[row + FD_M_START_INDEX] = sr.fd_res.fd_m_start[i]
            out[row + FD_M_NUM_INDEX] = sr.fd_res.fd_m_num[i]
        return out

    def run(self) -> torch.Tensor:
        self._params_init()
        sr = SplitResult.make(self.aicCoreNum, self.aivCoreNum)
        self._balance_schedule(sr)
        return self._gen_metadata(sr)


def sparse_attn_sharedkv_metadata(
    *,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: int = 0,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    ori_topk: int = 0,
    cmp_topk: int = 0,
    cmp_ratio: int = -1,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "PA_ND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
    aic_core_num: int = 24,
    aiv_core_num: int = 48,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Generate the SparseAttnSharedkvMetadata output tensor.

    Returns an INT32 tensor of shape ``[SAS_META_SIZE]`` whose layout
    matches the ``SasMetaData`` struct from the Ascend C kernel
    (``AIC_CORE_NUM * FA_METADATA_SIZE`` int32 followed by
    ``AIV_CORE_NUM * FD_METADATA_SIZE`` int32).

    ``layout_q``/``layout_kv``/seq tensors follow the same conventions
    as the C++ kernel. Pass empty / ``None`` tensors when not in TND
    mode (matching ``torch_npu.npu_sparse_attn_sharedkv_metadata``).
    """
    scheduler = _MetadataScheduler(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        ori_topk=ori_topk,
        cmp_topk=cmp_topk,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
        aic_core_num=aic_core_num,
        aiv_core_num=aiv_core_num,
    )
    out = scheduler.run()
    if device is not None:
        out = out.to(device)
    return out
