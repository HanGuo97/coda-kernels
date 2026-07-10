import math
import operator
import cutlass
import cutlass.cute as cute

from quack.cute_dsl_utils import ParamsBase
from quack.varlen_utils import VarlenManager
from quack.gemm_sm90 import GemmSm90
from quack import layout_utils
from quack.epi_ops import (
    EpiOp,
    EpiContext,
    VecReduce,
    colvec_reduce_accumulate,
    _get_lane_warp_layouts,
)
from coda.core.epilogue.base import Epilogue, Const
from coda.core.ops import misc_utils


class SqSumReduce(VecReduce):

    dim = 0
    epi_m_major_preference = -1

    @cute.jit
    def begin(self, gemm: GemmSm90, param: cute.Tensor, smem_tensor: cute.Tensor | None, ctx: EpiContext) -> tuple:
        vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride())
        layout = ctx.partition_for_epilogue_fn(cute.make_rmem_tensor(vec_mma_layout, cute.Float32)).layout
        tDrSSq = cute.make_rmem_tensor(layout, cute.Float32)
        tRS_cD = ctx.partition_for_epilogue_fn(cute.make_identity_tensor(gemm.cta_tile_shape_mnk[:2]))
        return tDrSSq, tRS_cD, ctx.tile_coord_mnkl

    @cute.jit
    def begin_loop(self, gemm: GemmSm90, state: tuple, epi_coord: cute.Coord) -> tuple:
        tDrSSq, tRS_cD, tile_coord_mnkl = state
        rSSq = tDrSSq[None, None, None, epi_coord[0], epi_coord[1]]
        coord = tRS_cD[None, None, None, epi_coord[0], epi_coord[1]]
        if cutlass.const_expr(epi_coord[self._reduce_dim()] == 0):
            cute.filter_zeros(rSSq).fill(0.0)
        return rSSq, coord, tile_coord_mnkl

    @cute.jit
    def end_loop(
        self,
        gemm: GemmSm90,
        param: cute.Tensor,
        state: tuple,
        epi_coord: cute.Coord,
        epi_tile: cute.Tile,
        tiled_copy_t2r: cute.TiledCopy | None,
        tiled_copy_r2s: cute.TiledCopy | None,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        tidx: cute.Int32,
    ) -> None:
        epi_tile_shape = cute.zipped_divide(cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile).shape[1]
        m_idx, n_idx, _, batch_idx = tile_coord_mnkl
        tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
        epi_tile_N = tile_N // epi_tile_shape[1]
        # head boundaries must land on segment ends so a head is never split mid-segment
        misc_utils.static_assert(gemm.head_dim % epi_tile_N == 0)
        n_offset_tile = n_idx * tile_N
        n_offset_epi_tile = n_offset_tile + cutlass.const_expr(epi_coord[1] * epi_tile_N)
        n_idx_offset = n_offset_epi_tile + epi_tile_N

        if cutlass.const_expr(epi_coord[1] == epi_tile_shape[1] - 1) or n_idx_offset % gemm.head_dim == 0:
            tDrSSq, tDcD, _ = state
            rSSq_cur = tDrSSq[None, None, None, epi_coord[0], epi_coord[1]]
            tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
            tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
            reference_src = tiled_copy_t2r is None

            # ── Derive lane layout from tiled_copy ──
            lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
            # For ColVecReduce: reduce across N lanes (lanes_in_N threads share same M row)
            lanes_in_N = cutlass.const_expr(cute.size(lane_layout_MN, mode=[1]))
            warps_in_N = cutlass.const_expr(cute.size(warp_layout_MN, mode=[1]))
            is_lane_n_leader = cute.arch.lane_idx() % lanes_in_N == 0
            # Typically lanes_in_N is 4 for Sm90
            misc_utils.static_assert(
                lanes_in_N == 1 << int(math.log2(lanes_in_N)),
                "lanes_in_N must be a power of 2 for butterfly reduction",
            )
            misc_utils.static_assert(warps_in_N == 1)

            # Intra-warp shuffle reduction across N lanes
            if cutlass.const_expr(lanes_in_N > 1):
                # Assumes threads for each M row are contiguous along N, so
                # warp_reduction over groups of lanes_in_N matches lane_layout_MN.
                misc_utils.static_assert(lane_layout_MN.stride[1] == 1)
                rSSq_flt = cute.filter_zeros(rSSq_cur)
                for i in cutlass.range_constexpr(cute.size(rSSq_flt)):
                    rSSq_flt[i] = cute.arch.warp_reduction(
                        rSSq_flt[i],
                        op=operator.add,
                        threads_in_group=lanes_in_N,
                    )

            rSSq_m = layout_utils.convert_layout_zero_stride(rSSq_cur, rSSq_cur.layout)[None, 0]
            tDcD_m = layout_utils.convert_layout_zero_stride(tDcD_cur, rSSq_cur.layout)[None, 0]

            head_idx = n_offset_epi_tile // gemm.head_dim
            segment_idx = n_idx - (head_idx * gemm.head_dim) // tile_N
            segment_coord = head_idx * gemm.num_segments + segment_idx

            # Write to gmem
            limit_m = min(varlen_manager.len_m(batch_idx) - m_idx * tile_M, tile_M)
            limit_n_tiles = param.shape[2] if not varlen_manager.varlen_m else param.shape[1]
            if cutlass.const_expr(not varlen_manager.varlen_m):
                mSSq = param[batch_idx, None, segment_coord]
            else:
                mSSq = cute.domain_offset(
                    (varlen_manager.params.cu_seqlens_m[batch_idx],),
                    param[None, segment_coord],
                )
            gSSq = cute.local_tile(mSSq, (tile_M,), (m_idx,))
            should_write_gmem = is_lane_n_leader
            if segment_coord < limit_n_tiles and should_write_gmem:
                for m in cutlass.range_constexpr(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        gSSq[row_idx] = rSSq_m[m]
            cute.filter_zeros(rSSq_cur).fill(0.0)


class SqSum(Epilogue):

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        else:
            self.name = "mSqSumVec"

    def declares(self) -> tuple[EpiOp, ...]:
        return (SqSumReduce(self.name),)

    def declare_constexprs(self) -> tuple[Const, ...]:
        return (Const("head_dim", int), Const("num_segments", int))

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        state = epi_loop_tensors.get(self.name)
        if cutlass.const_expr(state is not None):
            rSSq, _, _ = state
            colvec_reduce_accumulate(
                gemm=gemm,
                tDrReduce=rSSq,
                tRS_rInput=tRS_rD,
                rScale=tRS_rD,
            )

        return ()
