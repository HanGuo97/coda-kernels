import math
import cutlass
import cutlass.cute as cute

from quack import layout_utils
from quack.gemm_sm90 import GemmSm90
from quack.cute_dsl_utils import ParamsBase
from quack.varlen_utils import VarlenManager
from quack.epi_ops import EpiOp, ColVecLoad, VecReduce, EpiContext

from coda.core.ops import misc_utils
from coda.core.ops import reduction_utils
from coda.core.epilogue.base import Const, Epilogue


class LSEReduce(VecReduce):

    dim = 0
    epi_m_major_preference = -1

    @cute.jit
    def begin(self, gemm: GemmSm90, param: cute.Tensor, smem_tensor: cute.Tensor | None, ctx: EpiContext) -> tuple:
        vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride())
        layout = ctx.partition_for_epilogue_fn(cute.make_rmem_tensor(vec_mma_layout, cute.Float32)).layout
        tDrMax = cute.make_rmem_tensor(layout, cute.Float32)
        tDrSSE = cute.make_rmem_tensor(layout, cute.Float32)
        tRS_cD = ctx.partition_for_epilogue_fn(cute.make_identity_tensor(gemm.cta_tile_shape_mnk[:2]))
        return tDrMax, tDrSSE, tRS_cD, ctx.tile_coord_mnkl

    @cute.jit
    def begin_loop(self, gemm: GemmSm90, state: tuple, epi_coord: cute.Coord) -> tuple:
        tDrMax, tDrSSE, tRS_cD, tile_coord_mnkl = state
        rMax = tDrMax[None, None, None, epi_coord[0], epi_coord[1]]
        rSSE = tDrSSE[None, None, None, epi_coord[0], epi_coord[1]]
        coord = tRS_cD[None, None, None, epi_coord[0], epi_coord[1]]
        if cutlass.const_expr(epi_coord[self._reduce_dim()] == 0):
            cute.filter_zeros(rMax).fill(-cute.Float32.inf)
            cute.filter_zeros(rSSE).fill(0.0)
        return rMax, rSSE, coord, tile_coord_mnkl

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
        if cutlass.const_expr(epi_coord[1] == epi_tile_shape[1] - 1):
            m_idx, n_idx, _, batch_idx = tile_coord_mnkl
            tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]

            tDrMax, tDrSSE, tDcD, _ = state
            rMax_cur = tDrMax[None, None, None, epi_coord[0], epi_coord[1]]
            rSSE_cur = tDrSSE[None, None, None, epi_coord[0], epi_coord[1]]
            tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
            tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
            reference_src = tiled_copy_t2r is None

            # ── Derive lane layout from tiled_copy ──
            lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
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
                rMax_flt = cute.filter_zeros(rMax_cur)
                rSSE_flt = cute.filter_zeros(rSSE_cur)
                # Assumes threads for each M row are contiguous along N, so
                # warp_reduction over groups of lanes_in_N matches lane_layout_MN.
                misc_utils.static_assert(lane_layout_MN.stride[1] == 1)
                misc_utils.static_assert(cute.size(rMax_flt) == cute.size(rSSE_flt))
                for i in cutlass.range_constexpr(cute.size(rMax_flt)):
                    rMax_flt[i], rSSE_flt[i] = reduction_utils.online_softmax_combine_warp(
                        m=rMax_flt[i],
                        s=rSSE_flt[i],
                        width=lanes_in_N,
                    )

            rMax_m = layout_utils.convert_layout_zero_stride(rMax_cur, rMax_cur.layout)[None, 0]
            rSSE_m = layout_utils.convert_layout_zero_stride(rSSE_cur, rSSE_cur.layout)[None, 0]
            tDcD_m = layout_utils.convert_layout_zero_stride(tDcD_cur, rMax_cur.layout)[None, 0]

            # Write to gmem
            limit_m = min(varlen_manager.len_m(batch_idx) - m_idx * tile_M, tile_M)
            limit_n_tiles = param.shape[2] if not varlen_manager.varlen_m else param.shape[1]
            if cutlass.const_expr(not varlen_manager.varlen_m):
                mLSE = param[batch_idx, None, n_idx]
            else:
                mLSE = cute.domain_offset(
                    (varlen_manager.params.cu_seqlens_m[batch_idx],),
                    param[None, n_idx],
                )
            gLSE = cute.local_tile(mLSE, (tile_M,), (m_idx,))
            should_write_gmem = is_lane_n_leader
            if n_idx < limit_n_tiles and should_write_gmem:
                for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        # Empty-tile guard: if no element was ever observed,
                        # max stays at -inf and sse at 0. Substitute 0 for max
                        # so the fastmath log(0) flows to a clean -inf instead
                        # of -inf + log(0), which is implementation-defined
                        # under fastmath.
                        row_max = (
                            rMax_m[m]
                            if rMax_m[m] > -cute.Float32.inf
                            else cute.Float32.zero
                        )
                        lse = row_max + cute.math.log(rSSE_m[m], fastmath=True)
                        gLSE[row_idx] = lse.to(dtype=gLSE.dtype)


class LSE(Epilogue):

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        else:
            self.name = "mLSEVec"

    def declares(self) -> tuple[EpiOp, ...]:
        return (LSEReduce(self.name),)

    def declare_constexprs(self) -> tuple[Const, ...]:
        return (Const("vocab_size", int),)

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
            rMaxVec, rSSEVec, coord, tile_coord_mnkl = state
            n_offset_tile = tile_coord_mnkl[1] * gemm.cta_tile_shape_mnk[1]
            misc_utils.static_assert(cute.size(rMaxVec) == cute.size(rSSEVec))

            for i in cutlass.range_constexpr(cute.size(rMaxVec)):
                # Skip OOB N-columns when N % tile_N != 0. Without this,
                # OOB lanes feed `tRS_rD = 0` (GEMM accumulator default)
                # into combine_singleton, anchoring max at 0 and adding
                # spurious exp(0 - max) terms to the row's sse.
                col_idx = coord[i][1]
                col_idx_offset = col_idx + n_offset_tile

                if col_idx_offset < params.vocab_size:
                    rMaxVec[i], rSSEVec[i] = reduction_utils.online_softmax_combine_singleton(
                        m0=rMaxVec[i],
                        m1=tRS_rD[i],
                        s0=rSSEVec[i],
                        s1=misc_utils.get_dtype(rSSEVec)(1),
                    )
        return ()


class ColVecLoadNoCast(ColVecLoad):

    @cute.jit
    def begin(self, gemm: GemmSm90, param: cute.Tensor, smem_tensor: cute.Tensor | None, ctx: EpiContext) -> list:
        tDsV, _ = super().begin(gemm=gemm, param=param, smem_tensor=smem_tensor, ctx=ctx)
        tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
        tDrV_cvt = cute.make_rmem_tensor(tDsV_sub.layout, param.element_type)
        return [tDsV, tDrV_cvt]

    @cute.jit
    def begin_loop(self, gemm: GemmSm90, state: list, epi_coord: cute.Coord) -> cute.Tensor:
        tDsV, tDrV_cvt = state[0], state[1]
        should_load = cute.Boolean(True)
        if cutlass.const_expr(self.dim == 1):
            if cutlass.const_expr(gemm.epi_m_major):
                should_load = epi_coord[0] == 0
        else:
            if cutlass.const_expr(not gemm.epi_m_major):
                should_load = epi_coord[1] == 0
        if should_load:
            tDsV_cur = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, epi_coord]
            tDrV = cute.make_rmem_tensor(tDsV_cur.layout, tDsV_cur.element_type)
            cute.autovec_copy(cute.filter_zeros(tDsV_cur), cute.filter_zeros(tDrV))
            tDrV_cvt.store(tDrV.load())
        return tDrV_cvt


class SelectLogits(Epilogue):

    def __init__(self, target_name: str | None = None, logits_name: str | None = None) -> None:
        if target_name is not None:
            self.target_name = target_name
        else:
            self.target_name = "mTarget"

        if logits_name is not None:
            self.logits_name = logits_name
        else:
            self.logits_name = "mLogits"

    def declares(self) -> tuple[EpiOp, ...]:
        return (ColVecLoadNoCast(self.target_name), TargetLogitsSelect(self.logits_name))

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        state = epi_loop_tensors.get(self.logits_name)
        if cutlass.const_expr(state is not None):
            rLogits, coord, tile_coord_mnkl = state
            rTarget = epi_loop_tensors.get(self.target_name)
            n_offset_tile = tile_coord_mnkl[1] * gemm.cta_tile_shape_mnk[1]
            logits_dtype = misc_utils.get_dtype(rLogits)

            misc_utils.static_assert(cute.size(rTarget) == cute.size(coord))
            for i in cutlass.range_constexpr(cute.size(rTarget)):
                target  = rTarget[i]
                col_idx = coord[i][1]
                col_idx_offset = col_idx + n_offset_tile

                if col_idx_offset == target:
                    target_logits = logits_dtype(tRS_rD[i])
                    rLogits[i] = target_logits
        return ()
