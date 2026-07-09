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
from coda.core.epilogue.base import Epilogue
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
        cute.filter_zeros(rSSq).fill(cute.Float32.zero)
        return rSSq, coord, tile_coord_mnkl


class SqSum(Epilogue):

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        else:
            self.name = "mSqSumVec"

    def declares(self) -> tuple[EpiOp, ...]:
        return (SqSumStore(self.name),)

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
