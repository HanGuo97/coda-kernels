import operator
import cutlass
import cutlass.cute as cute
from quack.cute_dsl_utils import ParamsBase
from quack.gemm_sm90 import GemmSm90
from quack import layout_utils
from quack.epi_ops import (
    EpiOp,
    VecReduce,
    colvec_reduce_accumulate,
    _get_lane_warp_layouts,
)

from coda.core.epilogue.base import Epilogue
from coda.core.ops import misc_utils


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
