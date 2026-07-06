import cutlass
import cutlass.cute as cute

from quack.cute_dsl_utils import ParamsBase
from quack.gemm_sm90 import GemmSm90
from quack.epi_ops import (
    EpiOp,
    RowVecLoad,
    vec_multiply,
    colvec_reduce_accumulate,
    _get_lane_warp_layouts,
)

from coda.core.epilogue.base import Epilogue


class QKSqSum(Epilogue):

    def __init__(self, name: str | None = None, gamma_name: str | None = None) -> None:
        if name is not None:
            self.name = name
        else:
            self.name = "mSqSumVec"

        if gamma_name is not None:
            self.gamma_name = gamma_name
        else:
            self.gamma_name = "mGammaVec"

    def declares(self) -> tuple[EpiOp, ...]:
        return (SqSumStore(self.name), RowVecLoad(self.gamma_name))

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
        gamma = epi_loop_tensors.get(self.gamma_name)
        if cutlass.const_expr(state is not None and gamma is not None):
            rSq, _, _ = state
            colvec_reduce_accumulate(
                gemm=gemm,
                tDrReduce=rSq,
                tRS_rInput=tRS_rD,
                rScale=tRS_rD,
            )
            vec_multiply(
                gemm=gemm,
                tRS_rD=tRS_rD,
                tDrColVec=None,
                tDrRowVec=gamma,
            )
        return ()
