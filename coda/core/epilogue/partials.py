import cutlass
import cutlass.cute as cute

from quack.gemm_sm90 import GemmSm90
from quack.cute_dsl_utils import ParamsBase
from quack.epi_ops import EpiOp, ColVecReduce, colvec_reduce_accumulate

from coda.core.epilogue.base import Epilogue


class SqSum(Epilogue):

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        else:
            self.name = "mSqSumVec"

    def declares(self) -> tuple[EpiOp, ...]:
        return (ColVecReduce(self.name),)

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        tDrSSq = epi_loop_tensors.get(self.name)
        if cutlass.const_expr(tDrSSq is not None):
            colvec_reduce_accumulate(
                gemm=gemm,
                tDrReduce=tDrSSq,
                tRS_rInput=tRS_rD,
                rScale=tRS_rD,
            )

        return ()
