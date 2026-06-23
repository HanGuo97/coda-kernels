import cutlass
import cutlass.cute as cute
from typing import Callable

from quack.cute_dsl_utils import ParamsBase
from quack.epi_ops import EpiOp, TileStore
from quack.gemm_act import GemmActMixin
from quack.gemm_sm90 import GemmSm90

from coda.core.epilogue.base import Epilogue


class Act(Epilogue):

    def __init__(self, fn: Callable | None = None) -> None:
        self.fn = fn

    def declares(self) -> tuple[EpiOp, ...]:
        return (TileStore("mAuxOut"),)

    def cache_key(self) -> tuple:
        return ("Act", self.fn)

    def auxiliary_mixin(self) -> type | None:
        return GemmActMixin

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        if cutlass.const_expr(self.fn is not None):
            tRS_rAuxOut = cute.make_rmem_tensor(tRS_rD.layout.shape, gemm.acc_dtype)
            for i in cutlass.range_constexpr(cute.size(tRS_rAuxOut)):
                tRS_rAuxOut[i] = self.fn(tRS_rD[i])
        else:
            tRS_rAuxOut = tRS_rD
        return (tRS_rAuxOut,)
