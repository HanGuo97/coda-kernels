import operator
import cutlass
import cutlass.cute as cute

from quack.cute_dsl_utils import ParamsBase
from quack.gemm_sm90 import GemmSm90
from quack.epi_ops import (
    EpiOp,
    RowVecLoad,
    colvec_reduce_accumulate,
)

from coda.core.epilogue.base import Epilogue, Const


class QKNorm(Epilogue):

    def __init__(self, gamma_name: str | None = None, mask_name: str | None = None) -> None:
        if gamma_name is not None:
            self.gamma_name = gamma_name
        else:
            self.gamma_name = "mGammaVec"

        if mask_name is not None:
            self.mask_name = mask_name
        else:
            self.mask_name = "mMaskVec"

    def declares(self) -> tuple[EpiOp, ...]:
        return (PerHeadNormState(self.gamma_name), RowVecLoad(self.mask_name))

    def declare_constexprs(self) -> tuple[Const, ...]:
        return (Const("head_dim", int), Const("eps", float))

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        state = epi_loop_tensors.get(self.gamma_name)
        if cutlass.const_expr(state is not None):
            rGamma, rSSq, lanes_in_N = state
            rMask = epi_loop_tensors.get(self.mask_name)
            colvec_reduce_accumulate(
                gemm=gemm,
                tDrReduce=rSSq,
                tRS_rInput=tRS_rD,
                rScale=tRS_rD,
            )
            if cutlass.const_expr(lanes_in_N > 1):
                rSSq_flt = cute.filter_zeros(rSSq)
                for i in cutlass.range_constexpr(cute.size(rSSq_flt)):
                    rSSq_flt[i] = cute.arch.warp_reduction(
                        rSSq_flt[i],
                        op=operator.add,
                        threads_in_group=lanes_in_N,
                    )

            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                rms = cute.math.rsqrt(rSSq[i] / params.head_dim + params.eps, fastmath=True)
                normed = tRS_rD[i] * rms * rGamma[i].to(dtype=cute.Float32)
                mask = rMask[i].to(dtype=cute.Float32)
                tRS_rD[i] = mask * normed + (1.0 - mask) * tRS_rD[i]

        return ()
