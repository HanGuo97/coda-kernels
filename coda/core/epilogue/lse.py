import cutlass
import cutlass.cute as cute

from quack.epi_ops import EpiOp, ColVecLoad
from quack.cute_dsl_utils import ParamsBase
from quack.gemm_sm90 import GemmSm90

from coda.core.ops import misc_utils
from coda.core.ops import reduction_utils
from coda.core.epilogue.base import Const, Epilogue


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
            rMaxVec, rSSEVec, _, coord, tile_coord_mnkl = state
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
        return (ColVecLoad(self.target_name), TargetLogitsSelect(self.logits_name))

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
            rLogits, _, coord, tile_coord_mnkl = state
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
