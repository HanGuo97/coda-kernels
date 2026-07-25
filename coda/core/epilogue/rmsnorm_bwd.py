import cutlass
import cutlass.cute as cute

from quack.cute_dsl_utils import ParamsBase
from quack.gemm_act import GemmActMixin
from quack.gemm_sm90 import GemmSm90
from quack.epi_ops import (
    EpiOp,
    ColVecLoad,
    RowVecLoad,
    RowVecReduce,
    TileLoad,
    TileStore,
    rowvec_reduce_accumulate,
)

from coda.core.ops import misc_utils
from coda.core.epilogue.base import Epilogue


class ResidualRMSNormBwd(Epilogue):

    def declares(self) -> tuple[EpiOp, ...]:
        return (
            ColVecLoad("mColVecR"),
            ColVecLoad("mColVecZdZ"),
            RowVecLoad("mRowVecW"),
            TileLoad("mMatrixC"),
            TileStore("mAuxOut"),
            RowVecReduce("mDWVec"),
        )

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
        tDrC = epi_loop_tensors.get("mMatrixC")
        tDrR = epi_loop_tensors.get("mColVecR")
        tDrW = epi_loop_tensors.get("mRowVecW")
        tDrDW = epi_loop_tensors.get("mDWVec")
        tDrZdZ = epi_loop_tensors.get("mColVecZdZ")

        if cutlass.const_expr(tDrC is not None):
            tDrR = misc_utils.static_assert_is_Tensor(tDrR)
            tDrW = misc_utils.static_assert_is_Tensor(tDrW)
            tDrZdZ = misc_utils.static_assert_is_Tensor(tDrZdZ)
            rDCNorm = cute.make_rmem_tensor_like(tRS_rD, dtype=cute.Float32)
            tRS_rAuxOut = cute.make_rmem_tensor_like(tRS_rD, dtype=gemm.acc_dtype)

            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                d = tRS_rD[i].to(dtype=cute.Float32)
                c = tDrC[i].to(dtype=cute.Float32)
                r = tDrR[i].to(dtype=cute.Float32)
                w = tDrW[i].to(dtype=cute.Float32)
                zdz = tDrZdZ[i].to(dtype=cute.Float32)
                c_norm = c * r
                rDCNorm[i] = (d * c_norm).to(dtype=rDCNorm.dtype)
                tRS_rAuxOut[i] = (c_norm * w).to(dtype=tRS_rAuxOut.dtype)
                tRS_rD[i] = ((d * w - c_norm * zdz) * r).to(dtype=tRS_rD.dtype)

            if cutlass.const_expr(tDrDW is not None):
                rowvec_reduce_accumulate(
                    gemm=gemm,
                    tDrReduce=tDrDW,
                    tRS_rInput=rDCNorm,
                    rScale=None,
                )
        else:
            tRS_rAuxOut = tRS_rD

        return (tRS_rAuxOut,)
