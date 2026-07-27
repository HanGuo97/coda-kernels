import cutlass
import cutlass.cute as cute

from quack.activation import dswiglu
from quack.cute_dsl_utils import ParamsBase
from quack.gemm_act import GemmActMixin
from quack.gemm_sm90 import GemmSm90
from quack.epi_ops import EpiOp, ColVecReduce, TileLoad, TileStore, colvec_reduce_accumulate

from coda.core.ops import misc_utils
from coda.core.epilogue.base import Epilogue, Const


class SwiGLUBwdZdZ(Epilogue):

    def declares(self) -> tuple[EpiOp, ...]:
        return (
            TileLoad("mZPacked"),
            TileStore("mAuxOut"),
            ColVecReduce("mZdZVec"),
        )

    def declare_constexprs(self) -> tuple[Const, ...]:
        return (Const("scale", float),)

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
        tDrZPacked = epi_loop_tensors.get("mZPacked")
        tDrZdZ = epi_loop_tensors.get("mZdZVec")

        if cutlass.const_expr(tDrZPacked is not None):
            misc_utils.static_assert(tDrZPacked.dtype == cute.Int32)
            rZ = cute.recast_tensor(tDrZPacked, dtype=gemm.d_dtype)
            rDZ = cute.make_rmem_tensor_like(rZ)
            rZdZ = cute.make_rmem_tensor_like(tRS_rD, dtype=cute.Float32)
            tRS_rAuxOut = cute.recast_tensor(rDZ, dtype=cute.Int32)
            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                g = rZ[2 * i].to(dtype=cute.Float32)
                u = rZ[2 * i + 1].to(dtype=cute.Float32)
                dout = tRS_rD[i].to(dtype=cute.Float32)
                dg, du, o = dswiglu(x=g, y=u, dout=dout)
                rDZ[2 * i] = dg.to(dtype=rDZ.dtype)
                rDZ[2 * i + 1] = du.to(dtype=rDZ.dtype)
                zdz = g * dg + u * du
                if cutlass.const_expr(params.scale is not None):
                    zdz = zdz * params.scale
                rZdZ[i] = zdz.to(dtype=rZdZ.dtype)
                tRS_rD[i] = o.to(dtype=tRS_rD.dtype)

            if cutlass.const_expr(tDrZdZ is not None):
                colvec_reduce_accumulate(
                    gemm=gemm,
                    tDrReduce=tDrZdZ,
                    tRS_rInput=rZdZ,
                    rScale=None,
                )
        else:
            tRS_rAuxOut = tRS_rD

        return (tRS_rAuxOut,)
