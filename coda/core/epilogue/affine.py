import cutlass
import cutlass.cute as cute

from quack import utils as quack_utils
from quack.cute_dsl_utils import ParamsBase
from quack.epi_ops import EpiOp, Scalar, ColVecLoad, RowVecLoad, TileLoad
from quack.gemm_sm90 import GemmSm90

from coda.core.epilogue.base import Epilogue


class Affine(Epilogue):

    def declares(self) -> tuple[EpiOp, ...]:
        return (
            Scalar("alpha"),
            Scalar("beta"),
            RowVecLoad("mRowVecBroadcast"),
            ColVecLoad("mColVecBroadcast"),
        )

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:

        tDrRowVec = epi_loop_tensors.get("mRowVecBroadcast")
        tDrColVec = epi_loop_tensors.get("mColVecBroadcast")
        rD = tRS_rD.load()

        if cutlass.const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = quack_utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha

        if cutlass.const_expr(tRS_rC is not None):
            if cutlass.const_expr(not hasattr(params, "beta") or params.beta is None):
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = quack_utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)

        if cutlass.const_expr(tDrRowVec is not None):
            for i in cutlass.range_constexpr(cute.size(tDrRowVec)):
                tRS_rD[i] += tDrRowVec[i]

        if cutlass.const_expr(tDrColVec is not None):
            for i in cutlass.range_constexpr(cute.size(tDrColVec)):
                tRS_rD[i] += tDrColVec[i]

        return ()


class Scale(Epilogue):

    def declares(self) -> tuple[EpiOp, ...]:
        return (
            RowVecLoad("mRowVecBroadcast"),
            ColVecLoad("mColVecBroadcast"),
        )

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:

        tDrRowVec = epi_loop_tensors.get("mRowVecBroadcast")
        tDrColVec = epi_loop_tensors.get("mColVecBroadcast")

        if cutlass.const_expr(tDrRowVec is not None):
            for i in cutlass.range_constexpr(cute.size(tDrRowVec)):
                tRS_rD[i] *= tDrRowVec[i]

        if cutlass.const_expr(tDrColVec is not None):
            for i in cutlass.range_constexpr(cute.size(tDrColVec)):
                tRS_rD[i] *= tDrColVec[i]

        return ()


class Residual(Epilogue):

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        else:
            self.name = "mResidual"

    def declares(self) -> tuple[EpiOp, ...]:
        return (TileLoad(self.name),)

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        tDrResidual = epi_loop_tensors.get(self.name)
        if cutlass.const_expr(tDrResidual is not None):
            rD = tRS_rD.load()
            rD += tDrResidual.load().to(tRS_rD.element_type)
            tRS_rD.store(rD)
        return ()
