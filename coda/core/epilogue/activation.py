import cutlass
import cutlass.cute as cute
from typing import Callable

from quack.cute_dsl_utils import ParamsBase
from quack.epi_ops import EpiOp, ColVecLoad, RowVecLoad, TileStore
from quack.gemm_act import GemmActMixin, GemmGatedMixin, _gated_epi_tile_fn
from quack.gemm_sm90 import GemmSm90

from coda.core.ops import creation_utils
from coda.core.epilogue.base import Epilogue


class Act(Epilogue):

    def __init__(self, fn: Callable | None = None) -> None:
        self.fn = fn

    def declares(self) -> tuple[EpiOp, ...]:
        return (TileStore("mAuxOut"),)

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


class Pairwise(Epilogue):

    def __init__(self, fn: Callable | None = None) -> None:
        self.fn = fn

    def declares(self) -> tuple[EpiOp, ...]:
        return (TileStore("mAuxOut"),)

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
            for i in cutlass.range_constexpr(cute.size(tRS_rAuxOut) // 2):
                tRS_rAuxOut[2 * i], tRS_rAuxOut[2 * i + 1] = self.fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            tRS_rAuxOut = tRS_rD
        return (tRS_rAuxOut,)


class Gated(Epilogue):

    def __init__(self, fn: Callable | None = None) -> None:
        self.fn = fn

    def declares(self) -> tuple[EpiOp, ...]:
        return (TileStore("mAuxOut", epi_tile_fn=_gated_epi_tile_fn),)

    def auxiliary_mixin(self) -> type | None:
        return GemmGatedMixin

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
            tRS_rAuxOut = creation_utils.allocate_tensor_from_recast_layout(
                layout=tRS_rD.layout,
                new_type_bits=2,
                old_type_bits=1,
                memspace="rmem",
                dtype=gemm.acc_dtype,
            )
            for i in cutlass.range_constexpr(cute.size(tRS_rAuxOut)):
                tRS_rAuxOut[i] = self.fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            tRS_rAuxOut = tRS_rD
        return (tRS_rAuxOut,)


class RoPE(Epilogue):

    def __init__(self, pos_idx_name: str | None = None, inv_freq_name: str | None = None) -> None:
        if pos_idx_name is not None:
            self.pos_idx_name = pos_idx_name
        else:
            self.pos_idx_name = "pos_idx"

        if inv_freq_name is not None:
            self.inv_freq_name = inv_freq_name
        else:
            self.inv_freq_name = "inv_freq"

    def declares(self) -> tuple[EpiOp, ...]:
        return (ColVecLoad(self.pos_idx_name), RowVecLoad(self.inv_freq_name))

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        pos_idx = epi_loop_tensors.get(self.pos_idx_name)
        inv_freq = epi_loop_tensors.get(self.inv_freq_name)
        if cutlass.const_expr(pos_idx is not None and inv_freq is not None):
            for i in cutlass.range_constexpr(cute.size(tRS_rD) // 2):
                a = pos_idx[2 * i].to(dtype=gemm.acc_dtype) * inv_freq[2 * i].to(dtype=gemm.acc_dtype)
                c = cute.math.cos(a, fastmath=True)
                s = cute.math.sin(a, fastmath=True)
                x = tRS_rD[2 * i]
                y = tRS_rD[2 * i + 1]
                tRS_rD[2 * i] = x * c + y * s
                tRS_rD[2 * i + 1] = y * c - x * s
        return ()
