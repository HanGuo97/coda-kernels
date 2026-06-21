import cutlass
import cutlass.cute as cute
from dataclasses import MISSING
from typing import Callable, Iterable

from quack.gemm_sm90 import GemmSm90
from quack.cute_dsl_utils import mlir_namedtuple, ParamsBase
from quack.epi_ops import Scalar, RowVecLoad, ColVecLoad, TileStore, TileLoad, EpiOp


class Epilogue(object):

    def declares(self) -> tuple[EpiOp, ...]:
        return ()

    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        return ()

    def cache_key(self) -> tuple:
        return (type(self).__name__,)

    def bind(self, name: str, gemm_cls: type) -> type:
        return _lower(self, name=name, gemm_cls=gemm_cls)


class _Composite(Epilogue):

    def __init__(self, epilogues: Iterable["Epilogue"]) -> None:
        self._children = list(epilogues)

    def declares(self) -> tuple[EpiOp, ...]:
        return tuple(op for child in self._children for op in child.declares())

    def cache_key(self) -> tuple:
        return tuple(child.cache_key() for child in self._children)

    @cute.jit
    def visit(
        self,
        gemm: GemmSm90,
        params: ParamsBase,
        epi_loop_tensors: dict,
        tRS_rD: cute.Tensor,
        tRS_rC: cute.Tensor | None,
    ) -> tuple[cute.Tensor, ...]:
        tRS_rAuxOuts = []
        for child in self._children:
            tRS_rAuxOuts.extend(
                child.visit(
                    gemm=gemm,
                    params=params,
                    epi_loop_tensors=epi_loop_tensors,
                    tRS_rD=tRS_rD,
                    tRS_rC=tRS_rC,
                )
            )
        if cutlass.const_expr(len(tRS_rAuxOuts) > 1):
            raise NotImplementedError
        return tuple(tRS_rAuxOuts)


def compose(epilogues: Iterable["Epilogue"]) -> "Epilogue":
    return _Composite(list(epilogues))


def _arg_field(op: EpiOp) -> tuple:
    if isinstance(op, Scalar):
        if op.dtype is None:
            # if `dtype` is None, quack defaults to FP32
            # https://github.com/Dao-AILab/quack/blob/v0.5.2/quack/epi_ops.py#L281
            dtype = cute.Float32
        else:
            dtype = op.dtype
        return (op.name, dtype | cute.Tensor | None, None)
    if isinstance(op, (RowVecLoad, ColVecLoad, TileLoad)):
        return (op.name, cute.Tensor | None, None)
    if isinstance(op, TileStore):
        return (op.name, cute.Tensor, MISSING)
    raise TypeError(f"unknown op {op!r}")


def _ops_compatible(op_a: EpiOp, op_b: EpiOp) -> bool:
    return all([
        type(op_a) is type(op_b),
        _arg_field(op_a) == _arg_field(op_b),
        getattr(op_a, "epi_tile_fn", None) is getattr(op_b, "epi_tile_fn", None),
    ])


def _normalize(ops: Iterable[EpiOp]) -> tuple[EpiOp, ...]:
    ops_dict = {}
    ops_normalized = []
    for op in ops:
        op_prev = ops_dict.get(op.name)
        if op_prev is None:
            # new op
            ops_dict[op.name] = op
            ops_normalized.append(op)
        elif not _ops_compatible(op, op_prev):
            # duplicate but imcompatible op
            raise ValueError

    # we only support one output for now
    if sum(isinstance(op, TileStore) for op in ops_normalized) > 1:
        raise NotImplementedError

    return tuple(ops_normalized)
