import torch
import cutlass
import cutlass.cute as cute
from typing import NamedTuple

from quack.epi_composable import ComposableEpiMixin
from quack.epi_ops import (
    EpiOp,
    TileLoad,
    TileStore,
    VecReduce,
    ColVecLoad,
    RowVecLoad,
)


class EpilogueKeyTensor(NamedTuple):
    name: str
    dtype: torch.dtype
    major: str | None


class EpilogueKeyConst(NamedTuple):
    name: str
    dtype: type
    value: object


def _key_field(op: EpiOp, tensor: torch.Tensor) -> EpilogueKeyTensor:
    if isinstance(op, (TileLoad, TileStore)):
        assert tensor.ndim == 3
        if tensor.stride(1) == 1:
            major = "n"
        elif tensor.stride(0) == 1:
            major = "m"
        else:
            raise ValueError
        return EpilogueKeyTensor(
            name=op.name,
            dtype=tensor.dtype,
            major=major,
        )
    else:
        return EpilogueKeyTensor(
            name=op.name,
            dtype=tensor.dtype,
            major=None,
        )


def make_epi_keys(GemmCls: type[ComposableEpiMixin], epi_args: dict) -> tuple[EpilogueKeyTensor | EpilogueKeyConst, ...]:
    epi_op_by_name = {
        op.name: op
        for op in GemmCls._epi_ops
    }
    epi_keys = []
    for name, arg in epi_args.items():
        if isinstance(arg, torch.Tensor):
            epi_key = _key_field(
                op=epi_op_by_name[name],
                tensor=arg,
            )
        else:
            epi_key = EpilogueKeyConst(
                name=name,
                dtype=type(arg),
                value=arg,
            )
        epi_keys.append(epi_key)
    return tuple(epi_keys)
