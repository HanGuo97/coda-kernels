import torch
import cutlass
import cutlass.cute as cute
from typing import NamedTuple

from quack.rounding import RoundingMode
from quack.compile_utils import make_fake_tensor as quack_make_fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.epi_composable import ComposableEpiMixin
from quack.gemm_tvm_ffi_utils import div_for_dtype
from quack.epi_ops import (
    EpiOp,
    Scalar,
    TileLoad,
    TileStore,
    VecReduce,
    ColVecLoad,
    RowVecLoad,
    ColVecReduce,
    RowVecReduce,
)

from coda.core.epilogue.epi_ops import ColVecStore
from coda.core.ops.torch_utils import (
    preprocess_vector,
    preprocess_tensor,
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
    elif isinstance(op, VecReduce):
        assert tensor.ndim == 3
        if tensor.stride(2) == 1:
            major = "n"
        elif tensor.stride(1) == 1:
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
    epi_op_by_name = GemmCls._epi_op_by_name
    epi_const_names = {
        name
        for name, _, _ in GemmCls._extra_param_fields
    }
    epi_keys = []
    for name, arg in epi_args.items():
        if isinstance(arg, torch.Tensor):
            epi_key = _key_field(
                op=epi_op_by_name[name],
                tensor=arg,
            )
        else:
            assert name in epi_const_names
            epi_key = EpilogueKeyConst(
                name=name,
                dtype=type(arg),
                value=arg,
            )
        epi_keys.append(epi_key)
    return tuple(epi_keys)


def preprocess_epi_args(GemmCls: type[ComposableEpiMixin], epi_args: dict) -> dict:
    epi_op_by_name = GemmCls._epi_op_by_name
    epi_args_preprocessed = {}
    for name, arg in epi_args.items():
        if not isinstance(arg, torch.Tensor):
            epi_args_preprocessed[name] = arg
            continue
        op = epi_op_by_name[name]
        if isinstance(op, (ColVecLoad, RowVecLoad, ColVecStore)):
            assert arg.is_contiguous()
            epi_args_preprocessed[name] = preprocess_vector(arg, permute=False)
        elif isinstance(op, VecReduce):
            if isinstance(op, RowVecReduce):
                # tiles on the leading dimension: the .mT of an (n, m_tiles) buffer 
                assert arg.stride(-2) == 1
            else:
                assert arg.is_contiguous()
            epi_args_preprocessed[name] = preprocess_tensor(arg, permute=False)
        elif isinstance(op, (TileLoad, TileStore)):
            epi_args_preprocessed[name] = preprocess_tensor(arg, permute=True)
        elif isinstance(op, Scalar):
            # fp32: the compiled fake assumes 4-byte alignment
            assert arg.numel() == 1
            assert arg.dtype == torch.float32
            epi_args_preprocessed[name] = arg
        else:
            raise TypeError
    return epi_args_preprocessed


def _make_fake_epi_arg(
    op: EpiOp,
    dtype: torch.dtype,
    major: str | None,
    m: cute.SymInt,
    n: cute.SymInt,
    k: cute.SymInt,
    l: cute.SymInt,
) -> cute.Tensor | cute.Pointer:
    cutlass_dtype: type[cute.Numeric] = torch2cute_dtype_map[dtype]
    if isinstance(op, ColVecLoad):
        return quack_make_fake_tensor(
            dtype=cutlass_dtype,
            shape=(l, m),
            divisibility=4,
            leading_dim=1,
        )
    if isinstance(op, RowVecLoad):
        return quack_make_fake_tensor(
            dtype=cutlass_dtype,
            shape=(l, n),
            divisibility=4,
            leading_dim=1,
        )
    if isinstance(op, ColVecReduce):
        n_tiles = cute.sym_int()
        return quack_make_fake_tensor(
            dtype=cutlass_dtype,
            shape=(l, m, n_tiles),
            leading_dim=2,
            divisibility=1,
        )
    if isinstance(op, RowVecReduce):
        m_tiles = cute.sym_int()
        # This is almost always followed by a reduction across the tiles,
        # keep the tiles on the leading dimension makes the follow-up fast.
        return quack_make_fake_tensor(
            dtype=cutlass_dtype,
            shape=(l, m_tiles, n),
            leading_dim=1,
            divisibility=1,
        )
    if isinstance(op, TileLoad):
        leading_dim = 1 if major == "n" else 0
        return quack_make_fake_tensor(
            dtype=cutlass_dtype,
            shape=(m, n, l),
            divisibility=div_for_dtype(cutlass_dtype),
            leading_dim=leading_dim,
        )
    if isinstance(op, TileStore):
        leading_dim = 1 if major == "n" else 0
        if op.epi_tile_fn is not None:
            # if `epi_tile_fn` is present, we cannot determine auxiliary tensor shape
            # since `SymInt` does not support arbitrary algebra, we so far assume that
            # `epi_tile_fn` will only change `n`, not `m`
            _n = cute.sym_int()
        else:
            _n = n
        return quack_make_fake_tensor(
            dtype=cutlass_dtype,
            shape=(m, _n, l),
            divisibility=div_for_dtype(cutlass_dtype),
            leading_dim=leading_dim,
        )
    if isinstance(op, ColVecStore):
        return quack_make_fake_tensor(
            dtype=cutlass_dtype,
            shape=(l, m),
            divisibility=1,
            leading_dim=1,
        )
    if isinstance(op, Scalar):
        return cute.runtime.make_ptr(
            dtype=cutlass_dtype,
            value=0,
            mem_space=cute.AddressSpace.gmem,
            assumed_align=4,
        )
    raise NotImplementedError


def compile_epi_args(
    GemmCls: type[ComposableEpiMixin],
    epi_keys: tuple[EpilogueKeyTensor | EpilogueKeyConst, ...],
    add_to_output: bool,
    rounding_mode: RoundingMode,
    sr_seed: int | None,
    m: cute.SymInt,
    n: cute.SymInt,
    k: cute.SymInt,
    l: cute.SymInt,
) -> tuple:
    EpiArgCls = GemmCls.EpilogueArguments
    epi_op_by_name = GemmCls._epi_op_by_name
    epi_args_fake = {
        "add_to_output": add_to_output,
        "rounding_mode": rounding_mode,
        "sr_seed": sr_seed,
    }
    epi_args_fake = {
        name: value
        for name, value in epi_args_fake.items()
        if name in EpiArgCls._fields
    }
    for epi_key in epi_keys:
        assert epi_key.name not in epi_args_fake.keys()
        if isinstance(epi_key, EpilogueKeyTensor):
            epi_args_fake[epi_key.name] = _make_fake_epi_arg(
                op=epi_op_by_name[epi_key.name],
                dtype=epi_key.dtype,
                major=epi_key.major,
                m=m,
                n=n,
                k=k,
                l=l,
            )
        else:
            epi_args_fake[epi_key.name] = epi_key.value
    return EpiArgCls(**epi_args_fake)


def process_epi_args(
    GemmCls: type[ComposableEpiMixin],
    epi_args: dict,
    add_to_output: bool | None,
    rounding_mode: int | None,
    sr_seed: int | None,
) -> tuple:
    assert add_to_output is None
    assert rounding_mode is None
    assert sr_seed is None
    EpiArgCls = GemmCls.EpilogueArguments
    epi_op_by_name = GemmCls._epi_op_by_name
    epi_args_processed = {}
    for name in EpiArgCls._fields:
        # certain fields (`rounding_mode`, etc) aren't in `epi_args`
        epi_arg = epi_args.get(name, None)
        if isinstance(epi_arg, torch.Tensor):
            if isinstance(epi_op_by_name[name], Scalar):
                epi_args_processed[name] = epi_arg.data_ptr()
            else:
                epi_args_processed[name] = epi_arg
        else:
            epi_args_processed[name] = None
    return EpiArgCls(**epi_args_processed)
