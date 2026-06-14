import cutlass
import cutlass.cute as cute
from collections.abc import Callable
from cutlass.cutlass_dsl import T, dsl_user_op

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector

from coda.core.ops import misc_utils

Scalar = cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue
Tensor = cute.Tensor | cute.TensorSSA | Scalar


def make_dispatch_function(
    fn_tensor: Callable[[object], cute.Tensor] | None = None,
    fn_tensorssa: Callable[[object], cute.TensorSSA] | None = None,
    fn_scalar: Callable[[object], Scalar] | None = None,
    dispatch_policy: str | None = None,
) -> Callable[[object], Tensor]:
    """Creates a function that dispatches to appropriate function based on input types.

    :param fn_tensor: function to call for cute.Tensor inputs
    :param fn_tensorssa: function to call for cute.TensorSSA inputs
    :param fn_scalar: function to call for scalar inputs
    :param dispatch_policy: dispatching strategy
    :return: dispatcher function
    """
    if dispatch_policy is None:
        dispatch_policy = "first"

    def _dispatcher(*args, **kwargs) -> Tensor:
        if cutlass.const_expr(dispatch_policy == "first"):
            if cutlass.const_expr(len(args) > 0):
                dispatch_arg = args[0]
            else:
                raise ValueError

        if cutlass.const_expr(isinstance(dispatch_arg, cute.Tensor)):
            if cutlass.const_expr(fn_tensor is not None):
                return fn_tensor(*args, **kwargs)
            else:
                raise NotImplementedError
        if cutlass.const_expr(isinstance(dispatch_arg, cute.TensorSSA)):
            if cutlass.const_expr(fn_tensorssa is not None):
                return fn_tensorssa(*args, **kwargs)
            else:
                raise NotImplementedError
        if cutlass.const_expr(isinstance(dispatch_arg, Scalar)):
            if cutlass.const_expr(fn_scalar is not None):
                return fn_scalar(*args, **kwargs)
            else:
                raise NotImplementedError
        raise NotImplementedError

    return _dispatcher


def make_tensorssa_fn_from_scalar_fn(
    fn_scalar: Callable[[object], Scalar],
    variadic_policy: str | None = None,
) -> Callable[[object], cute.TensorSSA]:
    # https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

    if variadic_policy is None:
        variadic_policy = "first"

    @cute.jit
    def _tensorssa_fn(*args, **kwargs) -> cute.TensorSSA:
        if cutlass.const_expr(variadic_policy == "first"):
            if cutlass.const_expr(len(args) > 0):
                x = args[0]
                extra_args = args[1:]
            else:
                raise ValueError

        assert cutlass.const_expr(isinstance(x, cute.TensorSSA))
        res = cute.make_fragment(x.shape, x.dtype)
        res.store(x)

        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = fn_scalar(res[i], *extra_args, **kwargs)

        return res.load()

    return _tensorssa_fn


def make_tensorssa_fn_from_scalar_fn_different_dtype(
    fn_scalar: Callable[[object], Scalar],
    dtype: type[cute.Numeric],
    variadic_policy: str | None = None,
) -> Callable[[object], cute.TensorSSA]:
    # https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

    if variadic_policy is None:
        variadic_policy = "first"

    @cute.jit
    def _tensorssa_fn(*args, **kwargs) -> cute.TensorSSA:
        if cutlass.const_expr(variadic_policy == "first"):
            if cutlass.const_expr(len(args) > 0):
                x = args[0]
                extra_args = args[1:]
            else:
                raise ValueError

        assert cutlass.const_expr(isinstance(x, cute.TensorSSA))
        tensor_x = cute.make_fragment(x.shape, x.dtype)
        tensor_y = cute.make_fragment(x.shape, dtype)
        tensor_x.store(x)

        for i in cutlass.range_constexpr(cute.size(x.shape)):
            tensor_y[i] = fn_scalar(tensor_x[i], *extra_args, **kwargs)

        return tensor_y.load()

    return _tensorssa_fn


@dsl_user_op
def _fmin(a: cute.Float32 | float, b: cute.Float32 | float, *, loc=None, ip=None) -> cute.Float32:
    return cute.Float32(
        nvvm.fmin(
            cute.Float32.mlir_type,
            cute.Float32(a).ir_value(loc=loc, ip=ip),
            cute.Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


fmax = make_dispatch_function(
    fn_tensorssa=make_tensorssa_fn_from_scalar_fn(cute.arch.fmax),
    fn_scalar=cute.arch.fmax,
)
fmin = make_dispatch_function(
    fn_tensorssa=make_tensorssa_fn_from_scalar_fn(_fmin),
    fn_scalar=_fmin,
)


def clamp(
    x: cute.TensorSSA,
    min_val: cute.Numeric | float | None = None,
    max_val: cute.Numeric | float | None = None,
) -> cute.TensorSSA:
    if cutlass.const_expr(misc_utils.get_dtype(x) != cute.Float32):
        raise NotImplementedError
    if cutlass.const_expr(
        (min_val is not None) and
        (not isinstance(min_val, float)) and
        (misc_utils.get_dtype(min_val) != cute.Float32)):
        raise NotImplementedError
    if cutlass.const_expr(
        (max_val is not None) and
        (not isinstance(max_val, float)) and
        (misc_utils.get_dtype(max_val) != cute.Float32)):
        raise NotImplementedError
    if cutlass.const_expr(
        isinstance(min_val, cute.TensorSSA) or
        isinstance(max_val, cute.TensorSSA)):
        raise NotImplementedError

    y = x
    if cutlass.const_expr(min_val is not None):
        y = fmax(y, min_val)
    if cutlass.const_expr(max_val is not None):
        y = fmin(y, max_val)

    return y
