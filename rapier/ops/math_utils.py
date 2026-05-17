import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector

from hilt.dtype_utils import get_dtype
from hilt.math_utils import (
    make_dispatch_function,
    make_tensorssa_fn_from_scalar_fn,
)


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
    if cutlass.const_expr(get_dtype(x) != cute.Float32):
        raise NotImplementedError
    if cutlass.const_expr(
        (min_val is not None) and
        (not isinstance(min_val, float)) and
        (get_dtype(min_val) != cute.Float32)):
        raise NotImplementedError
    if cutlass.const_expr(
        (max_val is not None) and
        (not isinstance(max_val, float)) and
        (get_dtype(max_val) != cute.Float32)):
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
