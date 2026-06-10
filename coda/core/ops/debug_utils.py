import cutlass
import cutlass.cute as cute
from hilt.dtype_utils import get_dtype
from coda.core.ops.misc_utils import static_assert


def isnan(
    x: cute.Tensor | cute.TensorSSA | cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(x, cute.Tensor)):
        x = x.load()
    elif cutlass.const_expr(isinstance(x, cutlass.cutlass_dsl.cutlass_arith.ArithValue)):
        x = get_dtype(x)(x)
    return x != x


def check_nan(
    x: cute.Tensor | cute.TensorSSA | cute.Numeric,
    error: bool = False,
) -> cute.Boolean:

    has_nan = isnan(x)
    static_assert(isinstance(has_nan, cute.TensorSSA | cute.Numeric))
    if cutlass.const_expr(isinstance(has_nan, cute.TensorSSA)):
        has_nan = cute.any_(has_nan)

    static_assert(isinstance(has_nan, cute.Boolean))
    if cutlass.const_expr(error):
        # the `error` arguments require special compilation flag
        # in order to enable device-side assertions
        message = (
            "NaN detected in tensor. Common causes:\n"
            "  - Uninitialized memory (use zeros_like/ones_like for initialization)\n"
            "  - Division by zero or invalid arithmetic operations\n"
            "  - Overflow/underflow in numerical computations"
        )
        cute.testing.assert_(
            has_nan == cute.Boolean(False),
            msg=message,
        )

    return has_nan


def is_unequal(
    x: cute.Tensor | cute.TensorSSA | cute.Numeric | cutlass.cutlass_dsl.cutlass_arith.ArithValue,
    v: cute.Numeric | int | float,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(x, cute.Tensor)):
        x = x.load()
    elif cutlass.const_expr(isinstance(x, cutlass.cutlass_dsl.cutlass_arith.ArithValue)):
        x = get_dtype(x)(x)
    return x != v


def check_equal(
    x: cute.Tensor | cute.TensorSSA | cute.Numeric,
    v: cute.Numeric | int | float,
    error: bool = False,
) -> cute.Boolean:

    has_unequal = is_unequal(x, v)
    static_assert(isinstance(has_unequal, cute.TensorSSA | cute.Numeric))
    if cutlass.const_expr(isinstance(has_unequal, cute.TensorSSA)):
        has_unequal = cute.any_(has_unequal)

    static_assert(isinstance(has_unequal, cute.Boolean))
    if cutlass.const_expr(error):
        # the `error` arguments require special compilation flag
        # in order to enable device-side assertions
        message = "Equality check failed in tensor."
        cute.testing.assert_(
            has_unequal == cute.Boolean(False),
            msg=message,
        )

    return has_unequal
