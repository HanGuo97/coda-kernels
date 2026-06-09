import ipdb
import triton
import cutlass
import cutlass.cute as cute
from functools import reduce
from typing import cast

Numeric = cutlass.cutlass_dsl.cutlass_arith.ArithValue | cute.Numeric


def static_assert(condition: bool, message: str | None = None, set_trace: bool = False) -> None:
    try:
        if message is None:
            assert cutlass.const_expr(condition)
        else:
            assert cutlass.const_expr(condition), message
    except Exception as e:
        if set_trace:
            ipdb.set_trace()
        raise e


def product(a: tuple | list | int) -> int:
    # https://github.com/NVIDIA/cutlass/blob/main/python/pycute/int_tuple.py
    if isinstance(a, (tuple, list)):
        return reduce(lambda val, elem : val * product(elem), a, 1)
    else:
        return a


def numel(x: cute.Tensor | cute.TensorSSA) -> int:
    if cutlass.const_expr(isinstance(x, cute.Tensor)):
        return cute.size(x)
    else:
        return product(x.shape)


def next_power_of_2(n: int) -> int:
    return triton.next_power_of_2(n)


def greatest_power_of_2_dividing(n: int) -> int:
    if cutlass.const_expr(n == 0):
        raise ValueError
    return n & -n


def static_assert_is_Tensor(x: object) -> cute.Tensor:
    static_assert(isinstance(x, cute.Tensor))
    return cast(cute.Tensor, x)
