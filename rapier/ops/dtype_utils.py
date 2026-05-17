import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import as_numeric
from cutlass.cutlass_dsl import T, dsl_user_op

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector

from hilt.dtype_utils import get_dtype
from hilt.math_utils import (
    make_dispatch_function,
    make_tensorssa_fn_from_scalar_fn,
    make_tensorssa_fn_from_scalar_fn_different_dtype,
)
from rapier.ops.misc_utils import static_assert
from rapier.ops.creation_utils import allocate_tensor_like

Convertable = (
    cute.Tensor |
    cute.TensorSSA |
    cute.Numeric |
    cutlass.cutlass_dsl.cutlass_arith.ArithValue
)


# ------- Scalar Conversion -------

@dsl_user_op
def _cvt_rn_f32_i8(src_f32: cute.Float32, *, loc=None, ip=None) -> cute.Int8:
    i32_val = llvm.inline_asm(
        cute.Int32.mlir_type,
        [cute.Float32(src_f32).ir_value(loc=loc, ip=ip)],
        "cvt.rni.sat.s32.f32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    i8_val = llvm.trunc(
        cute.Int8.mlir_type,
        i32_val,
        llvm.IntegerOverflowFlags.none,
        loc=loc,
        ip=ip,
    )
    return i8_val


# @dsl_user_op
# def _cvt_rn_f16_i8(src_f16: cute.Float16, *, loc=None, ip=None) -> cute.Int8:
#     i32_val = llvm.inline_asm(
#         cute.Int32.mlir_type,
#         [cute.Float16(src_f16).ir_value(loc=loc, ip=ip)],
#         "cvt.rni.sat.s32.f16 $0, $1;",
#         "=r,h",
#         has_side_effects=False,
#         is_align_stack=False,
#         asm_dialect=llvm.AsmDialect.AD_ATT,
#     )
#     i8_val = llvm.trunc(
#         cute.Int8.mlir_type,
#         i32_val,
#         llvm.IntegerOverflowFlags.none,
#         loc=loc,
#         ip=ip,
#     )
#     return i8_val


# @dsl_user_op
# def _cvt_rn_bf16_i8(src_bf16: cute.BFloat16, *, loc=None, ip=None) -> cute.Int8:
#     i32_val = llvm.inline_asm(
#         cute.Int32.mlir_type,
#         [cute.BFloat16(src_bf16).ir_value(loc=loc, ip=ip)],
#         "cvt.rni.sat.s32.bf16 $0, $1;",
#         "=r,h",
#         has_side_effects=False,
#         is_align_stack=False,
#         asm_dialect=llvm.AsmDialect.AD_ATT,
#     )
#     i8_val = llvm.trunc(
#         cute.Int8.mlir_type,
#         i32_val,
#         llvm.IntegerOverflowFlags.none,
#         loc=loc,
#         ip=ip,
#     )
#     return i8_val


@dsl_user_op
def _round_rn_f32(src_f32: cute.Float32, *, loc=None, ip=None) -> cute.Float32:
    f32_val = llvm.inline_asm(
        cute.Float32.mlir_type,
        [cute.Float32(src_f32).ir_value(loc=loc, ip=ip)],
        "cvt.rni.f32.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return f32_val


@dsl_user_op
def f32x2_to_i64(a: cute.Float32, b: cute.Float32, *, loc=None, ip=None) -> cute.Int64:
    src_vec_dtype = ir.VectorType.get(
        [2],
        cute.Float32.mlir_type,
        loc=loc,
        # ip=ip,
    )
    dst_vec_dtype = ir.VectorType.get(
        [1],
        cute.Int64.mlir_type,
        loc=loc,
        # ip=ip,
    )
    src_vec = vector.from_elements(
        src_vec_dtype,
        (
            as_numeric(a).ir_value(loc=loc, ip=ip),
            as_numeric(b).ir_value(loc=loc, ip=ip),
        ),
        loc=loc,
        ip=ip,
    )
    dst_vec = vector.bitcast(
        dst_vec_dtype,
        src_vec,
        loc=loc,
        ip=ip,
    )
    c = cute.Int64(
        vector.extract(
            dst_vec,
            dynamic_position=[],
            static_position=[0],
            loc=loc,
            ip=ip,
        ),
    )
    return c


@dsl_user_op
def i64_to_f32x2(c: cute.Int64, *, loc=None, ip=None) -> tuple[cute.Float32, cute.Float32]:
    src_vec_dtype = ir.VectorType.get(
        [1],
        cute.Int64.mlir_type,
        loc=loc,
        # ip=ip,
    )
    dst_vec_dtype = ir.VectorType.get(
        [2],
        cute.Float32.mlir_type,
        loc=loc,
        # ip=ip,
    )
    src_vec = vector.from_elements(
        src_vec_dtype,
        (
            as_numeric(c).ir_value(loc=loc, ip=ip),
        ),
        loc=loc,
        ip=ip,
    )
    dst_vec = vector.bitcast(
        dst_vec_dtype,
        src_vec,
        loc=loc,
        ip=ip,
    )
    a = cute.Float32(
        vector.extract(
            dst_vec,
            dynamic_position=[],
            static_position=[0],
            loc=loc,
            ip=ip,
        ),
    )
    b = cute.Float32(
        vector.extract(
            dst_vec,
            dynamic_position=[],
            static_position=[1],
            loc=loc,
            ip=ip,
        ),
    )
    return a, b


@dsl_user_op
def pack2(
    a: cute.Numeric,
    b: cute.Numeric,
    src_dtype: type[cute.Numeric],
    dst_dtype: type[cute.Numeric],
    *,
    loc=None,
    ip=None,
) -> cute.Numeric:
    static_assert(get_dtype(a) is src_dtype)
    static_assert(get_dtype(b) is src_dtype)
    static_assert(dst_dtype.width == src_dtype.width * 2)

    src_vec_dtype = ir.VectorType.get(
        [2],
        src_dtype.mlir_type,
        loc=loc,
        # ip=ip,
    )
    dst_vec_dtype = ir.VectorType.get(
        [1],
        dst_dtype.mlir_type,
        loc=loc,
        # ip=ip,
    )
    src_vec = vector.from_elements(
        src_vec_dtype,
        (
            as_numeric(a).ir_value(loc=loc, ip=ip),
            as_numeric(b).ir_value(loc=loc, ip=ip),
        ),
        loc=loc,
        ip=ip,
    )
    dst_vec = vector.bitcast(
        dst_vec_dtype,
        src_vec,
        loc=loc,
        ip=ip,
    )
    c = dst_dtype(
        vector.extract(
            dst_vec,
            dynamic_position=[],
            static_position=[0],
            loc=loc,
            ip=ip,
        ),
    )
    return c


cvt_rn_f32_i8 = make_dispatch_function(
    fn_tensorssa=make_tensorssa_fn_from_scalar_fn_different_dtype(_cvt_rn_f32_i8, dtype=cute.Int8),
    fn_scalar=_cvt_rn_f32_i8,
)

# cvt_rn_f16_i8 = make_dispatch_function(
#     fn_tensorssa=make_tensorssa_fn_from_scalar_fn_different_dtype(_cvt_rn_f16_i8, dtype=cute.Int8),
#     fn_scalar=_cvt_rn_f16_i8,
# )

# cvt_rn_bf16_i8 = make_dispatch_function(
#     fn_tensorssa=make_tensorssa_fn_from_scalar_fn_different_dtype(_cvt_rn_bf16_i8, dtype=cute.Int8),
#     fn_scalar=_cvt_rn_bf16_i8,
# )

round_rn_f32 = make_dispatch_function(
    fn_tensorssa=make_tensorssa_fn_from_scalar_fn(_round_rn_f32),
    fn_scalar=_round_rn_f32,
)


CONVERTERS = {
    (cute.Float32 , cute.Int8, "rn"): cvt_rn_f32_i8,
    # (cute.Float16 , cute.Int8, "rn"): cvt_rn_f16_i8,
    # (cute.BFloat16, cute.Int8, "rn"): cvt_rn_bf16_i8,
}

ROUNDERS = {
    (cute.Float32 , "rn"): round_rn_f32,
}


def convert(
    source: Convertable,
    dtype: type[cute.Numeric],
    style: str | None = None,
) -> Convertable:
    """Convert a value to a different numeric type with specified conversion style.

    This low-level API provides fine-grained control over type conversions, including
    custom PTX-based conversions with specific rounding modes.

    Note:
        For most use cases, prefer `TensorSSA.to(dtype)` which provides a simpler,
        higher-level interface. Use this function when you need explicit control over
        the conversion style or rounding mode.

    Args:
        source: Value to convert. Can be a Tensor, TensorSSA, Numeric scalar, or
            ArithValue from the CUTLASS DSL.
        dtype: Target numeric type (e.g., `cute.Float32`, `cute.Int8`, `cute.BFloat16`).
        style: Conversion style or rounding mode. Common values include:
            - `"rn"`: Round to nearest even

    Returns:
        Convertable: Converted value with the target dtype. The return type matches
            the input type (e.g., TensorSSA input returns TensorSSA output).

    Raises:
        NotImplementedError: If the conversion from source dtype to target dtype with
            the specified style is not supported in the CONVERTERS registry.

    Examples:
        Convert a Float32 tensor to Int8 with round-to-nearest:

        >>> x = cute.TensorSSA(...)  # Float32 tensor
        >>> y = convert(x, cute.Int8, "rn")

        Convert a scalar value:

        >>> scalar = cute.Float32(3.7)
        >>> rounded = convert(scalar, cute.Int8, "rn")  # Result: 4
    """
    if cutlass.const_expr(style is None):
        if cutlass.const_expr(isinstance(source, cute.Tensor)):
            static_assert(source.memspace == cute.AddressSpace.rmem)
            target = allocate_tensor_like(
                source,
                memspace=source.memspace.name,
                smem_allocator=None,
                dtype=dtype,
            )
            target.store(source.load().to(dtype=dtype))
        elif cutlass.const_expr(isinstance(source, cute.TensorSSA | cute.Numeric)):
            target = source.to(dtype=dtype)
        else:
            raise NotImplementedError
    else:
        source_dtype = get_dtype(source)
        key = (source_dtype, dtype, style)
        if cutlass.const_expr(source_dtype == dtype):
            target = source
        elif cutlass.const_expr(key in CONVERTERS.keys()):
            target = CONVERTERS[key](source)
        else:
            raise NotImplementedError(
                f"Unsupported conversion from {source_dtype} "
                f"to {dtype} with {style} style")

    return target


def round(
    source: Convertable,
    style: str,
) -> Convertable:
    """Round a floating-point value to the nearest integer using a specified rounding mode.

    This low-level API rounds floating-point values using PTX-based rounding instructions
    while preserving the original floating-point type. The result is a floating-point
    value representing the rounded integer.

    Args:
        source: Value to round. Can be a Tensor, TensorSSA, Numeric scalar, or ArithValue
            in Float32, Float16, or BFloat16 format.
        style: Rounding mode. Currently supported:
            - `"rn"`: Round to nearest integer (even), also known as banker's rounding.
              Uses PTX's `cvt.rni` instruction implementing IEEE 754 round-to-nearest-even,
              where ties (e.g., 0.5, 1.5, 2.5) are rounded to the nearest even number.

    Returns:
        Convertable: Rounded value with the same dtype as the input. The return type
            matches the input type (e.g., TensorSSA input returns TensorSSA output).

    Raises:
        NotImplementedError: If the source dtype or rounding style is not supported in
            the ROUNDERS registry. Currently only Float32, Float16, and BFloat16 with
            "rn" style are supported.

    Examples:
        Round a Float32 tensor using round-to-nearest-even:

        >>> x = cute.TensorSSA(...)  # Float32 tensor with values [1.5, 2.5, 3.7]
        >>> y = round(x, "rn")  # Result: [2.0, 2.0, 4.0] (ties round to even)

        Round a scalar value:

        >>> scalar = cute.Float16(3.7)
        >>> rounded = round(scalar, "rn")  # Result: 4.0 in Float16 format

    Note:
        - For float-to-integer type conversion with rounding, use `convert(source, dtype, style)`
          which changes the type (e.g., Float32 → Int8). This function maintains the
          floating-point type.
        - Round-to-nearest-even examples: 0.5 → 0.0, 1.5 → 2.0, 2.5 → 2.0, 3.5 → 4.0
    """
    source_dtype = get_dtype(source)
    key = (source_dtype, style)
    if cutlass.const_expr(key in ROUNDERS.keys()):
        target = ROUNDERS[key](source)
    else:
        raise NotImplementedError(
            f"Unsupported rounding for dtype "
            f"{source_dtype} with {style} style")
    return target
