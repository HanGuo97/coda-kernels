# CuTeDSL Type Conversion Guide

Comprehensive reference for type conversion behavior, precision management, and common patterns in CuTeDSL kernel programming.

---

## Overview

Type conversion in CuTeDSL is explicit and follows strict rules that differ from Python conventions. Understanding these rules is essential for:
- Preserving numerical precision (fp16/bf16)
- Avoiding silent type promotions
- Implementing mixed-precision operations correctly

---

## Automatic Type Promotion

### Python Literals → Float32

**Critical Behavior**: Python float literals (`1.0`, `2.5`, etc.) **always** promote operations to `Float32`, regardless of input dtype.

```python
# ❌ WRONG - Silent fp32 promotion
def scale_op(x_ssa):
    return x_ssa * 2.0  # Returns Float32, even if x_ssa is Float16!

# ✅ CORRECT - Explicit dtype preservation
def scale_op(x_ssa):
    input_dtype = x_ssa.element_type
    return (x_ssa * 2.0).to(input_dtype)
```

**Best Practice**: Always capture `element_type` at the start and cast results back.

### Integer Literals

Integer literals (`1`, `2`, etc.) behave differently:
- Operations stay in the input dtype when possible
- May promote to accommodate the result (e.g., int8 × 200 → int16)

---

## Explicit Type Conversion

### Float → Integer: Truncation Behavior

`TensorSSA.to()` **truncates towards zero**, NOT round-to-nearest:

```python
# Truncation examples
values = [1.9, 2.5, -1.9, -2.5]
truncated = float_ssa.to(cute.Int32)  # → [1, 2, -1, -2]
```

#### Implementing Round-to-Nearest

For **positive values only**:
```python
rounded = (float_ssa + 0.5).to(cute.Int32)
```

For **all values** (handles negative correctly):
```python
# Add +0.5 for positive, -0.5 for negative
sign_mask = float_ssa >= 0.0
offset = sign_mask * 0.5 - (1.0 - sign_mask) * 0.5
rounded = (float_ssa + offset).to(cute.Int32)
```

### Integer → Float: Lossless (within range)

Integer to float conversion is straightforward:
```python
float_result = int_ssa.to(cute.Float32)  # Lossless for most int32 values
```

**Note**: Very large integers (>2^24) may lose precision when converting to Float32.

### Float → Float: Precision Changes

```python
# fp32 → fp16: May lose precision, rounds to nearest representable value
fp16_result = fp32_ssa.to(cute.Float16)

# fp16 → fp32: Lossless conversion
fp32_result = fp16_ssa.to(cute.Float32)
```

---

## Advanced Patterns

### Mixed-Precision Arithmetic

For operations prone to error accumulation (reductions, dot products, normalization), use fp32 for intermediate computations even with fp16/bf16 inputs:

```python
input_dtype = tensor_ssa.element_type
if cutlass.const_expr(input_dtype != cute.Float32):
    x_fp32 = tensor_ssa.to(cute.Float32)
    result = compute_in_fp32(x_fp32)
    result = result.to(input_dtype)
else:
    result = compute_in_fp32(tensor_ssa)
```

**Why This Matters**: Low-precision types (fp16/bf16) accumulate errors quickly in iterative operations. PyTorch uses fp32 internally for stability even with fp16 inputs.

**Performance**: This conditional approach only pays upcasting cost for fp16/bf16, preserving full fp32 performance otherwise.

**Note on Type Promotion**: While adding `0.0` (an fp32 literal) implicitly promotes to Float32, explicit `.to(cute.Float32)` casting is strongly preferred for code clarity and maintainability. Use explicit casting throughout your kernels.

### Mixed-Dtype Operations

When input and output have different dtypes, configure memory operations separately:

```python
# Example: Float16 input → Float32 output
input_config = memory_utils.MemoryCopyConfig(
    op="universal",
    dtype=cute.Float16,        # Match input
    num_bits_per_copy=128,
    tiler_mn=tiler_mn,
    layout_tv=tv_layout,
)

output_config = memory_utils.MemoryCopyConfig(
    op="universal",
    dtype=cute.Float32,        # Match output
    num_bits_per_copy=128,     # Same for both
    tiler_mn=tiler_mn,
    layout_tv=tv_layout,
)

# Use input_config for loading, output_config for storing
copy_in = memory_utils.copy(..., config=input_config)
# ... computation ...
memory_utils.copy(..., config=output_config)
```

**Key Points**:
- Each dtype needs its own `MemoryCopyConfig`
- Keep `num_bits_per_copy=128` for both (performance)
- The configs differ only in the `dtype` parameter

---

## Common Patterns

### Pattern 1: Dtype-Agnostic Operations

Operations that work correctly regardless of input dtype:

```python
def safe_op(x_ssa):
    input_dtype = x_ssa.element_type
    # Computation with float literals
    result = x_ssa * 2.0 + 1.0
    return result.to(input_dtype)
```

### Pattern 2: Conditional Precision

Use higher precision only when needed:

```python
def adaptive_op(x_ssa, use_fp32=False):
    input_dtype = x_ssa.element_type

    if use_fp32 or input_dtype == cute.Float32:
        # Force fp32 computation with explicit casting
        x_fp32 = x_ssa.to(cute.Float32)
        result = complex_computation(x_fp32)
    else:
        # Native precision computation
        result = complex_computation(x_ssa)

    return result.to(input_dtype)
```

### Pattern 3: Integer Rounding Helper

Reusable rounding function:

```python
def round_to_int(float_ssa, dtype=cute.Int32):
    """Round float to nearest integer, handling negative values correctly."""
    sign_mask = float_ssa >= 0.0
    offset = sign_mask * 0.5 - (1.0 - sign_mask) * 0.5
    return (float_ssa + offset).to(dtype)
```

---

## Quick Reference

| Operation | Behavior | Solution |
|-----------|----------|----------|
| `x * 1.0` | Promotes to Float32 | Cast back: `.to(input_dtype)` |
| `.to(Int32)` | Truncates towards zero | Add ±0.5 before cast for rounding |
| Mixed dtypes | Requires separate configs | Use different `MemoryCopyConfig` per dtype |
| Force fp32 | Explicit type conversion | `x_fp32 = x_ssa.to(cute.Float32)` |

---

## Summary

**Three Golden Rules**:
1. **Python float literals always promote to fp32** - Capture input dtype and cast back
2. **Float-to-int conversion truncates** - Add 0.5 for rounding (adjust sign for negatives)
3. **Mixed dtypes need separate configs** - One `MemoryCopyConfig` per dtype

Following these rules ensures correct behavior across all supported dtypes (fp32, fp16, bf16, int32, etc.) and prevents silent type promotion bugs.
