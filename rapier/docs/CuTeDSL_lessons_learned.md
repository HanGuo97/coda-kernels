# CuTeDSL Kernel Programming Guide

## Overview

Reference collection of practical patterns and common pitfalls from implementing custom CuTeDSL kernels. Covers elementwise operations, reductions, TMA pipelines, and epilogue composition.

**Quick navigation:**
- **Debugging?** → [Common Pitfalls](#common-pitfalls)
- **Type errors?** → [Type System Overview](#type-system-overview) + [Type Conversion Guide](./CuTeDSL_type_conversion.md)
- **Memory patterns?** → [Memory Operations](#memory-operations)
- **Reductions?** → [Reduction Operations](#reduction-operations)
- **GEMM epilogues?** → [Epilogue Composition Patterns](#epilogue-composition-patterns)
- **Examples?** → [Learning from Reference Implementations](#learning-from-reference-implementations)

## Table of Contents
1. [Type System Overview](#type-system-overview)
2. [Memory Operations](#memory-operations)
3. [Kernel Infrastructure](#kernel-infrastructure)
4. [Mathematical Operations](#mathematical-operations)
5. [Reduction Operations](#reduction-operations)
6. [Common Pitfalls](#common-pitfalls)
7. [Epilogue Composition Patterns](#epilogue-composition-patterns)
8. [Learning from Reference Implementations](#learning-from-reference-implementations)

**Note**: For comprehensive information about type conversion, precision management, and casting behavior, see [CuTeDSL Type Conversion Guide](./CuTeDSL_type_conversion.md)

---

## Type System Overview

CuTeDSL distinguishes between two fundamental types:

**`cute.Tensor`** - Memory reference (pointer + layout metadata)
- Cannot be used directly in arithmetic operations
- Analogous to a pointer in C/C++

**`cute.TensorSSA`** - Actual values available for computation
- Supports arithmetic operations, comparisons, and math functions
- Analogous to a dereferenced value in C/C++

```python
# Load: Tensor → TensorSSA (dereference to get values)
values = tensor_ref.load()

# Store: TensorSSA → Tensor (write values to memory)
tensor_ref.store(computed_values)
```

### Type Conversion Rules

See [CuTeDSL Type Conversion Guide](./CuTeDSL_type_conversion.md) for comprehensive details. Essential rules:

1. **Python float literals promote to fp32** - Always cast back: `result.to(input_dtype)`
2. **`.to()` truncates when converting to integers** - Add 0.5 before casting for rounding
3. **Mixed-dtype operations need separate memory configs** - Input/output may have different dtypes

```python
# Standard pattern for preserving input dtype
def _compute_op(tensor_ssa: cute.TensorSSA) -> cute.TensorSSA:
    input_dtype = tensor_ssa.element_type
    result = tensor_ssa * 2.0 + 1.0  # Float literals promote to fp32
    return result.to(input_dtype)     # Cast back to preserve dtype
```

---

## Memory Operations

### Vectorized Memory Access (128-bit)
**Performance Critical**: Always use 128-bit memory operations to achieve optimal memory bandwidth utilization:

```python
config = memory_utils.MemoryCopyConfig(
    op="universal",
    dtype=tensor.element_type,
    num_bits_per_copy=128,  # Critical for performance
    tiler_mn=tiler_mn,
    layout_tv=tv_layout,
)
```

### Standard Memory Copy Pattern
The canonical pattern for elementwise operations follows these steps:

```python
# Step 1: Load from global memory to registers
copy_outputs = memory_utils.copy(
    src=g_input, dst="rmem", crd=coords, shape=input_tensor.shape,
    config=config, thread_index=tidx, smem_allocator=allocator
)
t_input_reg = copy_outputs.dst_thread

# Step 2: Dereference to TensorSSA and perform computation
input_val = t_input_reg.load()  # Tensor → TensorSSA
result = operation(input_val)   # Operate on TensorSSA values

# Step 3: Store result back to Tensor
t_output_reg.store(result)  # TensorSSA → Tensor

# Step 4: Write from registers to global memory
memory_utils.copy(
    src=t_output_reg, dst=g_output, crd=coords, shape=output_tensor.shape,
    config=config, thread_index=tidx, smem_allocator=allocator
)
```

---

## Kernel Infrastructure

### Unary Operations
For single-input operations, leverage helper infrastructure to reduce boilerplate:

```python
# Define the operation function at module level
def _scale_op(tensor_ssa: cute.TensorSSA) -> cute.TensorSSA:
    input_dtype = tensor_ssa.element_type
    result = tensor_ssa * 2.0  # Example: scale by 2
    return result.to(input_dtype)

# Use the operation in a wrapper function
def scale_2d(x: torch.Tensor) -> torch.Tensor:
    return elementwise_op(_scale_op, x.detach())
```

**Critical Requirement**: Define operation functions at module level, not inside other functions. Defining them inside functions creates new closures with different identities on each call, causing cache misses and repeated recompilation.

### Binary Operations
Multi-input operations require custom kernel implementations:

```python
@cute.kernel
def _binary_kernel(op: cutlass.Constexpr, input_a, input_b, output, tiler_mn, tv_layout):
    tidx, bidx, bidy = cute.arch.thread_idx()[0], *cute.arch.block_idx()[:2]
    allocator = cutlass.utils.SmemAllocator()

    # Tile and load both inputs
    g_a, g_b, g_out = [
        cute.local_tile(m, tiler_mn, (bidx, bidy))
        for m in (input_a, input_b, output)
    ]
    coords = cute.local_tile(
        cute.make_identity_tensor(input_a.shape), tiler_mn, (bidx, bidy)
    )

    config = memory_utils.MemoryCopyConfig(...)
    copy_a = memory_utils.copy(src=g_a, dst="rmem", crd=coords, ...)
    copy_b = memory_utils.copy(src=g_b, dst="rmem", crd=coords, ...)

    # Compute and store result
    result = op(copy_a.dst_thread.load(), copy_b.dst_thread.load())
    t_output_reg.store(result)
    memory_utils.copy(src=t_output_reg, dst=g_out, ...)
```

### Compilation Caching
Compilation is expensive. Implement caching to avoid recompiling kernels with identical configurations:

```python
def binary_op_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Cache by: operation + row size + dtype
    compile_key = (op, x.shape[1], x.dtype)

    if compile_key not in binary_op_2d.compile_cache:
        binary_op_2d.compile_cache[compile_key] = cute.compile(
            kernel_func, ...
        )

    compiled_kernel = binary_op_2d.compile_cache[compile_key]
    return compiled_kernel(x, y, stream)

# Initialize cache as a function attribute
binary_op_2d.compile_cache = {}
```

**Cache Key Design**: Include the operation, row size (N), and dtype. Exclude batch size (M) if it varies dynamically—tiling handles different batch sizes with the same compiled kernel.

---

## Mathematical Operations

### Precision Considerations
When working with low-precision inputs (fp16, bfloat16, int8), convert to higher precision before computation to maintain numerical accuracy:

```python
def _compute_op(tensor_ssa: cute.TensorSSA) -> cute.TensorSSA:
    input_dtype = tensor_ssa.element_type

    # Upcast low-precision inputs to fp32
    if cutlass.const_expr(input_dtype in (cute.Float16, cute.BFloat16)):
        tensor_ssa = tensor_ssa.to(cute.Float32)

    result = complex_computation(tensor_ssa)

    # Downcast back to original dtype
    return result.to(input_dtype)
```

**Critical for:**
- Accumulation-heavy operations (reductions, dot products)
- Chained arithmetic where errors compound
- Operations with small values prone to underflow

### Built-in Functions

CuTeDSL provides mathematical primitives through the `cute.math` module:

```python
cute.math.exp(x, fastmath=True)  # Exponential (fastmath=True recommended for performance)
cute.math.erf(x)                 # Error function
cute.math.tanh(x)                # Hyperbolic tangent
cute.math.log(x)                 # Natural logarithm
cute.math.rsqrt(x)               # Reciprocal square root (1/√x)
```

**Available Functions**: The module provides `exp`, `erf`, `tanh`, `log`, and `rsqrt`. Functions like `maximum`, `minimum`, `floor`, `ceil`, and `sqrt` are not currently available.

**Computing Square Root**: Since `cute.math.sqrt` is not available, use one of these alternatives:
- `cute.math.rsqrt(x) * x` (preferred - single multiply instruction)
- `1.0 / cute.math.rsqrt(x)` (alternative - requires division)

**Computing Division**: For division operations, prefer approximate reciprocal multiplication:
- `a * cute.arch.rcp_approx(b)` (preferred - fast approximate reciprocal multiply)
- `a / b` (alternative - slower hardware division instruction)

The approximate reciprocal approach uses a single multiply instruction and is significantly faster than direct division while maintaining sufficient precision for most deep learning applications.

### Implementing Missing Operations

#### Comparison-Based Operations
Use boolean masks to implement conditional operations:

```python
# Thresholding: keep positive values
result = input_ssa * (input_ssa > 0.0)

# Element-wise maximum of two tensors
mask = a_ssa > b_ssa
result = a_ssa * mask + b_ssa * (1.0 - mask)

# Clamping values between [min_val, max_val]
above_min = input_ssa > min_val
clamped_lower = input_ssa * above_min + min_val * (1.0 - above_min)
below_max = clamped_lower < max_val
result = clamped_lower * below_max + max_val * (1.0 - below_max)
```

#### Mathematical Expressions
Implement operations using available primitives:

```python
# Example: Sigmoid using exponential
sigmoid = input_ssa * (1.0 / (1.0 + cute.math.exp(-input_ssa, fastmath=True)))

# Example: GELU approximation using error function
gelu = 0.5 * input_ssa * (1.0 + cute.math.erf(input_ssa / 1.414))
```

---

## Reduction Operations

Row-wise reductions aggregate values across rows and require different patterns than elementwise operations. Operations like LayerNorm, RMSNorm, and Softmax typically need multiple sequential reduction passes to compute statistics and normalize values.

### Hierarchical Reduction Pattern

Reductions in CuTeDSL follow a hierarchical pattern: thread-local → warp-level → block-level. The `reduction_utils` module handles all synchronization automatically:

```python
from rapier.ops import reduction_utils

# Configure thread shape for reduction
thread_shape = (
    tv_layout.shape[0][1],  # threads_per_col
    tv_layout.shape[0][0],  # threads_per_row
)

# Perform reduction (supported ops: add, max, min, mul)
(result,), reduction_buffer = reduction_utils.reduce(
    (tensor_ssa,),           # Tuple of tensors to reduce
    op="add",                # Reduction operation
    thread_shape=thread_shape,
    smem_allocator=allocator,
    reduction_buffer=None,   # First pass: allocate buffer
)
```

**Key Features**:
- Returns both reduced value(s) and reusable reduction buffer
- Supports simultaneous multi-tensor reduction: `(r1, r2), buf = reduce((t1, t2), ...)`
- Handles warp-level synchronization and shared memory coordination automatically

### Multi-Pass Reductions

Operations requiring multiple sequential reductions should reuse the reduction buffer to conserve shared memory:

```python
# First reduction pass
(reduced_x,), reduction_buffer = reduction_utils.reduce(
    (x,),
    op="add",
    thread_shape=thread_shape,
    smem_allocator=allocator,
    reduction_buffer=None,  # Allocate buffer
)
intermediate = compute_statistic(reduced_x)

# Second reduction pass (reuse buffer)
transformed = apply_transformation(x, intermediate)
(reduced_transformed,), _ = reduction_utils.reduce(
    (transformed,),
    op="add",
    thread_shape=thread_shape,
    smem_allocator=allocator,
    reduction_buffer=reduction_buffer,  # Reuse existing buffer
)
final_result = compute_final_value(reduced_transformed)
```

**Buffer Reuse Benefits**: Minimizes shared memory allocation overhead and reduces register pressure.

### Memory Management for Large Tensors

When row dimensions exceed available shared memory, implement adaptive reload strategies:

```python
# Determine reload strategy based on tensor dimensions
_, N = x.shape

# Example thresholds - tune for your specific workload
if cutlass.const_expr(N <= SMALL_THRESHOLD):
    reload = None        # Keep data in registers
elif cutlass.const_expr(N <= MEDIUM_THRESHOLD):
    reload = "smem"      # Reload from shared memory between passes
else:
    reload = "gmem"      # Reload from global memory

# Initial load based on strategy
if cutlass.const_expr(reload != "gmem"):
    # Small/medium: use shared memory staging
    tXsX = global_to_shared_copy(gX)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    tXrX = shared_to_register_copy(tXsX)
else:
    # Large: load directly to registers
    tXrX = global_to_register_copy(gX)
    tXsX = None

# Perform first reduction...
x = tXrX.load().to(cute.Float32)
(reduced_val,), buf = reduction_utils.reduce((x,), op="add", ...)

# Reload for subsequent operations if needed
if cutlass.const_expr(reload == "smem"):
    tXrX = shared_to_register_copy(tXsX)
elif cutlass.const_expr(reload == "gmem"):
    tXrX = global_to_register_copy(gX)

# Use reloaded data for second pass
x = tXrX.load().to(cute.Float32)
```

**Memory Strategy Selection**:
- **Registers-only**: Best for small dimensions where data fits comfortably in registers
- **Shared memory**: Effective when registers spill but shared memory can accommodate the data
- **Global memory**: Necessary when shared memory limits are exceeded (typically ~200 KB per SM)

**Critical**: Optimal thresholds depend on dtype, register pressure, and GPU architecture. Profile your specific workload to determine appropriate values.

### Numerical Stability

Always compute reductions in FP32, even with FP16/BF16 inputs, to prevent accumulation errors:

```python
# Upcast to FP32 before reduction
x_fp32 = tensor_ssa.to(cute.Float32)
(result,), buf = reduction_utils.reduce((x_fp32,), op="add", ...)
```

### Random Access After Reduction

Some operations require accessing specific tensor elements after computing reductions. This pattern combines global reduction results with individual element lookups:

```python
# After computing row-wise reductions...
(row_max,), buf = reduction_utils.reduce((values_fp32,), op="max", ...)
(row_sum,), _ = reduction_utils.reduce((transformed_values,), op="add", ...)
statistic = compute_from_reduction(row_max, row_sum)

# Only one thread per row performs the gather and final computation
if col_index == 0:
    # Load index from separate tensor (e.g., indices, masks, labels)
    element_idx_i64 = mIndices[row_index]
    element_idx = cute.Int32(element_idx_i64)  # Cast i64 to i32 for indexing

    # Random access to global tensor at specific location
    selected_value = mTensor[row_index, element_idx]
    selected_value_fp32 = cute.Float32(selected_value)

    # Compute final result combining reduced statistics with selected element
    result = compute_final(selected_value_fp32, statistic)
    mOutput[row_index] = mOutput.element_type(result)
```

**Key Pattern**: Thread-conditional execution (`if col_index == 0:`) ensures only one thread per row writes the final result, avoiding race conditions. The reduced values (computed by all threads) are available to the writing thread.

**Type Requirement**: Tensor indexing requires `cute.Int32` indices. Cast `torch.int64` (i64) values to `cute.Int32` before using them as indices.

**Use Cases**: Loss functions (cross-entropy, focal loss), sparse operations, conditional computations, gather/scatter patterns.

### Performance Best Practices

1. **Reuse Reduction Buffers**: Pass the buffer from first reduction to subsequent passes
2. **Use Compile-Time Conditionals**: Wrap reload strategies in `cutlass.const_expr()` to eliminate runtime overhead
3. **Leverage Warp Intrinsics**: Reduction utilities automatically use warp shuffle operations for efficiency
4. **Optimal Thread Configuration**: Use layout utilities to determine optimal thread counts

---

## Common Pitfalls

### 1. Unintended Type Promotion
**Symptom**: Receiving fp32 results when working with fp16/bf16 inputs.

**Root Cause**: Python float literals automatically promote to Float32.

**Solution**: Explicitly cast results back to the input dtype using `result.to(input_dtype)`.

**See**: [Type Conversion Guide](./CuTeDSL_type_conversion.md#python-literal-type-promotion) for detailed explanation.

### 2. Confusion Between Tensor and TensorSSA
**Symptom**: `AttributeError: 'Tensor' object has no attribute...`

**Root Cause**: Attempting arithmetic operations on `cute.Tensor` instead of `cute.TensorSSA`.

**Solution**: Call `.load()` to dereference the Tensor before performing operations.

### 3. Compile-Time vs Runtime Branching
**Symptom**: Errors using Python `if` statements with runtime-dependent conditions.

**Root Cause**: CuTeDSL requires compile-time constant conditions to generate efficient PTX code. The branch outcome must be known at compilation.

**Solution**: Wrap conditions in `cutlass.const_expr()` for compile-time evaluation:

```python
# ❌ BAD - Runtime condition
if tensor_ssa.element_type == cute.Float32:
    result = compute_fp32(tensor_ssa)

# ✅ GOOD - Compile-time evaluation
if cutlass.const_expr(tensor_ssa.element_type == cute.Float32):
    result = compute_fp32(tensor_ssa)
```

**Use Cases**: Dtype-specific optimizations, algorithm selection based on compile-time parameters, conditional feature enablement.

### 4. Mixed-Precision Arithmetic
**Symptom**: Numerical errors, overflow, or underflow with fp16/bfloat16 inputs, especially in accumulation-heavy operations.

**Root Cause**: Low-precision formats have limited dynamic range and mantissa precision. Errors accumulate quickly in multi-step computations.

**Solution**: Convert inputs to higher precision (typically fp32) before performing operations, then convert back:

```python
# Pattern for precision-sensitive operations
def _precision_safe_op(tensor_ssa: cute.TensorSSA) -> cute.TensorSSA:
    input_dtype = tensor_ssa.element_type

    # Upcast low-precision inputs
    if cutlass.const_expr(input_dtype in (cute.Float16, cute.BFloat16)):
        compute_val = tensor_ssa.to(cute.Float32)
    else:
        compute_val = tensor_ssa

    # Perform computation in higher precision
    result = compute_val * scale + bias  # Example operation

    # Cast back to original precision
    return result.to(input_dtype)
```

**When to Use**:
- Operations with multiple arithmetic steps
- Accumulations or reductions
- Operations involving very small or very large values
- When numerical accuracy is critical

### 5. Vector Size Calculation for Mixed Dtypes
**Symptom**: Shape mismatch errors or inefficient memory access patterns when working with tensors of different dtypes (e.g., fp16 input, fp32 scale, int8 output).

**Root Cause**: The vector size must accommodate ALL dtypes involved in the operation, not just the input or output. Using the wrong calculation leads to misaligned memory access or compilation errors.

**Solution**: Calculate vector size based on the maximum element width across ALL tensor arguments:

```python
# ❌ BAD - Only considers some tensors
vector_size = cutlass.const_expr(128 // max(
    tensor1.element_type.width,
    tensor2.element_type.width,
))

# ✅ GOOD - Considers all tensors in the operation
# Note: max() only accepts 2 arguments, so nest from right to left
vector_size = cutlass.const_expr(128 // max(
    tensor1.element_type.width,
    max(tensor2.element_type.width,
        max(tensor3.element_type.width,
            tensor4.element_type.width)),
))
```

**Important Notes**:
- CuTeDSL's `max()` operator only accepts exactly two arguments (unlike Python's `max()`)
- For more than two values, nest the calls from right to left: `max(a, max(b, max(c, d)))`
- Target 128-bit vectorized memory operations for optimal bandwidth

### 6. Repeated Kernel Recompilation
**Symptom**: Kernel recompiles on every function call, causing performance degradation.

**Root Cause**: Defining operation functions inside other functions creates new closures with different identities on each call.

**Solution**: Define operation functions at module level.

```python
# ❌ BAD - Creates new closure on each call
def compute(x):
    def _op(ssa):
        return ssa
    return elementwise_op(_op, x)

# ✅ GOOD - Stable function identity
def _op(ssa):
    return ssa

def compute(x):
    return elementwise_op(_op, x)
```

### 7. Numerical Accuracy Issues
**Symptom**: Numerical divergence between implementations.

**Root Cause**: Using approximations instead of exact formulas.

**Solution**: Prefer exact mathematical expressions when implementing operations.

### 8. Slow Transcendental Functions
**Symptom**: Poor performance on operations like exponential.

**Root Cause**: Not enabling fast math optimizations.

**Solution**: Pass `fastmath=True` to functions like `cute.math.exp(x, fastmath=True)`.

### 9. Math Function Location
**Symptom**: `AttributeError: module 'cutlass.cute.arch' has no attribute 'rsqrt'`

**Root Cause**: Mathematical functions are in `cute.math`, not `cute.arch`. Common confusion because `cute.arch` contains other intrinsics.

**Solution**:
```python
# ❌ WRONG - Math functions are not in cute.arch
rms = cute.arch.rsqrt(x)

# ✅ CORRECT - Use cute.math for mathematical functions
rms = cute.math.rsqrt(x)
```

**Module Organization**:
- `cute.math`: Mathematical functions (exp, log, rsqrt, erf, tanh)
- `cute.arch`: Hardware intrinsics (shuffle, barriers, warp operations)

### 10. Uninitialized Tensor Values (NaN Propagation)
**Symptom**: Kernel produces NaN despite valid inputs.

**Root Cause**: `allocate_tensor_like()` and `empty_like()` allocate uninitialized memory containing garbage values. NaN propagates through all operations (`nan * 0.0 = nan`), contaminating results.

**Solution**: Use initialized creation functions when reading before writing:

```python
# ❌ WRONG - Uninitialized memory
acc = creation_utils.empty_like(...)
value = acc.load()  # Garbage/NaN

# ✅ CORRECT - Initialized memory
acc = creation_utils.zeros_like(...)  # or ones_like(), full_like()
value = acc.load()  # Defined value
```

**Debugging**: Use `compute-sanitizer --tool=initcheck` to detect uninitialized reads. See [Compute Sanitizer Guide](./compute_sanitizer.md).

**Guidelines**:
- Use `empty_like()` / `allocate_tensor_like()` only when overwriting all elements before reading
- Use `zeros_like()` / `ones_like()` / `full_like()` when initializing accumulators or reading before writing

### 11. Integer Index Type Mismatch
**Symptom**: Compilation error with message `builtin.unrealized_conversion_cast` or `i64 -> i32` type mismatch when using PyTorch integer tensors for indexing.

**Root Cause**: CuTeDSL tensor indexing requires `cute.Int32` indices, but PyTorch integer tensors use `torch.int64` (i64). Direct use of i64 values causes LLVM translation failures.

**Solution**: Explicitly cast integer indices to `cute.Int32` before using them in tensor indexing operations:

```python
# ❌ WRONG - Direct use of i64 value
idx = mIndices[row_index]  # Returns i64
value = mTensor[row_index, idx]  # Compilation error

# ✅ CORRECT - Cast to i32 first
idx_i64 = mIndices[row_index]
idx = cute.Int32(idx_i64)  # Explicit cast
value = mTensor[row_index, idx]  # Works correctly
```

**When This Occurs**: Any operation that uses integer tensors for indexing, such as:
- Gather/scatter operations (selecting elements based on index tensors)
- Sparse matrix operations (indexing with row/column indices)
- Loss functions with label tensors (accessing elements by class/label indices)
- Advanced indexing patterns where indices come from other tensors
- Conditional operations requiring dynamic lookups

**Note**: This only applies to tensor indexing operations. Loop indices and other compile-time integer values don't require explicit casting.

### 12. TMA Shared Memory Alignment Requirements
**Symptom**: Kernel executes successfully but later CUDA operations fail with "misaligned address" errors, often in unrelated code.

**Root Cause**: TMA (Tensor Memory Accelerator) operations require shared memory buffers aligned to **at least 128 bytes**. Insufficient alignment causes silent memory corruption.

**Solution**: Use 128-byte minimum alignment (1024 bytes recommended) for TMA-related shared memory:

```python
# ❌ WRONG - Insufficient alignment
@cute.struct
class SharedStorage:
    smem_buffer: cute.struct.Align[
        cute.struct.MemRange[dtype, size],
        16,  # TOO SMALL for TMA
    ]

# ✅ CORRECT - Proper TMA alignment
@cute.struct
class SharedStorage:
    smem_buffer: cute.struct.Align[
        cute.struct.MemRange[dtype, size],
        1024,  # 128 minimum, 1024 recommended
    ]
```

**Note**: In GEMM examples, using a `buffer_align_bytes` constant (typically 1024) is a good default for all TMA buffers.

**Alignment Requirements**:
- **≥128 bytes required**: TMA copy operations, mainloop/epilogue buffers
- **16 bytes sufficient**: Non-TMA global loads, register operations, reduction buffers, scalars

**Debugging**:
1. Check TMA buffer alignment first when encountering mysterious corruption
2. Temporarily replace TMA with global loads to isolate the issue
3. Verify all TMA buffers use ≥128-byte alignment consistently

### 13. Missing Synchronization After Async Memory Operations

**Symptom**: Kernel produces incorrect results (zeros, garbage, or wrong values) despite valid inputs. May appear intermittent or hardware-dependent.

**Root Cause**: Asynchronous memory operations (e.g., `cp.async`, TMA copies) execute in the background. Reading destination memory before the operation completes accesses uninitialized or partially-written data.

**Solution**: Always synchronize after async operations before consuming the data:

```python
# Step 1: Initiate async copy (e.g., global-to-shared)
async_copy_operation(src=source, dst=destination, ...)

# Step 2: CRITICAL - Synchronize before accessing destination
cute.arch.cp_async_commit_group()    # Commit pending async group
cute.arch.cp_async_wait_group(0)     # Wait for all committed groups
barrier.arrive_and_wait()             # Thread-level synchronization

# Step 3: Now safe to read from destination
data = load_from_destination(destination)
```

**Synchronization Primitives**:
- `cp_async_commit_group()`: Groups pending async operations for collective waiting
- `cp_async_wait_group(N)`: Waits until at most N async groups remain pending (use 0 to wait for all)
- `arrive_and_wait()`: Ensures all threads reach the barrier before proceeding

**When This Applies**:
- Async copies between memory spaces (global↔shared, global↔register)
- TMA operations without implicit barriers
- Multi-stage pipelines with producer/consumer patterns
- Any operation using `cp.async` instructions

**Debugging**:
1. Check for missing barriers between async operations and subsequent reads
2. Use `compute-sanitizer --tool=racecheck` to detect memory access races
3. Review working implementations to verify barrier placement patterns

**Best Practice**: Refer to epilogue implementations in `rapier/ops/epilogue_utils.py` for canonical synchronization examples.

### 14. TMA Pipeline Producer-Consumer Protocol

**Symptom**: Kernel hangs or times out with no errors. Affects multi-stage TMA pipelines with producer-consumer patterns.

**Root Cause**: TMA pipelines require strict synchronization between asynchronous producer (loads data) and consumer (processes data) through acquire-commit-release barriers. Missing or misordered calls cause deadlocks.

**Producer Protocol** - Execute on every TMA load (initial prefetch and steady-state):

```python
# All producer loads follow the same pattern
if is_tma_warp:
    pipeline.producer_acquire(state)  # Wait for available stage slot
    tma_bar_ptr = pipeline.producer_get_barrier(state)
    cute.copy(atom, src, dst, tma_bar_ptr=tma_bar_ptr)
    pipeline.producer_commit(state)   # Signal transfer initiated
state.advance()
```

**Consumer Protocol** - Execute when consuming loaded data:

```python
pipeline.consumer_wait(state)          # Wait for data arrival
process_data()                         # Compute using loaded data
cute.arch.fence_proxy(                 # Memory fence ensures visibility
    kind=cute.arch.ProxyKind.async_shared,
    space=cute.arch.SharedSpace.shared_cta,
)
cute.arch.sync_warp()                  # Synchronize warp threads
with cute.arch.elect_one():
    pipeline.consumer_release(state)   # Signal stage available for reuse
state.advance()
```

**Pipeline Lifecycle**:

```python
# Initial prefetch: Fill pipeline stages (empty slots → immediately available)
for idx in range(min(num_stages, total_tiles)):
    if is_tma_warp:
        pipeline.producer_acquire(state)  # Succeeds immediately (empty pipeline)
        tma_bar = pipeline.producer_get_barrier(state)
        cute.copy(atom, src[idx], dst[state.index], tma_bar_ptr=tma_bar)
        pipeline.producer_commit(state)
    state.advance()

# Steady-state: Produce next tile while consuming current tile
for idx in range(num_stages, total_tiles):
    # Consumer processes previously loaded data
    pipeline.consumer_wait(state)
    compute(data[state.index])
    fence_and_sync()
    with cute.arch.elect_one():
        pipeline.consumer_release(state)
    state.advance()

    # Producer loads next tile (waits if stage still occupied)
    if is_tma_warp:
        pipeline.producer_acquire(state)  # Blocks until consumer releases
        tma_bar = pipeline.producer_get_barrier(state)
        cute.copy(atom, src[idx], dst[state.index], tma_bar_ptr=tma_bar)
        pipeline.producer_commit(state)
    state.advance()
```

**Critical Requirements**:
1. **Always use `producer_acquire()`** - Even initial prefetch needs acquire to maintain protocol invariants
2. **Fence before `consumer_release()`** - Ensures shared memory reads complete before signaling stage available
3. **Symmetric state advancement** - Producer and consumer must advance states in lockstep
4. **Elect-one for release** - Only one thread per CTA should call `consumer_release()`

**Common Mistakes**:
1. Missing `producer_acquire()` → overwrites data still being consumed
2. Missing `consumer_wait()` → reads uninitialized or in-flight data
3. Missing `consumer_release()` → producer deadlocks waiting for free stages
4. Missing fence/sync before release → shared memory race conditions
5. Releasing outside `elect_one()` → multiple threads signal release, corrupting pipeline state

**Debugging**:
- Kernel hangs → verify all acquire/commit/wait/release calls present and properly ordered
- Use `timeout 300 python script.py` to detect hangs during development
- Check producer/consumer states advance symmetrically (same number of advances)
- Verify fence and warp sync precede every `consumer_release()`

**Reference**: See epilogue implementations in `rapier/ops/epilogue_utils.py` for complete working examples including producer initialization, steady-state loading, and consumer processing patterns.


### 15. Shared Memory Overflow in Epilogue Composition

**Symptom**: Compilation fails with "Insufficient shared memory" error when composing multiple epilogue operations, typically exceeding H100's 227KB limit by 1-2KB.

**Root Cause**: Epilogue operations allocate shared memory buffers proportional to epilogue thread count. Default GEMM configurations use 8 warp groups (~256 threads) in the epilogue, creating per-operation overhead (e.g., 256 threads × 2 fp16 elements = 1KB). With 3+ composed operations loading auxiliary tensors, this overhead accumulates beyond hardware limits.

**Solution**: Enable warp specialization with `pingpong=True`:

```python
result = gemm_epilogue(..., pingpong=True)
```

This reduces epilogue thread count from ~256 to ~128 threads, cutting per-operation overhead by 50% (e.g., 1.5KB → 0.75KB). Also improves performance 5-15% by overlapping mainloop and epilogue execution.

**When to Use**:
- Prefer `pingpong=True` for complex epilogue compositions (3+ operations)
- Critical when loading auxiliary tensors (indices, scales, biases)—each adds 1-2KB overhead
- Safe for all GEMM configurations, often improves performance

**Alternatives** (if pingpong insufficient):
1. Load auxiliary data directly from global memory (trades bandwidth for capacity)
2. Split complex epilogues into multiple kernel launches (loses fusion benefits)
3. Reduce tile sizes (may impact occupancy)


---

## Epilogue Composition Patterns

Composable epilogue operations enable fusing multiple operations into GEMM kernels. This section documents critical patterns for implementing custom epilogue operations.

### Broadcasting Per-Row Values Across Columns

**Problem**: Computing per-row values and broadcasting them across columns for element-wise operations with GEMM output.

**❌ Register Memory Fails**

Register memory (`rmem`) is thread-local. Values written by one thread are invisible to others, even with stride-0 broadcast layouts.

**✅ Use Shared Memory**

```python
# 1. Allocate shared memory (one value per row)
@cute.jit
def get_smem_struct(...):
    @cute.struct
    class SharedStorage(EpilogueSharedStorage):
        sValues: cute.struct.Align[cute.struct.MemRange[dtype, tile_M], 16]
    return SharedStorage

# 2. Compute and write to shared memory (distributed across threads)
for m in cutlass.range(tile_M):
    if m % epi_num_threads == tidx:
        sValues[m] = compute_value(m)

# 3. Synchronize before reading
epi_barrier.arrive_and_wait()

# 4. Create broadcast view with stride-0 layout
broadcast_layout = cute.make_layout(shape=(tile_M, tile_N), stride=(1, 0))
broadcast_view = cute.make_tensor(iterator=sValues.iterator, layout=broadcast_layout)
```

**Key**: Shared memory + barrier synchronization + stride-0 layout for broadcasting.

### Direct vs Staged Global Memory Loads

**Direct Loads** (simpler, good for small auxiliary tensors):
```python
for m in cutlass.range(tile_M):
    if m % epi_num_threads == tidx:
        value = gAuxTensor[m].to(compute_dtype)
        result = compute(value)
        sShared[m] = result
```

**Staged Loads** (better coalescing, good for larger tensors):
```python
g2s_copy(src=gAuxTensor, dst=sAuxTensor, ...)
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)
epi_barrier.arrive_and_wait()

for m in cutlass.range(tile_M):
    value = sAuxTensor[m].to(compute_dtype)
    result = compute(value)
```

**Use direct loads for**: Small data (<10KB/tile), single read, different data per thread
**Use staged loads for**: Large data, multiple reads, shared access patterns

### Composing Multiple Operations

```python
from rapier.ops.epilogue_composite_utils import EVTList

def epilogue_visitor_tree_cls(acc_dtype, tile_shape_mnk, buffer_align_bytes):
    evt_a = EVTOperationA(...)
    evt_b = EVTOperationB(...)
    return EVTList(evts=[evt_a, evt_b])  # Sequential execution

epi_args = EVTList.EpilogueArguments(
    _list=[EVTOperationA.EpilogueArguments(...), EVTOperationB.EpilogueArguments(...)]
)
```

**Execution order**: `consumer_visit` runs sequentially, `consumer_begin/end` run in parallel. Total shared memory is sum of all operations.

### Avoiding Unnecessary Warp Reductions

**Don't reduce values already broadcast from shared memory**. All threads have identical values after broadcast—reduction would corrupt them (e.g., sum would multiply by thread count).

**Use warp reduction when**:
- Computing statistics from GEMM output (each thread has different values)
- Aggregating partial results from tensor cores

**Don't use warp reduction when**:
- Values already computed once per row in shared memory
- All threads read the same broadcast value

### Ensuring Correct Thread Writes

```python
# Use coordinate tensors to identify responsible threads
tCoords = partition_for_epilogue(make_identity_tensor(shape))
tCoords_filtered = select_nonzero_stride_modes(tCoords, layout)

# Check thread responsibility before writing
if tCoords_filtered[0][1] == 0:  # Column index check for row-wise writes
    for m in range(size):
        if tCoords_filtered[m][0] < limit:  # Bounds check
            gOutput[tCoords_filtered[m][0]] = value[m]
```

### Performance Considerations for Epilogue Operations

Efficient epilogue operations require choosing the right memory staging strategy based on data dimensionality and size. The fundamental distinction is between **1D vectors** (can be loaded entirely) and **2D matrices** (require tiled access for performance).

**Strategy 1: Async Copy for Vectors (1D Data)**

For 1D auxiliary tensors like bias vectors or per-row/per-column statistics, use simple asynchronous copy operations to load the entire vector upfront:

**High-level pattern**:
1. Issue async copy from global to shared memory using `cp.async` operations
2. Load the **entire vector** in a single batch (typically ≤1KB for row/column biases)
3. Commit the async group and wait for completion with barrier synchronization
4. Create broadcast views with stride-0 layouts to efficiently replicate values across threads
5. Access from shared memory during epilogue computation with minimal latency

**Key characteristics**: One-time upfront load, simple broadcast semantics, entire vector fits in shared memory. See the row/column bias implementation in `rapier/ops/epilogue_utils.py` for reference.

**Strategy 2: TMA Pipelines with Tiling for Matrices (2D Data)**

For 2D auxiliary matrices (e.g., residual connections [M×N], attention masks), **tiled memory access is critical for performance**. Use TMA (Tensor Memory Accelerator) with producer-consumer pipelines to load matrix tiles incrementally:

**High-level pattern**:
1. **Partition the matrix into tiles** that fit in shared memory (e.g., [128×128] blocks)
2. Set up multi-stage pipeline with dedicated producer and consumer states
3. **Producer warps**: Asynchronously load tiles using TMA, advancing through pipeline stages
4. **Consumer warps**: Wait for tile availability, process loaded tile, release stage back to producer
5. **Critical**: Overlaps loading of next tile with computation on current tile across multiple stages

**Why tiling matters**: Matrix data is too large to fit in shared memory at once. Tiled access with pipelining:
- Enables **high-bandwidth TMA transfers** (up to 1TB/s on H100) by loading contiguous 2D tiles
- **Hides memory latency** by overlapping computation with memory transfers across pipeline stages
- Avoids register/shared memory pressure from attempting to load entire matrices
- Maintains spatial locality for efficient cache utilization

Without tiling, performance degrades severely due to insufficient memory bandwidth utilization and exposed latency. See the residual connection implementation in `rapier/ops/epilogue_utils.py` for reference.

**Available Reference Implementations**

The `rapier/ops/epilogue_utils` module provides working examples of both patterns:
- Row/column bias operations: Vector loading with async copy and broadcast
- Residual connection operations: Matrix tiling with TMA producer-consumer pipeline

Study these implementations to understand the detailed mechanics rather than implementing from scratch.

**Choosing the Right Strategy**:
- **1D vectors (≤1KB)**: Use async copy—load entire vector upfront, broadcast from shared memory
- **2D matrices (>1KB)**: Use TMA pipelines with tiling—overlap tiled loading with computation

### Key Takeaways

1. **Shared memory required for cross-thread broadcasting** - Registers are thread-private
2. **Synchronize after shared writes** - Use barriers before reading shared values
3. **Choose memory staging based on size** - Direct loads for small, staged for large
4. **Epilogue composition enables modularity** - Sequential execution in `consumer_visit`
5. **Skip redundant reductions** - Don't reduce already-singular broadcast values
6. **Verify thread responsibility** - Use coordinate tensors to avoid write races
7. **Match staging strategy to data size** - Vector staging for small tensors, TMA pipelines for large matrices

---

## Learning from Reference Implementations

Reference implementations provide well-tested patterns for common operations:

1. **Unary Operations**: Complete infrastructure for single-input operations
   - Demonstrates proper tiling strategies
   - Shows memory copy patterns
   - Implements JIT compilation and caching

2. **Type Conversion**: Handling mixed-dtype operations
   - Different memory copy widths per dtype
   - Separate configurations for source and destination
   - Type conversion between different precisions
   - See [Type Conversion Guide](./CuTeDSL_type_conversion.md#mixed-dtype-operations) for detailed patterns

**Best Practice**: Study reference examples to understand underlying patterns and design principles. Adapt the patterns to your specific use case rather than copying code verbatim.

---

## Quick Reference Summary

### Critical Patterns to Remember

**Type System (Pitfalls #1-2, #11):**
- Python literals promote to fp32 → always cast back: `result.to(input_dtype)`
- `cute.Tensor` = memory reference, `cute.TensorSSA` = actual values → use `.load()` to dereference
- PyTorch i64 indices must be cast to `cute.Int32` for tensor indexing
- Low-precision inputs (fp16/bf16) → upcast to fp32 for computation, downcast back
- See [Type Conversion Guide](./CuTeDSL_type_conversion.md) for comprehensive patterns

**Memory Access (Pitfalls #10, #12-13):**
- Use 128-bit vectorized copies (`num_bits_per_copy=128`) for optimal bandwidth
- Initialize tensors before reading to avoid NaN propagation
- TMA buffers require ≥128-byte alignment (use 1024 bytes)
- **Always synchronize after async operations**: `cp_async_commit_group()` + `cp_async_wait_group(0)` + barriers

**TMA Pipelines (Pitfall #14):**
- Producer: `producer_acquire()` → TMA copy → `producer_commit()` → `state.advance()`
- Consumer: `consumer_wait()` → compute → fence + sync → `elect_one() { consumer_release() }` → `state.advance()`
- Missing any step causes deadlocks
- Reference: Epilogue implementations in `rapier/ops/epilogue_utils.py`

**Shared Memory (Pitfall #15):**
- Composing 3+ epilogues? Enable warp specialization to reduce thread count and avoid 227KB limit
- Reuse reduction buffers across passes: pass buffer from first call to subsequent calls
- Cross-thread broadcasting requires shared memory + barriers (registers are thread-private)

**Compilation (Pitfall #6):**
- Define operation functions at module level (not inside other functions)
- Cache kernels by (operation, row size, dtype) but exclude dynamic batch dimensions
- Use `cutlass.const_expr()` for compile-time branching (Pitfall #3)

**Mathematical Operations:**
- Enable `fastmath=True` for transcendentals: `cute.math.exp(x, fastmath=True)`
- No sqrt available → use `cute.math.rsqrt(x) * x`
- No division intrinsic → use `a * cute.arch.rcp_approx(b)`
- No min/max/floor/ceil → use boolean mask patterns (see [Mathematical Operations](#mathematical-operations))

### Implementation Pattern Quick Reference

| **Task** | **Pattern** | **Key File** |
|----------|-------------|--------------|
| Elementwise unary op | Define operation at module level, use helper infrastructure | `rapier/ops/elementwise_op.py` |
| Binary/multi-input op | Custom kernel with multiple memory copy calls | `rapier/ops/elementwise_op.py` |
| Row-wise reduction | Use reduction utilities, upcast to fp32 | `rapier/ops/reduction_utils.py` |
| Multi-pass reduction | Reuse buffer across calls | `rapier/ops/reduction_utils.py` |
| GEMM epilogue (simple) | Single epilogue operation class | `rapier/ops/epilogue_utils.py` |
| GEMM epilogue (complex) | Composition with warp specialization | `rapier/ops/epilogue_composite_utils.py` |
| 1D auxiliary tensor | Async copy to shared memory, broadcast with stride-0 layout | `rapier/ops/epilogue_utils.py` |
| 2D auxiliary matrix | TMA pipeline with producer-consumer protocol | `rapier/ops/epilogue_utils.py` |

### Debugging Quick Reference

**Symptom** → **Likely Cause** → **Solution**

- Getting fp32 output with fp16 input → Pitfall #1 (type promotion) → Add `.to(input_dtype)`
- `AttributeError: 'Tensor' has no attribute...` → Pitfall #2 (Tensor vs TensorSSA) → Call `.load()` first
- Branching error with runtime condition → Pitfall #3 (compile-time branching) → Wrap in `cutlass.const_expr()`
- Numerical errors in reductions → Pitfall #4 (mixed precision) → Upcast to fp32 before reduction
- Kernel produces NaN → Pitfall #10 (uninitialized memory) → Use initialized tensor creation
- i64 → i32 cast error → Pitfall #11 (index types) → Cast to `cute.Int32` before indexing
- Misaligned address error → Pitfall #12 (TMA alignment) → Use ≥128-byte alignment (1024 recommended)
- Wrong values from async copies → Pitfall #13 (missing sync) → Add barriers after async operations
- Kernel hangs/timeout → Pitfall #14 (pipeline protocol) → Verify all acquire/commit/wait/release calls
- Insufficient shared memory → Pitfall #15 (epilogue composition) → Enable warp specialization

**Debugging Tools:**
```bash
compute-sanitizer --tool=initcheck python script.py  # Detect uninitialized reads
compute-sanitizer --tool=racecheck python script.py  # Detect memory races
timeout 300 python script.py                         # Detect hangs during development
```

---

## Reference Implementation Index

**Core utilities:**
- `rapier/ops/elementwise_op.py` - Unary/binary operation infrastructure
- `rapier/ops/memory_utils.py` - Memory copy configuration and vectorized access
- `rapier/ops/reduction_utils.py` - Hierarchical reduction with warp intrinsics
- `rapier/ops/layout_utils.py` - Thread layout generation for reductions/epilogues

**Epilogue patterns:**
- `rapier/ops/epilogue_utils.py` - Epilogue implementations:
  - 1D vector loading with async copy and broadcast
  - 2D matrix loading with TMA producer-consumer pipeline
  - Multi-stage epilogues with auxiliary tensors
- `rapier/ops/epilogue_composite_utils.py` - Composing multiple operations
- `rapier/gemm/gemm_interface.py` - GEMM kernel integration with custom epilogues

**Related documentation:**
- [CuTeDSL Type Conversion Guide](./CuTeDSL_type_conversion.md) - Comprehensive dtype handling
- [Compute Sanitizer Guide](./compute_sanitizer.md) - Memory debugging tools

