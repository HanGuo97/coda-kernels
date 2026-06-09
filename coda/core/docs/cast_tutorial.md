# Your First CuTeDSL Kernel: 2D Tensor Cast

**A beginner-friendly introduction to GPU kernel programming**

Learn CuTeDSL fundamentals by building a simple but complete kernel that copies a 2D tensor while converting its data type. This tutorial covers the essential concepts needed for any GPU kernel, without the complexity of reductions or advanced optimizations.

## Table of Contents
1. [Why Start Here?](#why-start-here)
2. [The Task](#the-task)
3. [Prerequisites](#prerequisites)
4. [Step 1: Kernel Structure](#step-1-kernel-structure)
5. [Step 2: Memory Tiling](#step-2-memory-tiling)
6. [Step 3: Loading Data](#step-3-loading-data)
7. [Step 4: Type Conversion](#step-4-type-conversion)
8. [Step 5: Storing Results](#step-5-storing-results)
9. [Step 6: Bounds Checking](#step-6-bounds-checking)
10. [Complete Implementation](#complete-implementation)
11. [Testing](#testing)
12. [What's Next?](#whats-next)

---

## Why Start Here?

**This is the simplest non-trivial GPU kernel.** Before tackling reductions, attention, or matrix multiplication, master the fundamentals:

| Concept | Why It Matters | Covered Here |
|---------|----------------|--------------|
| **Memory tiling** | Divide work among thread blocks | ✓ Full explanation |
| **Thread layout** | Map threads to data elements | ✓ Simple 2D case |
| **Async copy** | Overlap memory and compute | ✓ GMEM→SMEM→REG |
| **Type conversion** | Register operations | ✓ Cast in registers |
| **Bounds checking** | Handle irregular shapes | ✓ Predicate usage |
| **No reductions** | Avoid complexity | ✓ Independent elements |
| **No shared memory coordination** | No synchronization needed | ✓ Simple data flow |

**Learning progression:** Cast (this tutorial) → RMSNorm (reduction) → LayerNorm (statistics) → Attention (complex patterns)

---

## The Task

**Goal:** Copy a 2D tensor `(M, N)` from one data type to another.

```
Input: X (M × N, dtype=fp32)         Output: Y (M × N, dtype=fp16)
┌───────────────────────┐            ┌───────────────────────┐
│  1.234   5.678  ...   │            │  1.234   5.678  ...   │
│  9.012   3.456  ...   │  ──cast──→ │  9.012   3.456  ...   │
│   ...     ...   ...   │            │   ...     ...   ...   │
└───────────────────────┘            └───────────────────────┘
     float32 in memory                    float16 in memory
```

**Why is this a GPU kernel task?**
- Millions of elements processed in parallel
- Memory bandwidth optimization critical
- Demonstrates GPU memory hierarchy

**Operations per element:**
1. Load from global memory (fp32)
2. Convert type (fp32 → fp16)
3. Store to global memory (fp16)

**That's it!** No complex math, no reductions, just memory + conversion.

---

## Prerequisites

### Required Knowledge
- Basic Python
- Understanding of array/tensor shapes
- Concept of parallel processing (threads doing work simultaneously)

### Required Imports
```python
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
```

### Vocabulary

| Term | Meaning | Example |
|------|---------|---------|
| **Dtype** | Data type | `torch.float32`, `torch.float16` |
| **GMEM** | Global memory (large, slow) | Where tensors live |
| **SMEM** | Shared memory (small, fast) | CTA-local cache |
| **REG** | Registers (tiny, fastest) | Thread-private storage |
| **CTA** | Cooperative Thread Array | Thread block |
| **Tile** | Chunk of data | Block processes a tile |

---

## Step 1: Kernel Structure

### The Big Picture

CuTeDSL kernels have three parts:

```python
class Cast2D:
    def __init__(self, ...):
        """Configure kernel (compile-time)"""

    @cute.jit
    def __call__(self, ...):
        """Launch kernel (runtime)"""

    @cute.kernel
    def kernel(self, ...):
        """GPU code (runs on each thread)"""
```

### Our Implementation

```python
class Cast2D:
    def __init__(
        self,
        in_dtype: cute.Numeric,
        out_dtype: cute.Numeric,
    ) -> None:
        """
        Initialize with data types known at compile time.

        Args:
            in_dtype: Input data type (e.g., cute.Float32)
            out_dtype: Output data type (e.g., cute.Float16)
        """
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # Input (M, N)
        mY: cute.Tensor,  # Output (M, N), pre-allocated
        stream: cuda.CUstream | None = None,
    ):
        """
        Launch the kernel.

        Args:
            mX: Input tensor, row-major (M, N), dtype=in_dtype
            mY: Output tensor, row-major (M, N), dtype=out_dtype
            stream: CUDA stream for async execution
        """
        M, N = mX.shape

        # Tiling configuration (explained in Step 2)
        tiler_mn = (4, 256)  # Each block: 4 rows × 256 columns
        num_threads = 128

        # Launch grid: one block per tile
        self.kernel(mX, mY, tiler_mn).launch(
            grid=[cute.ceil_div(M, tiler_mn[0]), cute.ceil_div(N, tiler_mn[1]), 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        tiler_mn: cute.Shape,
    ):
        """Device kernel: executed by each thread."""
        # Implementation in following steps
        pass
```

### Understanding Launch Configuration

```python
grid = [ceil(M/4), ceil(N/256), 1]  # How many blocks?
block = [128, 1, 1]                  # Threads per block
```

**Example:** For `M=1024, N=2048`:
- `grid = [256, 8, 1]` → 2,048 thread blocks
- Each block has 128 threads
- Total: 262,144 threads running in parallel

**Each thread processes:** `(4 × 256) / 128 = 8` elements

---

## Step 2: Memory Tiling

### The Problem

We have a large tensor `(M, N)` and thousands of threads. How do we divide the work?

**Naive approach (bad):**
```
Thread 0 → process entire row 0
Thread 1 → process entire row 1
...
```
**Problem:** Poor memory access patterns, no parallelism within rows.

**Tiling approach (good):**
```
Divide tensor into tiles (4 × 256)
Each thread block processes one tile
Threads within block cooperate on tile
```

### Tile Configuration

```python
tiler_mn = (4, 256)  # Tile shape: 4 rows × 256 columns
```

**Why 4 × 256?**
- `4 rows`: Small enough to fit in shared memory, enough to amortize overheads
- `256 columns`: Good for memory coalescing (multiple of 128 bytes)
- `4 × 256 = 1024 elements`: Divisible by thread count (128)

### Visual Breakdown

```
Full Tensor (1024 × 2048)
┌────────────────────────────────────────┐
│ ╔════╗ ╔════╗ ╔════╗ ... ╔════╗       │  Grid of tiles
│ ║ T0 ║ ║ T1 ║ ║ T2 ║     ║ T7 ║       │  Each tile = 4×256
│ ╚════╝ ╚════╝ ╚════╝     ╚════╝       │
│ ╔════╗ ╔════╗ ╔════╗ ... ╔════╗       │  256 tiles total
│ ║ T8 ║ ║ T9 ║ ║T10 ║     ║T15 ║       │  (256 row-tiles ×
│ ╚════╝ ╚════╝ ╚════╝     ╚════╝       │   8 col-tiles)
│  ...    ...    ...         ...        │
└────────────────────────────────────────┘

Single Tile (4 × 256) processed by 128 threads:
┌────────────────────────────────────┐
│ T0: [e0 e1 e2 e3 e4 e5 e6 e7 ...]  │  Each thread handles
│ T1: [e0 e1 e2 e3 e4 e5 e6 e7 ...]  │  8 consecutive elements
│ ... (128 threads total)            │  (1024 / 128 = 8)
└────────────────────────────────────┘
```

### Determining Thread's Work

```python
@cute.kernel
def kernel(self, mX, mY, tiler_mn):
    # 1. Get thread's position
    tidx = cute.arch.thread_idx()[0]  # 0-127
    bidx = cute.arch.block_idx()[0]   # Row tile index
    bidy = cute.arch.block_idx()[1]   # Column tile index

    # 2. Extract this block's tile
    gX = cute.local_tile(mX, tiler_mn, (bidx, bidy))
    gY = cute.local_tile(mY, tiler_mn, (bidx, bidy))
    # gX shape: (4, 256)
```

---

## Step 3: Loading Data

### Memory Hierarchy Strategy

GPU memory has three levels with vastly different speeds:

```
Global Memory (GMEM)  →  Shared Memory (SMEM)  →  Registers (REG)
   ~400 GB/s              ~10 TB/s                ~50 TB/s
   All threads            CTA-local               Thread-private
   Slow access            Medium access           Fastest access
```

**Strategy:**
1. **GMEM → SMEM**: Async copy (overlaps with other work)
2. **SMEM → REG**: Fast synchronous copy
3. **Compute in REG**: Type conversion
4. **REG → GMEM**: Write results

### Allocating Shared Memory

```python
@cute.kernel
def kernel(self, mX, mY, tiler_mn):
    # Allocate shared memory for this block's tile
    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(
        self.in_dtype,
        cute.make_ordered_layout(tiler_mn, order=(1, 0)),  # Column-major
        byte_alignment=16,
    )
    # sX shape: (4, 256), stored in shared memory
```

**Why column-major layout?** Ensures coalesced memory access (consecutive threads access consecutive addresses).

### Setting Up Copy Operations

```python
# Create copy atom (fundamental copy instruction)
copy_op = cute.nvgpu.cpasync.CopyG2SOp()  # Async copy from global to shared
cdtype = cute.Float32  # Or use TORCH_DTYPE_TO_CUTLASS_DTYPE_MAP[self.in_dtype]
copy_atom_X = cute.make_copy_atom(
    op=copy_op,
    copy_internal_type=cdtype,
    num_bits_per_copy=8 * cute.sizeof_bits(cdtype),  # 8 elements per instruction
)

# Create tiled copy (distributes work across threads)
tv_layout = (128, 8)  # 128 threads × 8 elements = 1024
tiled_copy_X = cute.make_tiled_copy(copy_atom_X, tv_layout, tiler_mn)

# Get this thread's partition
tidx = cute.arch.thread_idx()[0]
thread_copy_X = tiled_copy_X.get_slice(tidx)
```

### Partitioning Tensors

```python
# Partition global tensor for this thread
tXgX = thread_copy_X.partition_S(gX)  # Source (global memory)
tXsX = thread_copy_X.partition_D(sX)  # Destination (shared memory)

# tXgX shape: (8,) - this thread handles 8 elements
```

### Async Copy GMEM → SMEM

```python
# Initiate async copy
cute.copy(tXgX, tXsX, is_async=True)
cute.arch.cp_async_commit_group()

# ... could do other work here ...

# Wait for copy to complete
cute.arch.cp_async_wait_group(0)
cute.arch.syncthreads()  # All threads sync
```

### Copy SMEM → Registers

```python
# Allocate register storage
tXrX = cute.make_fragment_like(tXgX)

# Fast synchronous copy
cute.autovec_copy(tXsX, tXrX)
```

---

## Step 4: Type Conversion

Now we have data in registers. Time to convert types!

```python
# Load values from register tensor
x_values = tXrX.load()  # Load as current dtype

# Convert to output dtype
if self.out_dtype != self.in_dtype:
    y_values = x_values.to(self.out_dtype)
else:
    y_values = x_values

# Create output register tensor
tYrY = cute.make_fragment(self.out_dtype, tXrX.shape, tXrX.layout)
tYrY.store(y_values)
```

**What happens here?**
- `load()`: Fetch data from registers into intermediate representation
- `.to(dtype)`: Hardware type conversion instruction (e.g., `CVT.F16.F32`)
- `store()`: Put converted values back in register tensor

**Performance:** Type conversion is extremely fast in registers (1-2 cycles per element).

---

## Step 5: Storing Results

### Preparing Output Partition

```python
# Extract output tile
gY = cute.local_tile(mY, tiler_mn, (bidx, bidy))

# Partition for this thread
tYgY = thread_copy_X.partition_S(gY)  # Reuse same layout
```

### Direct Copy to Global Memory

```python
# Copy from registers to global memory
cute.copy(tYrY, tYgY)
```

**Note:** No async copy for store (not typically beneficial). Direct register→GMEM write.

---

## Step 6: Bounds Checking

### The Problem

What if tensor dimensions aren't perfect multiples of tile size?

**Example:** `M=127, N=300` with tiles `(4, 256)`
- Row tiles: `ceil(127/4) = 32` blocks, but last block only has 3 valid rows
- Column tiles: `ceil(300/256) = 2` blocks, but last block only has 44 valid columns

**Without bounds checking:** Out-of-bounds memory access → crash or corruption!

### Solution: Identity Tensors and Predicates

#### Step 6.1: Create Identity Tensor

```python
# Identity tensor: each element stores its own coordinates
idX = cute.make_identity_tensor(mX.shape)  # (M, N)
# idX[i, j] = (i, j)

# Partition same as data
cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))
tXcX = thread_copy_X.partition_S(cX)
```

#### Step 6.2: Extract Coordinates

```python
# Get this thread's coordinates
coords = tXcX[(0, None), None, None]
row = coords[0][0]  # This thread's row
col = coords[0][1]  # This thread's first column
```

#### Step 6.3: Generate Predicates

```python
# Check if dimensions are tile-aligned
M, N = mX.shape
is_even_M = cutlass.const_expr(M % tiler_mn[0] == 0)
is_even_N = cutlass.const_expr(N % tiler_mn[1] == 0)

# Create column predicates (mask out-of-bounds columns)
if not is_even_N:
    # Generate predicates based on coordinates
    coords = tXcX[None, (0, None), None]
    tXpX = cute.make_tensor(cute.Bool, tXgX.shape, tXgX.layout)
    for i in range(cute.size(tXpX)):
        col_idx = coords[i][1]
        tXpX[i] = col_idx < N
else:
    tXpX = None  # All columns valid
```

#### Step 6.4: Guard Memory Operations

```python
# Check row bounds
if is_even_M or row < M:
    # Load with predicate (skips out-of-bounds elements)
    cute.copy(tXgX, tXsX, is_async=True, pred=tXpX)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.syncthreads()

    # Copy to registers
    cute.autovec_copy(tXsX, tXrX)

    # Type conversion
    x_values = tXrX.load()
    y_values = x_values.to(self.out_dtype)
    tYrY.store(y_values)

    # Store with predicate
    cute.copy(tYrY, tYgY, pred=tXpX)
```

**How predicates work:**
- `pred=tXpX`: Only copy elements where `tXpX[i] == True`
- Hardware instruction includes mask bit
- Out-of-bounds accesses are skipped entirely

---

## Complete Implementation

```python
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


class Cast2D:
    def __init__(
        self,
        in_dtype: cute.Numeric,
        out_dtype: cute.Numeric,
    ) -> None:
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        stream: cuda.CUstream | None = None,
    ):
        M, N = mX.shape

        # Tiling configuration
        tiler_mn = (4, 256)
        num_threads = 128

        # Launch kernel
        self.kernel(mX, mY, tiler_mn).launch(
            grid=[cute.ceil_div(M, tiler_mn[0]), cute.ceil_div(N, tiler_mn[1]), 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        tiler_mn: cute.Shape,
    ):
        # Thread indices
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        bidy = cute.arch.block_idx()[1]

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            self.in_dtype,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        # Setup copy operations
        copy_op = cute.nvgpu.cpasync.CopyG2SOp()
        copy_atom_X = cute.make_copy_atom(
            op=copy_op,
            copy_internal_type=self.in_dtype,
            num_bits_per_copy=8 * cute.sizeof_bits(self.in_dtype),
        )
        tv_layout = (128, 8)
        tiled_copy_X = cute.make_tiled_copy(copy_atom_X, tv_layout, tiler_mn)
        thread_copy_X = tiled_copy_X.get_slice(tidx)

        # Extract tiles
        gX = cute.local_tile(mX, tiler_mn, (bidx, bidy))
        gY = cute.local_tile(mY, tiler_mn, (bidx, bidy))

        # Partition tensors
        tXgX = thread_copy_X.partition_S(gX)
        tXsX = thread_copy_X.partition_D(sX)
        tYgY = thread_copy_X.partition_S(gY)

        # Allocate registers
        tXrX = cute.make_fragment_like(tXgX)
        tYrY = cute.make_fragment(self.out_dtype, tXrX.shape, tXrX.layout)

        # Bounds checking
        M, N = mX.shape
        is_even_M = cutlass.const_expr(M % tiler_mn[0] == 0)
        is_even_N = cutlass.const_expr(N % tiler_mn[1] == 0)

        # Identity tensor for coordinates
        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))
        tXcX = thread_copy_X.partition_S(cX)
        row = tXcX[(0, None), None, None][0][0]

        # Predicates
        if not is_even_N:
            coords = tXcX[None, (0, None), None]
            tXpX = cute.make_tensor(cute.Bool, tXgX.shape, tXgX.layout)
            for i in range(cute.size(tXpX)):
                col_idx = coords[i][1]
                tXpX[i] = col_idx < N
        else:
            tXpX = None

        # Main computation
        if is_even_M or row < M:
            # Load from global to shared
            cute.copy(tXgX, tXsX, is_async=True, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.syncthreads()

            # Copy to registers
            cute.autovec_copy(tXsX, tXrX)

            # Type conversion
            x_values = tXrX.load()
            y_values = x_values.to(self.out_dtype)
            tYrY.store(y_values)

            # Store results
            cute.copy(tYrY, tYgY, pred=tXpX)


# Usage example
def cast_2d(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Wrapper for easy use with PyTorch."""
    # Map PyTorch dtypes to CuTe dtypes
    dtype_map = {
        torch.float32: cute.Float32,
        torch.float16: cute.Float16,
        torch.bfloat16: cute.BFloat16,
    }

    in_dtype = dtype_map[x.dtype]
    out_dtype_cute = dtype_map[out_dtype]

    # Create output tensor
    out = torch.empty_like(x, dtype=out_dtype)

    # Create and run kernel
    kernel = Cast2D(in_dtype, out_dtype_cute)
    kernel(x, out)

    return out
```

---

## Testing

The test suite in [kernels/tests/cast.py](../../kernels/tests/cast.py) verifies correctness:

```python
# Run all tests
pytest kernels/tests/cast.py -v

# Run specific size
pytest kernels/tests/cast.py::test_cast_2d[1024-2048-float32-float16] -v

# Run irregular shapes (tests bounds checking)
pytest kernels/tests/cast.py::test_cast_2d_irregular_shapes -v
```

**Test coverage:**
- Various tensor sizes (1×64 to 8192×4096)
- All dtype combinations (fp32↔fp16↔bf16)
- Irregular shapes (non-tile-aligned dimensions)
- Numerical accuracy validation

---

## What's Next?

You now understand the core building blocks of GPU kernels:

✓ Kernel structure (host/device separation)
✓ Memory tiling and thread partitioning
✓ Async copy operations
✓ Type conversion in registers
✓ Bounds checking with predicates

**Ready for more?** Try these progressively harder tutorials:

1. **[RMSNorm](reduction.md)** - Add reduction operations (sum, mean)
2. **[LayerNorm](layernorm.md)** - Multi-pass algorithms, welford aggregation
3. **[Softmax](softmax.md)** - Numerically stable reductions
4. **[Attention](attention.md)** - Matrix multiplication, advanced tiling

**Key concepts to explore next:**
- Warp shuffle reductions
- Shared memory synchronization
- Multi-stage pipelines
- Block-wide coordination

Happy kernel programming! 🚀
