# Task: Implement High-Performance CUDA Kernels with CuTeDSL

Implement high-performance CUDA kernels using **CuTeDSL** (a Python DSL for CUTLASS) targeting **H100 (Hopper)** architecture.

## Goal

Implement `{{FUNCTION_NAME}}` in `gpt.py` as a **single, fully fused CuTeDSL kernel** using Epilogue Visitor Tree (EVT) patterns—no multi-kernel decomposition or PyTorch fallbacks.

## Workflow

1. Study the reference implementation in `kernels/refs/` to understand the mathematical operations
2. Review `rapier.ops.*` modules and working examples in `rapier/examples/`
3. **IMPORTANT: If implementing custom epilogue operations**, read `rapier/docs/epilogue_lessons_learned.md` carefully first. It covers critical patterns for TMA pipelines, memory staging, broadcasting, and synchronization that are essential for correct epilogue implementations
4. Design your Epilogue Visitor Tree (EVT) by composing reusable visitors from `rapier.epilogue`—such as bias addition, reductions, activations, cross-entropy, and residual connections. These proven abstractions ensure correctness and optimal performance. For kernels requiring multiple epilogue operations, use `rapier.epilogue.composite` to chain individual visitors into a unified epilogue (see `rapier/examples/gemm.py` for reference). Before writing a new visitor, ask: (1) Can this be implemented by composing existing visitors from `rapier.epilogue`? (2) Can existing visitors in the current file be reused or adapted? Only write custom visitor classes when the required functionality is genuinely unavailable through composition
5. Start with default GEMM configurations during development (see `rapier/gemm/gemm_interface.py`). All GEMM wrapper functions must include a boolean parameter to toggle between default and autotuned execution. Default to non-tuned mode for faster iteration
6. Test iteratively: start with a subset of representative test cases, then expand to the full suite. Watch for deadlocks—if tests hang, use `timeout 60 python script.py` to detect early. Common causes include missing barriers after async operations, incomplete TMA pipeline protocol (`acquire/commit/wait/release`), and barrier mismatches. See `CuTeDSL_lessons_learned.md` for debugging patterns
7. Optimize using compile-time conditionals (`cutlass.const_expr`) to specialize for different problem shapes, tensor dimensions, and H100 hardware characteristics

**You have everything you need**—all necessary tools, abstractions, and working examples are provided in the Rapier library. When debugging, study the patterns carefully, experiment with different approaches, and persist through challenges. If you get stuck, `rapier/docs/CuTeDSL_lessons_learned.md` may have useful insights.

## Key Resources

**CuTeDSL Documentation:**
- Core documentation: `/export/share/cutlass-docs/cute_dsl_general/`
- Limitations: `/export/share/cutlass-docs/limitations.md`
- Example kernels: `/workspace/main/hilt/cutlass/examples/python/CuTeDSL/`
- Source code: `/workspace/main/hilt/cutlass/python/CuTeDSL/`
- Full docs: `/export/share/cutlass-docs/`

**IMPORTANT:** Review the Rapier library section below before starting — it contains essential utilities and working examples for kernel development.

## Implementation Requirements

**File Structure:**
- Reference implementations (PyTorch): `kernels/refs/`
- Output location: `kernels/gens/` (follow existing interfaces)
- Success criteria: All tests in `kernels/tests/` must pass

## Rapier Library

Rapier provides high-level utilities for CuTeDSL kernel development.

### Available Modules

#### Utilities

Import and use functions from `rapier.ops` modules directly.

- **`rapier.ops.creation_utils`**: Allocate tensors in register (`rmem`) or shared memory (`smem`) from shape, layout, or existing tensor. Provides `empty_like`, `zeros_like`, `ones_like`, `full_like` for both `Tensor` and `TensorSSA`
- **`rapier.ops.layout_utils`**: Create ordered layouts (row/col-major) and compute thread-value decompositions
- **`rapier.ops.memory_utils`**: Memory copy operations (tiled, vectorized) with automatic predication for bounds checking. TMA descriptor creation for efficient global-to-shared and shared-to-global transfers
- **`rapier.ops.math_utils`**: Mathematical operations (min/max, clamping) with automatic scalar/tensor dispatch
- **`rapier.ops.dtype_utils`**: Type conversion and rounding operations with PTX-based implementations
- **`rapier.ops.reduction_utils`**: Hierarchical reductions (thread→warp→block) with built-in `add`, `mul`, `max`, `min` ops. Supports custom reduction registration and auto-manages shared memory buffers
- **`rapier.ops.gemm_utils`**: Comprehensive GEMM configuration and validation utilities — tile shape validation, warp group and register allocation, data type compatibility checks, shared memory layout construction, accumulator transformations, grid/stage computation, and pipeline creation for mainloop and scheduler phases
- **`rapier.ops.pipeline_utils`**: Pipeline state advancement utilities for multi-iteration control flow in persistent kernels
- **`rapier.ops.epilogue_utils`**: Low-level SM90 epilogue infrastructure — register↔shared copy descriptor setup, TMA descriptor creation for async global↔shared transfers, TMA load pipeline initialization with producer/consumer state, and shared memory sizing helpers for both vectors (fixed/stage-independent) and matrices (per-stage)
- **`rapier.ops.profiling_utils`**: Profile kernels using NVIDIA Nsight Compute (NCU) — see below
- **`rapier.ops.benchmark_utils`**: Query GPU hardware info (memory bandwidth, clock rates)

#### Epilogue Implementations

Import and use epilogue visitors from `rapier.epilogue` modules directly. All implementations follow the Epilogue Visitor Tree (EVT) pattern with a unified lifecycle: host arguments → device parameters → TMA prefetch → shared memory allocation → pipeline setup → producer/consumer execution with interleaved TMA loads → pipeline advancement. Each visitor manages its own shared memory across three budget categories: fixed (stage-independent), consumer-staged, and producer-staged. Compose multiple visitors using the composite module.

- **`rapier.epilogue.base`**: Abstract visitor interface defining the full epilogue lifecycle — argument conversion, TMA prefetch, pipeline management, producer phases (prefetch/TMA load), consumer phases (begin/end, per-epi-tile visit/smem_store/tma_store), and shared memory layout. Each visitor declares typed data containers for its arguments, parameters, tensors, and pipelines. Includes a no-op default. Extend this for custom epilogue operations
- **`rapier.epilogue.bias`**: Row and/or column bias addition — loads bias vectors via `cp.async` to shared memory (fixed allocation), broadcasts via stride-0 layouts, and adds element-wise to the accumulator. Both biases are optional
- **`rapier.epilogue.reduction`**: Block-level row/column reductions with three variants. **Store**: per-tile partial reductions with thread→warp hierarchy, one value per tile. **Store-2X**: two independent values per tile via i64 packing. **Load**: loads partials via `cp.async`, aggregates across tiles, and fuses into the accumulator. Supports standard ops (sum, max, min, product). Combine store and load across two passes for full reductions
- **`rapier.epilogue.cross_entropy`**: Two visitors for fused cross-entropy. **Partial cross-entropy**: online softmax computing per-tile max and SSE statistics with warp-level reduction — requires a second pass to aggregate across tiles. **Logit selection**: loads target indices via `cp.async`, then extracts the target logit per row by matching column index
- **`rapier.epilogue.activation`**: Dual-output activation (pre and post-activation) — supports elementwise (1→1), pairwise (2→2), contraction (2→1), and expansion (1→2) modes. Stores post-activation via register→shared then TMA store, overlapping with consumer computation
- **`rapier.epilogue.matrix`**: Matrix loading and fusion via multi-stage TMA pipeline — producer warps load tiles to shared memory while consumer warps fuse with the accumulator, overlapping load and compute. Includes a **residual-add** specialization and a **2X variant** loading double-width tiles for paired operations (e.g., gating) with a custom `fn(acc, elem0, elem1)`
- **`rapier.epilogue.composite`**: Chains multiple visitors into a unified epilogue — delegates all lifecycle methods left-to-right and aggregates shared memory across all budget categories

#### GEMM Implementations

Import and use GEMM kernels from `rapier.gemm` modules directly. All implementations support batched operations, TMA memory transfers, and epilogue fusion via Epilogue Visitor Tree pattern.

- **`rapier.gemm.gemm_simple`**: Baseline WGMMA kernel with TMA loads/stores, cluster multicast, multi-stage pipelining, and warp-0 scheduling. Supports fp16/bf16 and fp8 (e4m3/e5m2) inputs with fp32 accumulation. Best for learning core GEMM structure
- **`rapier.gemm.gemm_quack`**: Production WGMMA kernel with warp specialization (dedicated load/compute warps), persistent tile scheduling, and optional pingpong mode (dual warp groups alternating on output tiles). Provides higher occupancy and better memory-compute overlap. Recommended for performance-critical workloads
- **`rapier.gemm.gemm_interface`**: PyTorch wrapper with transparent compilation caching, automatic layout transformations, epilogue configuration, and optional autotuning. Handles device/stream management and validates alignment. Use for PyTorch integration

### Available Examples

Study patterns in `rapier/examples/` for guidance, then adapt and reimplement for your specific use case.

- **`rapier/examples/elementwise_apply.py`**: 2D elementwise kernel pattern — shows tiling, global→register copy, in-register computation, register→global writeback, and PyTorch integration with compilation caching
- **`rapier/examples/cast.py`**: 2D type conversion kernel pattern — shows mixed-dtype tensor operations, vectorized memory access, register-level type casting via `Tensor.load().to()`, optimal vector size selection, and dtype-aware compilation caching. **Critical:** When selecting vector size for mixed-dtype kernels, you MUST consider EVERY input AND output tensor's dtype — partial consideration (e.g., only inputs or only some tensors) will cause memory coalescing failures or misaligned accesses
- **`rapier/examples/round.py`**: 2D rounding and type conversion — demonstrates `convert(source, dtype, style="rn")` for fused rounding+conversion vs `round(source, style="rn").to(dtype)` for separate operations, showing trade-offs between instruction fusion and explicit control
- **`rapier/examples/reduction.py`**: Row-wise reduction with RMS normalization — shows hierarchical reduction usage, adaptive memory reload (registers/smem/gmem), and stride-0 broadcasting for per-row operations
- **`rapier/examples/gemm.py`**: PyTorch GEMM wrapper demonstrating epilogue composition — row/column bias, block-wise reductions (load/store modes), partial cross-entropy, logit selection, custom activations (elementwise/pairwise/expansion/contraction), and residual connections. Shows composing multiple visitors from `rapier.epilogue` into a unified epilogue, batched tensor layout transformations, and optional autotuning. Reference for building application-specific GEMM APIs
- **`rapier/examples/nan_debug.py`**: NaN debugging in reductions — demonstrates `isnan()`/`check_nan()` from `debug_utils`, device assertions (`--enable-device-assertions` flag to abort at NaN origin), and `cutlass.const_expr()` to toggle buggy vs fixed versions. Example: handling (-inf) - (-inf) = NaN in reduction combine function

**Additional References:**
- **`rapier/docs/epilogue_lessons_learned.md`**: **Essential guide for implementing custom epilogue operations.** Covers TMA pipeline protocol (four-phase lifecycle with producer/consumer patterns), shared memory alignment requirements, async synchronization, broadcasting patterns, and memory staging strategies. Read this carefully before writing custom EVT classes.
- **`rapier/docs/CuTeDSL_lessons_learned.md`**: Collection of practical patterns and common pitfalls covering type system behavior, memory operations, reductions, epilogue patterns, performance tips, and debugging techniques. Consult when encountering issues.
- **`rapier/docs/type_conversion.md`**: Comprehensive guide to type conversion behavior, precision management, and dtype handling. Essential reading for mixed-precision operations and avoiding silent type promotion.
