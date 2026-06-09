# Epilogue Visitor Tree: Lessons Learned

## TMA Shared Memory Sizing

Use `cute.size()` to compute element counts for TMA shared memory allocations:

**Correct:**
```python
smem_bytes = cute.size(epi_tile) * dtype.width // 8
```

**Incorrect:**
```python
smem_bytes = tile_M * tile_N * dtype.width // 8
```

## Minimum Shared Memory Requirements (Non-TMA)

For non-TMA shared memory allocations, ensure minimum size to avoid OOB access (see [CUTLASS #2980](https://github.com/NVIDIA/cutlass/issues/2980)).

**Key requirement**: Maintain consistency between `get_smem_struct` and `get_smem_bytes_per_stage`. Even if computed size is small, allocate sufficient space to prevent hardware from accessing beyond allocated boundaries.

This primarily applies to standard async copies and register-to-shared operations, not TMA which has its own sizing via `cute.size()`.

## TMA Shared Memory Alignment

TMA operations require shared memory buffers aligned to at least 128 bytes. Use 1024 bytes as default.

**Correct:**
```python
@cute.struct
class SharedStorage:
    smem_buffer: cute.struct.Align[
        cute.struct.MemRange[dtype, size],
        1024,  # 128 minimum, 1024 recommended
    ]
```

**Incorrect:**
```python
@cute.struct
class SharedStorage:
    smem_buffer: cute.struct.Align[
        cute.struct.MemRange[dtype, size],
        16,  # Too small - causes silent corruption
    ]
```

## Async Memory Synchronization

Always synchronize after async operations before accessing destination memory:

```python
# Initiate async copy
async_copy_operation(src=source, dst=destination)

# Synchronize before reading destination
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)
barrier.arrive_and_wait()

# Now safe to access destination
data = load_from_destination(destination)
```

Missing synchronization causes incorrect results (zeros, garbage, intermittent failures).

## TMA Pipeline Protocol

TMA pipelines require strict producer-consumer protocol through four phases:

### Phase 1: Prefetch TMA Descriptors

Prefetch descriptor to L1 cache (done in `prefetch_tma_descriptors`):

```python
cute.nvgpu.cpasync.prefetch_descriptor(epi_tma_atom)
```

### Phase 2: Create Pipeline and States

Use utility function (done in `prepare_pipelines`):

```python
from rapier.ops.epilogue_utils import prepare_epi_load_pipeline

epi_load_pipeline, consumer_state, producer_state = prepare_epi_load_pipeline(
    epi_load_stage=epi_load_stage,
    epi_dtype=epi_dtype,
    epi_num_warps=epi_num_warps,
    epi_smem_layout=epi_smem_layout,
    epi_load_pipeline_mbar_ptr=barrier_ptr,
)
```

### Phase 3: Fill Pipeline with Initial Tiles

Prefetch initial tiles to fill pipeline (done in `producer_begin`):

```python
epi_prefetch = cutlass.min(epi_tile_num, epi_load_stage)
for epi_idx in cutlass.range(epi_prefetch, unroll=1):
    epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
    if is_tma_warp:
        epi_load_pipeline.producer_acquire(producer_state)
        tma_bar_ptr = epi_load_pipeline.producer_get_barrier(producer_state)
        src = tDgData[None, epi_coord]
        dst = tDsData[None, producer_state.index]
        cute.copy(atom=atom, src=src, dst=dst, tma_bar_ptr=tma_bar_ptr)
        epi_load_pipeline.producer_commit(producer_state)
    producer_state.advance()
```

### Phase 4: Steady-State Consumer and Producer

**Consumer pattern** (in `consumer_begin_loop`) - wait, copy, fence, release:

```python
# Wait for TMA transfer completion
epi_load_pipeline.consumer_wait(consumer_state)

# Load from shared memory to registers
src = tSR_sData[None, None, None, consumer_state.index]
cute.copy(atom=tiled_copy, src=src, dst=tSR_rData)

# Fence to make sure shared memory read is visible to TMA load
cute.arch.fence_proxy(
    kind=cute.arch.ProxyKind.async_shared,
    space=cute.arch.SharedSpace.shared_cta,
)
cute.arch.sync_warp()

# Release stage (only one thread)
with cute.arch.elect_one():
    epi_load_pipeline.consumer_release(consumer_state)
consumer_state.advance()
```

**Producer pattern** (in `producer_tma_load`) - acquire, copy, commit:

```python
# Only load if more tiles remain (producer stays ahead by epi_load_stage)
if cutlass.const_expr(epi_idx + epi_load_stage < epi_tile_num):
    epi_coord = epi_tile_layout.get_hier_coord(epi_idx + epi_load_stage)
    if is_tma_warp:
        tma_bar_ptr = epi_load_pipeline.producer_get_barrier(producer_state)

        # Wait for available pipeline slot
        epi_load_pipeline.producer_acquire(producer_state)

        # Issue TMA copy
        src = tDgData[None, epi_coord]
        dst = tDsData[None, producer_state.index]
        cute.copy(atom=atom, src=src, dst=dst, tma_bar_ptr=tma_bar_ptr)

        epi_load_pipeline.producer_commit(producer_state)
    producer_state.advance()
```

**Key**: Producer loads `epi_load_stage` tiles ahead, overlapping transfers with computation.

### Critical Requirements

1. **Four-phase sequence** - Prefetch descriptor → create pipeline → fill pipeline → steady-state loop
2. **Separate methods** - `prefetch_tma_descriptors` → `prepare_pipelines` → `producer_begin` → loop with `consumer_begin_loop` + `producer_tma_load`
3. **Get barrier before acquire** - Call `producer_get_barrier(state)` before `producer_acquire(state)`
4. **Always acquire in producer** - Even during initial prefetch in `producer_begin`
5. **Fence before release** - Consumer must fence + sync before releasing stage (makes shared memory read visible to TMA)
6. **Elect-one for release** - Only one thread calls `consumer_release()`
7. **State advancement** - Both consumer and producer states advance every iteration (even when no TMA operation)

Missing any step causes deadlocks or data corruption.

## Warp Specialization for Complex Epilogues

Enable `pingpong=True` to reduce epilogue thread count and avoid shared memory overflow when composing multiple operations:

```python
result = gemm_epilogue(..., pingpong=True)
```

Reduces per-operation overhead by 50% (from ~256 to ~128 threads) and improves performance 5-15%. Critical when composing 3+ operations or loading auxiliary tensors.

## Memory Staging Strategy

Choose the loading strategy based on data size and dimensionality:

### 1D Vectors (≤1KB): Use cp.async

For small broadcasted vectors (biases, target indices, scales):

```python
g2s_copy(src=gVector, dst=sVector)
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)
epi_barrier.arrive_and_wait()
```

Loads complete in <10 cycles; pipelining overhead not justified.

### 2D Matrices (>1KB): Use TMA - REQUIRED for Performance

**⚠️ CRITICAL: TMA pipelining is required for 2D matrix data to avoid 10-40x performance degradation.**

Using cp.async for large matrices causes severe slowdowns due to:
- No latency hiding: GPU stalls waiting for each tile load
- Memory latency dominates computation time → low GPU utilization

**TMA overlaps memory transfers with computation** through producer-consumer pipelining, achieving high utilization:

```python
# Producer warps load tile N+k while consumer warps process tile N
# Implementation: EVTResidual (rapier/ops/epilogue_utils.py:2457)
```

**Decision criteria:**
- **Use TMA:** 2D matrices >1KB (residual connections, per-tile auxiliary data)
- **Use cp.async:** 1D vectors ≤1KB with stride-0 broadcast (biases, scales)

## Broadcasting Per-Row Values

Register memory is thread-local. Use shared memory for cross-thread broadcasting.

```python
# 1. Allocate shared memory
@cute.struct
class SharedStorage(EpilogueSharedStorage):
    sValues: cute.struct.Align[cute.struct.MemRange[dtype, tile_M], 16]

# 2. Compute and write to shared memory (distributed)
for m in cutlass.range(tile_M):
    if m % epi_num_threads == tidx:
        sValues[m] = compute_value(m)

# 3. Synchronize
epi_barrier.arrive_and_wait()

# 4. Create broadcast view with stride-0 layout
broadcast_layout = cute.make_layout(shape=(tile_M, tile_N), stride=(1, 0))
broadcast_view = cute.make_tensor(iterator=sValues.iterator, layout=broadcast_layout)
```

## Loading Bias Vectors and Additional Tensors (Non-TMA)

When epilogue operations need small additional data beyond the GEMM output (row/column biases, scale factors, target vectors), use **standard async copies** (not TMA):

```python
# Example: Loading a column bias vector [M] to add to GEMM output [M×N]
mColVec = epi_params.mColVec[batch_idx, None]
gColVec = cute.local_tile(mColVec, (tile_M,), (m_idx,))
sColVec = epi_tensors_smem.sColVec

# Step 1: Copy from global to shared memory
g2s_copy(
    src=gColVec,
    dst=sColVec,
    crd=cute.make_identity_tensor(tile_M),
    shape=(tile_M,),
    num_threads=epi_num_threads,
    thread_index=tidx,
)

# Step 2: Synchronize before accessing shared memory
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)
epi_barrier.arrive_and_wait()

# Step 3: Create broadcast view (see Broadcasting section)
sColVec_view_layout = cute.make_layout(shape=(tile_M, tile_N), stride=(1, 0))
sColVec_view = cute.make_tensor(iterator=sColVec.iterator, layout=sColVec_view_layout)
tDsColVec = partition_for_epilogue(sColVec_view)
```

**Why staged loading:** Provides coalesced memory access and efficient distribution across threads.

**When to use:** For 1D vectors (biases, scales) and small auxiliary tensors. For large 2D matrices, use TMA (see Memory Staging Strategy section).

## Epilogue Composition

Use `EVTList` for sequential execution of multiple operations:

```python
from rapier.ops.epilogue_composite_utils import EVTList

def epilogue_visitor_tree_cls(acc_dtype, tile_shape_mnk, buffer_align_bytes):
    evt_a = EVTOperationA(...)
    evt_b = EVTOperationB(...)
    return EVTList(evts=[evt_a, evt_b])

epi_args = EVTList.EpilogueArguments(
    _list=[EVTOperationA.EpilogueArguments(...),
           EVTOperationB.EpilogueArguments(...)]
)
```

Total shared memory is sum of all operations.
