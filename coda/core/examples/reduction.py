import torch
import warnings

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda
from typing import Callable

from coda.core.ops import misc_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops import reduction_utils
from coda.core.ops.launch_utils import launch_check

ALLOWED_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Cluster is not yet supported
ENABLE_CLUSTER = False

# Whether to use faster but less precise math ops
FAST_MATH = False


@cute.kernel
def reduction_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor | None,
    mB: cute.Tensor | None,
    mY: cute.Tensor,
    mZ: cute.Tensor | None,
    eps: cute.Float32,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    online: cutlass.Constexpr,
    reload: cutlass.Constexpr,
    vector_size: cutlass.Constexpr,
    cluster_size: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(cluster_size > 1):
        # cluster index for the y-axis
        cidy = cute.arch.block_idx()[1]
    else:
        cidy = cutlass.const_expr(0)
    allocator = cutlass.utils.SmemAllocator()

    idX = cute.make_identity_tensor(mX.shape)
    gX = cute.local_tile(mX, tiler_mn, (bidx, cidy))
    gY = cute.local_tile(mY, tiler_mn, (bidx, cidy))
    # W and B are row vectors with shape [N], hence we do not need to index their row
    gW = cute.local_tile(mW, tiler_mn, (0, cidy)) if cutlass.const_expr(mW is not None) else None
    gB = cute.local_tile(mB, tiler_mn, (0, cidy)) if cutlass.const_expr(mB is not None) else None
    cX = cute.local_tile(idX, tiler_mn, (bidx, cidy))

    def _tiled_copy(
        src: cute.Tensor,
        dst: cute.Tensor | str,
        is_async: bool,
    ) -> tuple[cute.Tensor, memory_utils.MemoryCopyStruct]:
        """Performs a predicated, tiled copy with thread partitioning and optional async support.

        Creates a tiled copy operation that distributes work across threads using a CopyAtom
        and TiledCopy. Each thread receives a partition of the source and destination tensors
        according to the configured tiler_mn and layout_tv. Predicates are automatically
        generated using cX coordinates and mX.shape to guard against out-of-bounds access.

        Args:
            src: Source tensor (can be block or thread tensor depending on memspace).
            dst: Destination tensor or memspace string ("smem", "rmem"). If string, memory
                 will be allocated with the same shape as the source.
            is_async: If True, uses cp.async for asynchronous global->shared memory copy
                      (requires dst == "smem"). If False, uses universal copy operation.

        Returns:
            Thread-local partition of the destination tensor after partitioning by ThrCopy.
        """
        if cutlass.const_expr(is_async):
            misc_utils.static_assert(dst == "smem")
            copy_op = "cp.async"
        else:
            copy_op = "universal"

        copy_config = memory_utils.MemoryCopyConfig(
            op=copy_op,
            dtype=src.element_type,
            num_bits_per_copy=src.element_type.width * vector_size,
            tiler_mn=tiler_mn,
            layout_tv=tv_layout,
        )

        # We assume that all copy operations will use the same
        # number of elements as X and hence the same predicate
        crd = cX
        shape = mX.shape

        copy_outputs = memory_utils.copy(
            src=src,
            dst=dst,
            crd=crd,
            shape=shape,
            config=copy_config,
            thread_index=tidx,
            smem_allocator=allocator,
        )
        return copy_outputs.dst_thread, copy_outputs

    def _simple_copy(
        src: cute.Tensor,
        dst: cute.Tensor | str,
    ) -> cute.Tensor:
        """Performs a simple auto-vectorized copy without tiling or thread partitioning.

        Uses autovec_copy for efficient element-wise copying within a single thread context.
        Unlike _tiled_copy, this does not create CopyAtom/TiledCopy machinery or apply
        predicates. Assumes src and dst are already thread-local tensors (typically in
        register memory) and directly copies elements with automatic vectorization.

        Args:
            src: Source thread-local tensor (assumed to be in thread's view).
            dst: Destination thread-local tensor or memspace string ("smem", "rmem").
                 If string, memory will be allocated with the same shape as the source.

        Returns:
            Thread-local destination tensor (same as input dst if tensor, or newly
            allocated tensor if dst was a memspace string).
        """
        copy_outputs = memory_utils.simple_copy(
            src=src,
            dst=dst,
            # `crd` is not used
            crd=None,
            smem_allocator=allocator,
        )
        return copy_outputs.dst_thread

    tXsX, gmem_smem_copy_outputs = _tiled_copy(
        src=gX,
        dst="smem",
        is_async=True,
    )
    # Extract row and column indices from the coordinate tensor. Although these
    # coordinates come from the gmem->smem copy operation, they apply identically
    # to the smem->rmem copy since both operations share the same tiling pattern
    row_index = gmem_smem_copy_outputs.crd_thread[0][0]
    col_index = gmem_smem_copy_outputs.crd_thread[0][1]

    # commit the async copy for `X`
    cute.arch.cp_async_commit_group()

    if cutlass.const_expr(mW is not None):
        tXrW, _ = _tiled_copy(
            src=gW,
            dst="rmem",
            is_async=False,
        )
    if cutlass.const_expr(mB is not None):
        tXrB, _ = _tiled_copy(
            src=gB,
            dst="rmem",
            is_async=False,
        )

    # wait for the async copy operations to finish
    cute.arch.cp_async_wait_group(0)

    tXrX = _simple_copy(
        src=tXsX,
        dst="rmem",
    )
    tYrY = creation_utils.allocate_tensor_like(
        tensor=tXrX,
        memspace="rmem",
        smem_allocator=allocator,
        dtype=mY.element_type,
    )

    # it's usually recommended, if not required, to perform
    # reduction in higher precision
    x = tXrX.load().to(cute.Float32)
    # many of the math ops are available under `cute.math.*` namespace
    z = cute.math.exp(x, fastmath=FAST_MATH)

    # for some historical reasons, the theads shape in
    # TV layout is `(threads_per_row, threads_per_col)`
    # so we need to swap them as the reduction API needs
    # `(threads_per_col, threads_per_row)`
    thread_shape = (
        tv_layout.shape[0][1],  # threads_per_col
        tv_layout.shape[0][0],  # threads_per_row
    )

    if cutlass.const_expr(not online):
        # the `reduce` function takes in a sequence of inputs to
        # reduce and returns a sequence of reduced outputs
        (x_reduced,), reduction_buffer = reduction_utils.reduce(
            (x,),
            op="add",
            thread_shape=thread_shape,
            smem_allocator=allocator,
            # passing `None` means the function will
            # create a buffer and return it if needed
            reduction_buffer=None,
        )

        # when doing multiple sequential reductions, we should reuse the
        # reduction buffer to save shared memory as long as the buffer
        # require is of the same size, dtype, etc
        (z_reduced,), _ = reduction_utils.reduce(
            (z,),
            op="add",
            thread_shape=thread_shape,
            smem_allocator=allocator,
            reduction_buffer=reduction_buffer,
        )

    else:
        # Flexible BlockReductionOp API
        # Reduce multiple values (x, z) together in a single pass - more efficient
        # than separate reduce() calls. Useful for computing multiple statistics
        # (e.g., sum and sum-of-squares for variance).

        # Step 1: Define combine_fn (required)
        # Combines two partially-reduced tuples at warp/block levels.
        # Must be associative for correctness.
        def _combine_fn(
            lhs: tuple[cute.Numeric, cute.Numeric],
            rhs: tuple[cute.Numeric, cute.Numeric],
        ) -> tuple[cute.Numeric, cute.Numeric]:
            x0, z0 = lhs
            x1, z1 = rhs
            new_x = x0 + x1
            new_z = z0 + z1
            return (new_x, new_z)

        # Step 2: Define reduce_ssa (optional but recommended)
        # Optimized thread-level reduction using TensorSSA primitives.
        # If None, falls back to iterative combine_fn.
        def _reduce_ssa(
            xz: tuple[cute.TensorSSA, cute.TensorSSA],
        ) -> tuple[cute.Numeric, cute.Numeric]:
            x0, z0 = xz
            new_x = x0.reduce(
                op=cute.ReductionOp.ADD,
                init_val=x0.element_type(0.),
                reduction_profile=0,
            )
            new_z = z0.reduce(
                op=cute.ReductionOp.ADD,
                init_val=z0.element_type(0.),
                reduction_profile=0,
            )
            return (new_x, new_z)

        # Step 3: Define reduce_wrp (optional but recommended)
        # Custom warp-level reduction for thread-reduced values (post-SSA reduction).
        # Useful for specialized optimizations like:
        #   - Custom shuffle patterns for complex data structures
        #   - Fused operations that combine reduction with other warp-level primitives
        #   - Non-standard reduction operations not supported by default warp_reduce
        # If None, falls back to generic warp_reduce using combine_fn.
        def _reduce_wrp(
            xz: tuple[cute.Numeric, cute.Numeric],
        ) -> tuple[cute.Numeric, cute.Numeric]:
            # This example shows the structure but doesn't provide additional optimization
            # over the default. In practice, you might fuse multiple operations or use
            # custom shuffle patterns for better efficiency with complex reduction logic.
            x0, z0 = xz
            (new_x,) = reduction_utils.warp_reduce(
                (x0,),
                op="add",
                # number of throws per row
                width=min(thread_shape[1], cute.arch.WARP_SIZE),
            )
            (new_z,) = reduction_utils.warp_reduce(
                (z0,),
                op="add",
                # number of throws per row
                width=min(thread_shape[1], cute.arch.WARP_SIZE),
            )
            return (new_x, new_z)

        # Step 4: Construct BlockReductionOp
        # Encapsulates the full reduction strategy:
        #   - init_value: Starting values (0 for sum, -inf for max, etc.)
        #   - reduce_ssa: Thread-level optimization (optional)
        #   - reduce_wrp: Warp-level optimization (optional)
        #   - combine_fn: Binary combiner for warp/block levels (required)
        init_value = (
            x.element_type(0.),
            z.element_type(0.),
        )
        op = reduction_utils.BlockReductionOp(
            combine_fn=_combine_fn,
            reduce_ssa=_reduce_ssa,
            reduce_wrp=_reduce_wrp,
            init_value=init_value,
        )

        # Step 5: Execute hierarchical reduction (thread -> warp -> block)
        # Returns reduced tuple and auto-allocated buffer (if needed for block reduction).
        (x_reduced, z_reduced), reduction_buffer = reduction_utils.reduce(
            # Tuple of tensors to reduce together
            (x, z),
            op=op,
            thread_shape=thread_shape,
            smem_allocator=allocator,
            # Auto-allocate if needed; reusable for subsequent calls
            reduction_buffer=None,
        )

    if cutlass.const_expr(reload == "smem"):
        _simple_copy(
            src=tXsX,
            dst=tXrX,
        )
        x = tXrX.load().to(cute.Float32)

    if cutlass.const_expr(reload == "gmem"):
        _tiled_copy(
            src=gX,
            dst=tXrX,
            is_async=False,
        )
        x = tXrX.load().to(cute.Float32)

    if cutlass.const_expr(FAST_MATH):
        # y = x * (1.0 / denom)
        y = x * cute.arch.rcp_approx(x_reduced + eps)
    else:
        y = x / (x_reduced + eps)

    y = y - cute.math.log(z_reduced, fastmath=FAST_MATH)
    if cutlass.const_expr(mW is not None):
        y = y * tXrW.load().to(cute.Float32)
    if cutlass.const_expr(mB is not None):
        y = y + tXrB.load().to(cute.Float32)

    # store the results
    tYrY.store(y.to(tYrY.element_type))

    # Write the reduction result to mZ if provided. Since mZ has shape [M] (one value
    # per row), only one thread per row needs to write the reduced result. We arbitrarily
    # choose the thread handling column 0 to perform the write 
    if cutlass.const_expr(mZ is not None):
        if col_index == 0:
            mZ[row_index] = mZ.element_type(z_reduced)

    # Copy the results back
    _tiled_copy(
        src=tYrY,
        dst=gY,
        is_async=False,
    )


@cute.jit
def reduction(
    mX: cute.Tensor,
    mW: cute.Tensor | None,
    mB: cute.Tensor | None,
    mY: cute.Tensor,
    mZ: cute.Tensor | None,
    stream: cuda.CUstream,
    eps: cute.Float32,
    online: cutlass.Constexpr,
    reload: cutlass.Constexpr,
) -> None:
    # When working with mixed data types, select a vector size that accommodates
    # ALL tensor arguments (inputs and outputs) for optimal memory coalescing.
    # **IMPORTANT**: With N tensor arguments, compute the maximum element size across
    # ALL N dtypes—not just a subset like mX and mY. Every tensor's dtype must be
    # considered to ensure compatibility and efficient vectorized memory access.
    vector_size = cutlass.const_expr(
        128 //
        cutlass.max(
            mX.element_type.width,
            mW.element_type.width if cutlass.const_expr(mW is not None) else cutlass.const_expr(0),
            mB.element_type.width if cutlass.const_expr(mB is not None) else cutlass.const_expr(0),
            mY.element_type.width,
            mZ.element_type.width if cutlass.const_expr(mZ is not None) else cutlass.const_expr(0),
        )
    )
    if cutlass.const_expr(ENABLE_CLUSTER):
        cluster_size = layout_utils.get_cluster_size_per_row(
            size=mX.shape[1],
            dtype=mX.element_type,
        )
    else:
        cluster_size = cutlass.const_expr(1)

    # Create the tiling configuration and thread-value layout for row-wise reduction.
    # This determines how the 2D input tensor [M, N] is partitioned into tiles and how
    # threads within a block are mapped to data elements for efficient parallel reduction.
    #
    # The function computes:
    # 1. Thread block configuration: (threads_per_col, threads_per_row) using size-based heuristics
    #    to balance register usage, occupancy, and reduction efficiency
    # 2. Vectorization strategy: Groups elements into vector_size chunks for coalesced memory access
    # 3. Block iteration: Splits work across multiple iterations if needed for very large row sizes
    #
    # Returns:
    # - tiler_mn: Tile shape (M_tile, N_tile) defining each block's workload
    #   * M_tile = number of rows processed per block
    #   * N_tile = total elements per row processed per block (accounting for vectorization and iterations)
    #
    # - tv_layout: Thread-Value layout with hierarchical structure
    #   * Shape: ((threads_per_row, threads_per_col), (vector_size, num_blocks_per_row))
    #   * Maps thread IDs to vectorized data chunks across multiple iteration blocks
    #   * Stride pattern ensures coalesced memory access within each block iteration
    tiler_mn, tv_layout = layout_utils.make_2D_row_reduction_layout(
        size=mX.shape[1],               # Number of columns (N) to reduce across
        dtype=mX.element_type,          # Element type for alignment and heuristics
        num_threads=None,               # Auto-select thread config based on size
        cluster_size=cluster_size,      # Thread block cluster size (typically 1, increases for very wide reductions)
        num_bits_per_copy=(             # Vectorization width: typically 128 bits for optimal coalescing
            vector_size *               # (e.g., 8×fp16 or 4×fp32 loads per thread)
            mX.element_type.width
        ),
    )

    # Unsqueeze operations add broadcasting dimensions (with stride-0) to make 1D tensors
    # compatible with 2D tiling operations. This is a zero-copy transformation that modifies
    # only the tensor's layout metadata, not the underlying memory.
    #
    # How stride-0 broadcasting works:
    # - A dimension with stride-0 means advancing along that axis doesn't change the memory address
    # - For tensor[i, j] with stride [0, s], all values tensor[*, j] point to the same memory location
    # - This enables implicit broadcasting without data duplication
    #
    # Transformations applied:
    # 1. mW and mB: [N] -> [1, N] with stride [0, *] (prepend dimension via dim=0)
    #    - Original: Row vectors containing per-column weights/biases
    #    - After: 2D tensors where each row shares the same values
    #    - Usage: Enables local_tile operations to extract tiles using 2D indexing, where the
    #      first index is always 0 since all M rows share identical weight/bias values
    #    - Broadcast semantics: mW[i, j] accesses mW_original[j] for any i
    #
    # 2. mZ: [M] -> [M, 1] with stride [*, 0] (append dimension via dim=-1)
    #    - Original: Column vector for per-row reduction results
    #    - After: 2D tensor where each column shares the same values
    #    - Usage: Makes mZ compatible with 2D block-level tiling expecting shape [M, N]
    #    - Broadcast semantics: mZ[i, j] accesses mZ_original[i] for any j
    if cutlass.const_expr(mW is not None):
        mW = layout_utils.unsqueeze(mW, dim=0, size=tiler_mn[0])
    if cutlass.const_expr(mB is not None):
        mB = layout_utils.unsqueeze(mB, dim=0, size=tiler_mn[0])
    if cutlass.const_expr(mZ is not None):
        mZ = layout_utils.unsqueeze(mZ, dim=-1, size=mX.shape[1])

    num_blocks = cute.ceil_div(mX.shape[0], tiler_mn[0])
    num_threads = cute.size(tv_layout, mode=[0])
    num_clusters = [1, cluster_size, 1] if cutlass.const_expr(cluster_size > 1) else None

    kernel = reduction_kernel(
        mX,
        mW,
        mB,
        mY,
        mZ,
        eps,
        tiler_mn,
        tv_layout,
        online,
        reload,
        vector_size,
        cluster_size,
    )
    kernel.launch(
        grid=[num_blocks, cluster_size, 1],
        block=[num_threads, 1, 1],
        cluster=num_clusters,
        smem=None,
        stream=stream,
    )
    launch_check(kernel)


def to_cute_tensor(torch_tensor: torch.Tensor, assumed_align: int = 16, dynamic: bool = True) -> cute.Tensor:
    torch_tensor = torch_tensor.detach()
    cute_tensor = cute.runtime.from_dlpack(
        torch_tensor,
        assumed_align=assumed_align,
    )
    if dynamic:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=0,
            stride_order=torch_tensor.dim_order(),
        )

    return cute_tensor


def _reduction_op(
    x: torch.Tensor,
    w: torch.Tensor | None,
    b: torch.Tensor | None,
    y: torch.Tensor,
    z: torch.Tensor | None,
    eps: float,
    online: bool,
) -> None:
    if x.dim() != 2:
        raise ValueError("Input must be 2D")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    if x.dtype not in ALLOWED_DTYPES:
        raise ValueError("Unsupported dtype")
    if w is not None:
        if w.dim() != 1:
            raise ValueError("Weight must be 1D")
        if x.shape[-1] != w.shape[0]:
            raise ValueError("Last dimension of input must match weight dimension")
        if not w.is_cuda:
            raise ValueError("Weight tensor must be on CUDA device")
        if w.dtype not in ALLOWED_DTYPES:
            raise ValueError("Weight must be float32, float16 or bfloat16")

    _, N = x.shape
    x_tensor = to_cute_tensor(x)
    w_tensor = to_cute_tensor(w, dynamic=False)   if w is not None else None
    b_tensor = to_cute_tensor(b, dynamic=False)   if b is not None else None
    y_tensor = to_cute_tensor(y)
    z_tensor = to_cute_tensor(z, assumed_align=4) if z is not None else None

    # https://github.com/Dao-AILab/quack/issues/6
    # When the input tensor is too large, loading it would cause register spills.
    # This flag reloads from smem instead of keeping it in registers, to free up some registers
    reload = None if N <= 16384 else "smem"

    # Get current CUstream from torch
    current_stream = cutlass_torch.current_stream()
    compile_key = (
        N,
        x.dtype,
        w.dtype if w is not None else None,
        b.dtype if b is not None else None,
        y.dtype,
        z.dtype if z is not None else None,
        online,
        reload,
    )
    if compile_key not in _reduction_op.compile_cache:
        warnings.warn(
            f"Kernel compilation cache miss for shape={x.shape[1]}, x.dtype={x.dtype}, "
            f"y.dtype={y.dtype}. This will trigger JIT compilation and may be slow.",
        )
        _reduction_op.compile_cache[compile_key] = cute.compile(
            reduction,
            x_tensor,
            w_tensor,
            b_tensor,
            y_tensor,
            z_tensor,
            current_stream,
            eps,
            online,
            reload,
        )

    _reduction_op.compile_cache[compile_key](
        x_tensor,
        w_tensor,
        b_tensor,
        y_tensor,
        z_tensor,
        current_stream,
        eps,
    )


_reduction_op.compile_cache = {}


def reduction_op(
    x: torch.Tensor,
    w: torch.Tensor | None,
    b: torch.Tensor | None,
    dtype: torch.dtype | None = None,
    eps: float = 0.,
    online: bool = False,
    return_z: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if dtype is None:
        dtype = x.dtype
    y = torch.empty_like(x, dtype=dtype)
    if return_z:
        z = torch.empty(
            x.shape[0],
            dtype=torch.float32,
            device=x.device,
        )
    else:
        z = None
    _reduction_op(
        x=x,
        w=w,
        b=b,
        y=y,
        z=z,
        eps=eps,
        online=online,
    )
    return y, z
