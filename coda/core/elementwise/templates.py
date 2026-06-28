import torch
import warnings
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda
from typing import Callable

from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops.misc_utils import static_assert
from coda.core.ops.launch_utils import launch_check


@cute.kernel
def elementwise_apply_kernel(
    fn: cutlass.Constexpr[Callable],
    mX: cute.Tensor,
    mY: cute.Tensor,
    mZ: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idX = cute.make_identity_tensor(mX.shape)
    gX = cute.local_tile(mX, tiler_mn, (bidx, bidy))
    gY = cute.local_tile(mY, tiler_mn, (bidx, bidy))
    gZ = cute.local_tile(mZ, tiler_mn, (bidx, bidy))
    cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))
    config_X = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mX.element_type,
        num_bits_per_copy=mX.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    config_Y = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mY.element_type,
        num_bits_per_copy=mY.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    config_Z = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mZ.element_type,
        num_bits_per_copy=mZ.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    copy_outputs_X = memory_utils.copy(
        src=gX,
        dst="rmem",
        crd=cX,
        shape=mX.shape,
        config=config_X,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    copy_outputs_Y = memory_utils.copy(
        src=gY,
        dst="rmem",
        crd=cX,
        shape=mY.shape,
        config=config_Y,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    tXrX = copy_outputs_X.dst_thread
    tYrY = copy_outputs_Y.dst_thread
    tZrZ = creation_utils.allocate_tensor_like(
        tensor=tXrX,
        memspace="rmem",
        smem_allocator=allocator,
        dtype=mZ.element_type,
    )

    # apply custom function
    fn(tXrX, tYrY, tZrZ)

    # Copy the results back
    _ = memory_utils.copy(
        src=tZrZ,
        dst=gZ,
        crd=copy_outputs_X.crd_thread,
        shape=mZ.shape,
        config=config_Z,
        thread_index=tidx,
        smem_allocator=allocator,
    )


@cute.jit
def elementwise_apply(
    fn: cutlass.Constexpr[Callable],
    mX: cute.Tensor,
    mY: cute.Tensor,
    mZ: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    vector_size = cutlass.const_expr(
        128 //
        cutlass.max(
            mX.element_type.width,
            mY.element_type.width,
            mZ.element_type.width,
        )
    )
    tiler_mn, tv_layout = layout_utils.make_2D_elementwise_layout(
        dtype=mX.element_type,
        num_bits_per_copy=(
            vector_size *
            mX.element_type.width
        ),
    )

    # ((TileM, TileN), (RestM, RestN))
    gX = cute.zipped_divide(mX, tiler_mn)
    num_blocks = gX.shape[1]
    num_threads = cute.size(tv_layout, mode=[0])
    static_assert(len(num_blocks) == 2)
    kernel = elementwise_apply_kernel(
        fn,
        mX,
        mY,
        mZ,
        tiler_mn,
        tv_layout,
        vector_size,
    )
    kernel.launch(
        grid=[*num_blocks, 1],
        block=[num_threads, 1, 1],
        cluster=None,
        smem=None,
        stream=stream,
    )
