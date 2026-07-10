import operator
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.cache import jit_cache
from coda.core.ops import misc_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils

_NUM_BITS = 128


@cute.kernel
def rope_kernel(
    mX_packed: cute.Tensor,
    mY_packed: cute.Tensor,
    mSSq: cute.Tensor,
    mGamma: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    head_dim: cutlass.Constexpr[int],
    num_heads: cutlass.Constexpr[int],
    num_segments: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    dtype: type[cute.Numeric],
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idX_packed = cute.make_identity_tensor(mX_packed.shape)
    gX_packed = cute.local_tile(mX_packed, tiler_mn, (bidx, bidy))
    gY_packed = cute.local_tile(mY_packed, tiler_mn, (bidx, bidy))
    cX_packed = cute.local_tile(idX_packed, tiler_mn, (bidx, bidy))
    config = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mX_packed.element_type,
        num_bits_per_copy=mX_packed.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    copy_outputs = memory_utils.copy(
        src=gX_packed,
        dst="rmem",
        crd=cX_packed,
        shape=mX_packed.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    tXrX_packed = copy_outputs.dst_thread
    tXcX_packed = copy_outputs.crd_thread
    tXrY_packed = creation_utils.allocate_tensor_like(
        tensor=tXrX_packed,
        memspace="rmem",
        smem_allocator=allocator,
        dtype=mY_packed.element_type,
    )
    tXrX = cute.recast_tensor(tXrX_packed, dtype=dtype)
    tXrY = cute.recast_tensor(tXrY_packed, dtype=dtype)

    _ = memory_utils.copy(
        src=tXrY_packed,
        dst=gY_packed,
        crd=tXcX_packed,
        shape=mY_packed.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )
