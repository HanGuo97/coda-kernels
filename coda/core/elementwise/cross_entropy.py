import cutlass
import cutlass.cute as cute
from quack.cache import jit_cache

from coda.core.ops import memory_utils
from coda.core.ops import layout_utils
from coda.core.ops.misc_utils import static_assert


@cute.kernel
def cross_entropy_fwd_bwd_kernel(
    mLogits: cute.Tensor,
    mLSE: cute.Tensor,
    mTarget: cute.Tensor,
    mLoss: cute.Tensor,
    ignore_index: cutlass.Constexpr[int],
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idLogits = cute.make_identity_tensor(mLogits.shape)
    gLogits = cute.local_tile(mLogits, tiler_mn, (bidx, bidy))
    cLogits = cute.local_tile(idLogits, tiler_mn, (bidx, bidy))
    config = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mLogits.element_type,
        num_bits_per_copy=mLogits.element_type.width * vector_size,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    copy_outputs = memory_utils.copy(
        src=gLogits,
        dst="rmem",
        crd=cLogits,
        shape=mLogits.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    tXrLogits = copy_outputs.dst_thread
    tXcLogits = copy_outputs.crd_thread

    _ = memory_utils.copy(
        src=tXrLogits,
        dst=gLogits,
        crd=tXcLogits,
        shape=mLogits.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )
