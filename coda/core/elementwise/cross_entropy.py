import torch
import cutlass
import cutlass.cute as cute
from quack.cache import jit_cache
from coda.core.ops import misc_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils


_NUM_BITS = 128


@cute.kernel
def cross_entropy_fwd_bwd_kernel(
    mLogits: cute.Tensor,
    mLSE: cute.Tensor,
    mTarget: cute.Tensor,
    mLoss: cute.Tensor,
    ignore_index: cutlass.Constexpr[int],
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    num_rows: cutlass.Constexpr[int],
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idLogits = cute.make_identity_tensor(mLogits.shape)
    gLogits = cute.local_tile(mLogits, tiler_mn, (bidx, bidy))
    cLogits = cute.local_tile(idLogits, tiler_mn, (bidx, bidy))
    misc_utils.static_assert(vector_size == _NUM_BITS // mLogits.element_type.width)
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

    for row_index in cutlass.range_constexpr(num_rows):
        row_coord, _ = tXcLogits[row_index * vector_size]
        lse = mLSE[row_coord]
        target = mTarget[row_coord]
        ignored = target == ignore_index

        for col_index in cutlass.range_constexpr(vector_size):
            flat_index = row_index * vector_size + col_index
            _, col_coord = tXcLogits[flat_index]
            logits = tXrLogits[flat_index].to(dtype=cute.Float32)
            probs = cute.math.exp(logits - lse, fastmath=True)
            dlogits = probs

            if col_coord == target:
                dlogits = probs - 1.0
                if not ignored:
                    mLoss[row_coord] = lse - logits

            if ignored:
                dlogits = cute.Float32.zero

            tXrLogits[flat_index] = dlogits.to(dtype=tXrLogits.element_type)

    _ = memory_utils.copy(
        src=tXrLogits,
        dst=gLogits,
        crd=tXcLogits,
        shape=mLogits.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )
