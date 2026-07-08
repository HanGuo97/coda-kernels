import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.cache import jit_cache
from quack.cute_dsl_utils import torch2cute_dtype_map
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


@jit_cache
def _compile_cross_entropy_fwd_bwd(
    vocab_size: int,
    ignore_index: int,
    logits_dtype: type[cute.Numeric],
    target_dtype: type[cute.Numeric],
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> Callable:
    m = cute.sym_int(divisibility=thr_m * val_m)
    vector_size = cutlass.const_expr(_NUM_BITS // logits_dtype.width)
    target_align = cutlass.const_expr(target_dtype.width // 8)
    mLogits = cute.runtime.make_fake_tensor(
        dtype=logits_dtype,
        shape=(m, vocab_size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mLSE = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m,),
        stride=(1,),
        assumed_align=4,
    )
    mTarget = cute.runtime.make_fake_tensor(
        dtype=target_dtype,
        shape=(m,),
        stride=(1,),
        assumed_align=target_align,
    )
    mLoss = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m,),
        stride=(1,),
        assumed_align=4,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _cross_entropy_fwd_bwd,
        mLogits=mLogits,
        mLSE=mLSE,
        mTarget=mTarget,
        mLoss=mLoss,
        ignore_index=ignore_index,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        stream=stream,
        options="--enable-tvm-ffi",
    )


def cross_entropy_fwd_bwd_(
    logits: torch.Tensor,
    lses: torch.Tensor,
    target: torch.Tensor,
    losses: torch.Tensor,
    ignore_index: int,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    assert target.dtype == torch.int32
    fn = _compile_cross_entropy_fwd_bwd(
        vocab_size=logits.shape[1],
        ignore_index=ignore_index,
        logits_dtype=torch2cute_dtype_map[logits.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(logits, lses, target, losses)
