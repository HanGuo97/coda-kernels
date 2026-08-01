import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from typing import Callable

from quack.cache import jit_cache
from quack.cute_dsl_utils import torch2cute_dtype_map
from coda.core.ops.constants import NUM_BITS_PER_COPY
from coda.core.ops import misc_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops import reduction_utils


@cute.kernel
def cross_entropy_fwd_bwd_kernel(
    mLogits: cute.Tensor,
    mLSE: cute.Tensor,
    mTarget: cute.Tensor,
    mLoss: cute.Tensor,
    mZdZ: cute.Tensor | None,
    ignore_index: cutlass.Constexpr[int],
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    vector_size: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idLogits = cute.make_identity_tensor(mLogits.shape)
    gLogits = cute.local_tile(mLogits, tiler_mn, (bidx, bidy))
    cLogits = cute.local_tile(idLogits, tiler_mn, (bidx, bidy))

    is_full_tile = cutlass.const_expr(mLogits.shape[1] % tiler_mn[1] == 0)
    misc_utils.static_assert(vector_size == NUM_BITS_PER_COPY // mLogits.element_type.width)

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

    if cutlass.const_expr(mZdZ is not None):
        rZdZ = creation_utils.allocate_tensor_from_shape(
            shape=(val_m,),
            order="row",
            dtype=cute.Float32,
            memspace="rmem",
        )
        rZdZ.fill(value=0.0)

    for row_index in cutlass.range_constexpr(val_m):
        row_coord, _ = tXcLogits[row_index * vector_size]
        row_in_bound = row_coord < mLogits.shape[0]
        row_coord_safe = row_coord
        if not row_in_bound:
            row_coord_safe = 0
        lse = mLSE[row_coord_safe]
        target = mTarget[row_coord_safe]

        ignored = target == ignore_index

        for col_index in cutlass.range_constexpr(vector_size):
            flat_index = row_index * vector_size + col_index
            _, col_coord = tXcLogits[flat_index]

            if cutlass.const_expr(is_full_tile) or col_coord < mLogits.shape[1]:
                logits = tXrLogits[flat_index].to(dtype=cute.Float32)
                probs = cute.math.exp(logits - lse, fastmath=True)
                dlogits = probs

                if col_coord == target:
                    dlogits = probs - 1.0
                    if not ignored and row_in_bound:
                        mLoss[row_coord] = lse - logits

                if ignored:
                    dlogits = cute.Float32.zero

                if cutlass.const_expr(mZdZ is not None):
                    rZdZ[row_index] = rZdZ[row_index] + logits * dlogits

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

    if cutlass.const_expr(mZdZ is not None):
        zdzs = []
        for row_index in cutlass.range_constexpr(val_m):
            zdz = cute.make_rmem_tensor((1,), cute.Float32)
            zdz[0] = rZdZ[row_index]
            zdzs.append(zdz.load())
        zdzs_reduced, _ = reduction_utils.reduce(
            zdzs,
            op="add",
            thread_shape=(thr_m, thr_n),
            smem_allocator=allocator,
            reduction_buffer=None,
        )
        for row_index in cutlass.range_constexpr(val_m):
            row_coord, _ = tXcLogits[row_index * vector_size]
            row_in_bound = row_coord < mLogits.shape[0]
            if ((tidx % thr_n) == 0) and row_in_bound:
                mZdZ[row_coord, bidy] = zdzs_reduced[row_index]


@cute.jit
def _cross_entropy_fwd_bwd(
    mLogits: cute.Tensor,
    mLSE: cute.Tensor,
    mTarget: cute.Tensor,
    mLoss: cute.Tensor,
    mZdZ: cute.Tensor | None,
    ignore_index: cutlass.Constexpr[int],
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    stream: cuda.CUstream,
) -> int:
    vector_size = cutlass.const_expr(NUM_BITS_PER_COPY // mLogits.element_type.width)
    misc_utils.static_assert(len(mLogits.shape) == 2)
    misc_utils.static_assert(len(mLSE.shape) == 1)
    misc_utils.static_assert(len(mTarget.shape) == 1)
    misc_utils.static_assert(len(mLoss.shape) == 1)
    if cutlass.const_expr(mZdZ is not None):
        misc_utils.static_assert(len(mZdZ.shape) == 2)
    misc_utils.static_assert(mLogits.shape[1] % vector_size == 0)
    tiler_mn, tv_layout = layout_utils.make_layout_tv_from_shape(
        thread_shape=(thr_m, thr_n),
        thread_order="row",
        value_shape=(val_m, vector_size),
        value_order="row",
    )

    # ((TileM, TileN), (RestM, RestN))
    gLogits = cute.zipped_divide(mLogits, tiler_mn)
    num_blocks = gLogits.shape[1]
    num_threads = cute.size(tv_layout, mode=[0])
    misc_utils.static_assert(len(num_blocks) == 2)
    kernel = cross_entropy_fwd_bwd_kernel(
        mLogits=mLogits,
        mLSE=mLSE,
        mTarget=mTarget,
        mLoss=mLoss,
        mZdZ=mZdZ,
        ignore_index=ignore_index,
        tiler_mn=tiler_mn,
        tv_layout=tv_layout,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        vector_size=vector_size,
    )
    kernel.launch(
        grid=[*num_blocks, 1],
        block=[num_threads, 1, 1],
        cluster=None,
        smem=None,
        stream=stream,
    )
    return kernel.smem_usage()


@jit_cache
def _compile_cross_entropy_fwd_bwd(
    vocab_size: int,
    n_tiles: int | None,
    ignore_index: int,
    logits_dtype: type[cute.Numeric],
    target_dtype: type[cute.Numeric],
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> Callable:
    m = cute.sym_int()
    vector_size = cutlass.const_expr(NUM_BITS_PER_COPY // logits_dtype.width)
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
    if n_tiles is None:
        mZdZ = None
    else:
        mZdZ = cute.runtime.make_fake_tensor(
            dtype=cute.Float32,
            shape=(m, n_tiles),
            stride=(n_tiles, 1),
            assumed_align=4,
        )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _cross_entropy_fwd_bwd,
        mLogits=mLogits,
        mLSE=mLSE,
        mTarget=mTarget,
        mLoss=mLoss,
        mZdZ=mZdZ,
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
    partials: torch.Tensor | None,
    ignore_index: int,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    assert target.dtype == torch.int32
    fn = _compile_cross_entropy_fwd_bwd(
        vocab_size=logits.shape[1],
        n_tiles=partials.shape[1] if partials is not None else None,
        ignore_index=ignore_index,
        logits_dtype=torch2cute_dtype_map[logits.dtype],
        target_dtype=torch2cute_dtype_map[target.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(logits, lses, target, losses, partials)
