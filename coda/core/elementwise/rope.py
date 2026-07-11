import operator
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
from coda.core.ops import creation_utils

_NUM_BITS = 128


@cute.kernel
def qknorm_rope_kernel(
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
    val_m: cutlass.Constexpr[int],
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

    for row_index in cutlass.range_constexpr(val_m):
        row_coord, col_coord_begin = tXcX_packed[row_index * vector_size]
        if row_coord < mX_packed.shape[0]:
            head_idx = (2 * col_coord_begin) // head_dim
            rms = cute.Float32(1.0)
            if head_idx < num_heads:
                ssq = cute.Float32.zero
                for i in cutlass.range_constexpr(num_segments):
                    ssq = ssq + mSSq[row_coord, head_idx * num_segments + i]
                rms = cute.math.rsqrt(ssq / head_dim + eps, fastmath=True)

            for col_index in cutlass.range_constexpr(vector_size):
                flat_index = row_index * vector_size + col_index
                _, col_coord = tXcX_packed[flat_index]
                a = mPos[row_coord].to(dtype=cute.Float32) * mFreq[col_coord].to(dtype=cute.Float32)
                c = cute.math.cos(a, fastmath=True)
                s = cute.math.sin(a, fastmath=True)
                x = tXrX[2 * flat_index].to(dtype=cute.Float32) * mGamma[2 * col_coord].to(dtype=cute.Float32) * rms
                y = tXrX[2 * flat_index + 1].to(dtype=cute.Float32) * mGamma[2 * col_coord + 1].to(dtype=cute.Float32) * rms
                tXrY[2 * flat_index] = (x * c + y * s).to(dtype=tXrY.element_type)
                tXrY[2 * flat_index + 1] = (y * c - x * s).to(dtype=tXrY.element_type)

    _ = memory_utils.copy(
        src=tXrY_packed,
        dst=gY_packed,
        crd=tXcX_packed,
        shape=mY_packed.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )


@cute.jit
def _qknorm_rope(
    mX: cute.Tensor,
    mY: cute.Tensor,
    mSSq: cute.Tensor,
    mGamma: cute.Tensor,
    mPos: cute.Tensor,
    mFreq: cute.Tensor,
    head_dim: cutlass.Constexpr[int],
    num_heads: cutlass.Constexpr[int],
    num_segments: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    thr_m: cutlass.Constexpr[int],
    thr_n: cutlass.Constexpr[int],
    val_m: cutlass.Constexpr[int],
    stream: cuda.CUstream,
) -> int:
    mX_packed = cute.recast_tensor(mX, dtype=cute.Int32)
    mY_packed = cute.recast_tensor(mY, dtype=cute.Int32)
    vector_size = cutlass.const_expr(_NUM_BITS // mX_packed.element_type.width)
    misc_utils.static_assert(len(mX_packed.shape) == 2)
    misc_utils.static_assert(len(mY_packed.shape) == 2)
    misc_utils.static_assert(len(mSSq.shape) == 2)
    misc_utils.static_assert(len(mGamma.shape) == 1)
    misc_utils.static_assert(len(mPos.shape) == 1)
    misc_utils.static_assert(len(mFreq.shape) == 1)
    misc_utils.static_assert(mX.shape[1] % head_dim == 0)
    misc_utils.static_assert(mX_packed.shape[1] == mY_packed.shape[1])
    misc_utils.static_assert(mX_packed.shape[1] % vector_size == 0)
    misc_utils.static_assert(mX_packed.shape[1] % (thr_n * vector_size) == 0)
    tiler_mn, tv_layout = layout_utils.make_layout_tv_from_shape(
        thread_shape=(thr_m, thr_n),
        thread_order="row",
        value_shape=(val_m, vector_size),
        value_order="row",
    )

    # ((TileM, TileN), (RestM, RestN))
    gX_packed = cute.zipped_divide(mX_packed, tiler_mn)
    num_blocks = gX_packed.shape[1]
    num_threads = cute.size(tv_layout, mode=[0])
    misc_utils.static_assert(len(num_blocks) == 2)
    kernel = qknorm_rope_kernel(
        mX_packed=mX_packed,
        mY_packed=mY_packed,
        mSSq=mSSq,
        mGamma=mGamma,
        mPos=mPos,
        mFreq=mFreq,
        head_dim=head_dim,
        num_heads=num_heads,
        num_segments=num_segments,
        eps=eps,
        dtype=mX.element_type,
        tiler_mn=tiler_mn,
        tv_layout=tv_layout,
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
def _compile_qknorm_rope(
    size: int,
    head_dim: int,
    num_heads: int,
    num_segments: int,
    eps: float,
    dtype: type[cute.Numeric],
    pos_dtype: type[cute.Numeric],
    freq_dtype: type[cute.Numeric],
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> Callable:
    m = cute.sym_int()
    vector_size = cutlass.const_expr(_NUM_BITS // dtype.width)
    misc_utils.static_assert(size % head_dim == 0)
    misc_utils.static_assert(vector_size % 2 == 0)
    mX = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mY = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(m, size),
        stride=(cute.sym_int64(divisibility=vector_size), 1),
        assumed_align=16,
    )
    mSSq = cute.runtime.make_fake_tensor(
        dtype=cute.Float32,
        shape=(m, (size // head_dim) * num_segments),
        stride=(cute.sym_int64(divisibility=1), 1),
        assumed_align=4,
    )
    mGamma = cute.runtime.make_fake_tensor(
        dtype=dtype,
        shape=(size,),
        stride=(1,),
        assumed_align=dtype.width // 8,
    )
    mPos = cute.runtime.make_fake_tensor(
        dtype=pos_dtype,
        shape=(m,),
        stride=(1,),
        assumed_align=pos_dtype.width // 8,
    )
    mFreq = cute.runtime.make_fake_tensor(
        dtype=freq_dtype,
        shape=(size // 2,),
        stride=(1,),
        assumed_align=freq_dtype.width // 8,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _qknorm_rope,
        mX=mX,
        mY=mY,
        mSSq=mSSq,
        mGamma=mGamma,
        mPos=mPos,
        mFreq=mFreq,
        head_dim=head_dim,
        num_heads=num_heads,
        num_segments=num_segments,
        eps=eps,
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
        stream=stream,
        options="--enable-tvm-ffi",
    )


def qknorm_rope_(
    x: torch.Tensor,
    y: torch.Tensor,
    ssq: torch.Tensor,
    gamma: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads: int,
    num_segments: int,
    eps: float,
    thr_m: int,
    thr_n: int,
    val_m: int,
) -> None:
    fn = _compile_qknorm_rope(
        size=x.shape[1],
        head_dim=head_dim,
        num_heads=num_heads,
        num_segments=num_segments,
        eps=eps,
        dtype=torch2cute_dtype_map[x.dtype],
        pos_dtype=torch2cute_dtype_map[pos.dtype],
        freq_dtype=torch2cute_dtype_map[freq.dtype],
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    fn(x, y, ssq, gamma, pos, freq)
