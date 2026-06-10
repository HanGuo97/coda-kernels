# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    op: cutlass.Constexpr,
    mX: cute.Tensor,
    mY: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    precise: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idX = cute.make_identity_tensor(mX.shape)
    gX = cute.local_tile(mX, tiler_mn, (bidx, bidy))
    gY = cute.local_tile(mY, tiler_mn, (bidx, bidy))
    cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))
    config = memory_utils.MemoryCopyConfig(
        op="universal",
        dtype=mX.element_type,
        num_bits_per_copy=128,
        tiler_mn=tiler_mn,
        layout_tv=tv_layout,
    )
    copy_outputs = memory_utils.copy(
        src=gX,
        dst="rmem",
        crd=cX,
        shape=mX.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )
    tXrX = copy_outputs.dst_thread
    tYrY = creation_utils.allocate_tensor_like(
        tensor=tXrX,
        memspace="rmem",
        smem_allocator=allocator,
        dtype=mY.element_type,
    )

    # Note: Maintain TensorSSA values in higher precision (e.g., Float32) when:
    # (1) Operations are precision-critical, requiring minimal accumulated rounding errors
    # (2) Operations mandate higher precision (e.g., mathematical functions like exp, log, sqrt)
    # (3) Multiple arithmetic operations are chained before final conversion
    # (4) Working with low-precision types (Float16, BFloat16) that lack sufficient dynamic range
    # This approach keeps intermediate computations accurate before converting to the target dtype.
    if cutlass.const_expr(precise):
        X_vec = tXrX.load().to(cute.Float32)
    else:
        X_vec = tXrX.load()

    # Load data before use. The compiler will optimize the copy and load
    # operations to convert some memory ld/st into register uses.
    result = op(X_vec).to(mY.element_type)

    # Save the results back to registers. Here we reuse b's registers.
    tYrY.store(result)

    # Copy the results back
    _ = memory_utils.copy(
        src=tYrY,
        dst=gY,
        crd=cX,
        shape=mY.shape,
        config=config,
        thread_index=tidx,
        smem_allocator=allocator,
    )


@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    mX: cute.Tensor,
    mY: cute.Tensor,
    precise: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    tiler_mn, tv_layout = layout_utils.make_2D_elementwise_layout(
        dtype=mX.element_type,
        num_bits_per_copy=128,
    )

    # ((TileM, TileN), (RestM, RestN))
    gX = cute.zipped_divide(mX, tiler_mn)
    num_blocks = gX.shape[1]
    num_threads = cute.size(tv_layout, mode=[0])
    static_assert(len(num_blocks) == 2)
    kernel = elementwise_apply_kernel(
        op,
        mX,
        mY,
        tiler_mn,
        tv_layout,
        precise,
    )
    kernel.launch(
        grid=[*num_blocks, 1],
        block=[num_threads, 1, 1],
        cluster=None,
        smem=None,
        stream=stream,
    )
    launch_check(kernel)


def to_cute_tensor(torch_tensor: torch.Tensor) -> cute.Tensor:
    torch_tensor = torch_tensor.detach()
    cute_tensor = cute.runtime.from_dlpack(
        torch_tensor,
        assumed_align=16,
    )
    return cute_tensor.mark_compact_shape_dynamic(
        mode=0,
        stride_order=(0, 1),
    )


def _elementwise_op(
    op: Callable[[cute.TensorSSA], cute.TensorSSA],
    x: torch.Tensor,
    y: torch.Tensor,
    precise: bool = True,
) -> None:
    if x.ndim != 2:
        raise ValueError("Input and output must be 2D")
    if not x.is_cuda:
        raise ValueError("Tensor must be on CUDA device")
    if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        raise ValueError("Unsupported dtype")
    if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
        raise NotImplementedError

    x_tensor = to_cute_tensor(x)
    y_tensor = to_cute_tensor(y)

    # Get current CUstream from torch
    current_stream = cutlass_torch.current_stream()
    # we use the op, row-size, and dtype as the cache key
    compile_key = (
        op,
        x.shape[1],
        x.dtype,
        precise,
    )
    if compile_key not in _elementwise_op.compile_cache:
        warnings.warn(
            f"Kernel compilation cache miss for op={op}, shape={x.shape[1]}, "
            f"dtype={x.dtype}. This will trigger JIT compilation and may be slow.",
        )
        _elementwise_op.compile_cache[compile_key] = cute.compile(
            elementwise_apply,
            op,
            x_tensor,
            y_tensor,
            precise,
            current_stream,
        )
    # When compiled we inlined Constexpr in the kernel, so we do not pass it when evaluating at runtime
    _elementwise_op.compile_cache[compile_key](
        x_tensor,
        y_tensor,
        current_stream,
    )


_elementwise_op.compile_cache = {}


def elementwise_op(
    op: Callable[[cute.TensorSSA], cute.TensorSSA],
    x: torch.Tensor,
    precise: bool = True,
) -> torch.Tensor:
    """Apply elementwise operation to input tensor.

    **Important:** `op` is used in the compilation cache key. Define `op` at
    module level (not inside functions) to avoid cache misses from closure
    identity changes on every call.

    Cache miss every call:
    ```python
    def f(x: torch.Tensor) -> torch.Tensor:

        def _op(tensor_ssa: cute.TensorSSA) -> cute.TensorSSA:
            return tensor_ssa

        x = x.detach()
        return elementwise_op(_op, x)
    ```

    Cache hit:
    ```python
    def _op(tensor_ssa: cute.TensorSSA) -> cute.TensorSSA:
        return tensor_ssa

    def f(x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        return elementwise_op(_op, x)
    ```
    """
    y = torch.empty_like(x)
    _elementwise_op(
        op=op,
        x=x,
        y=y,
        precise=precise,
    )
    return y
