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
from typing import Callable, NamedTuple
from torch.utils import _pytree as pytree
from quack.autotuner import autotune, AutotuneConfig
from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm_tvm_ffi_utils import (
    make_scheduler_args,
    make_fake_scheduler_args,
)

from hilt.dtype_utils import get_dtype
from rapier.ops.gemm_utils import (
    get_major,
    is_valid_dtypes,
    is_valid_tensor_alignment,
)
from rapier.gemm import gemm_quack
from rapier.epilogue import (
    EVTNoOp,
    EpilogueVisitorTree,
)

# Default: Quack's GEMM template with Ping-Pong Persistent kernel.
# Use autotuning for optimal performance.
ACC_DTYPE = cute.Float32
USE_QUACK = True
IS_PERSISTENT = True
PERMUTE_BATCH = False
DYNAMIC_SCHEDULER = False
DEFAULT_PINGPONG = True
DEFAULT_TILE_M = 128
DEFAULT_TILE_N = 128
DEFAULT_CLUSTER_M = 1
DEFAULT_CLUSTER_N = 1


def to_cute_tensor(
    torch_tensor: torch.Tensor,
    assumed_align: int = 16,
    leading_dim: int | None = None,
) -> cute.Tensor:
    torch_tensor = torch_tensor.detach()
    cute_tensor = cute.runtime.from_dlpack(
        torch_tensor,
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    )
    if leading_dim is not None:
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=leading_dim
        )

    return cute_tensor


def to_cute_tensor_or_vector(
    torch_tensor: torch.Tensor,
) -> cute.Tensor:
    assert isinstance(torch_tensor, torch.Tensor)

    # [L, *]
    if torch_tensor.ndim == 2:
        assert torch_tensor.shape[0] == 1
        if torch_tensor.dtype.itemsize == 8:
            # occasionally using 4 will break `cp.async`
            assumed_align = 8
        else:
            assumed_align = 4

    # [L, *, *] or [*, *, L]
    elif torch_tensor.ndim == 3:
        assert torch_tensor.shape[0] == 1 or torch_tensor.shape[2] == 1
        assumed_align = 16

    else:
        raise NotImplementedError

    return to_cute_tensor(
        torch_tensor,
        assumed_align=assumed_align,
    )


def preprocess_vector(
    x: torch.Tensor,
    permute: bool,
) -> torch.Tensor:
    if x.dim() == 1:
        x = torch.unsqueeze(x, dim=0)
    if x.dim() != 2:
        raise ValueError("Input must be 2D")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    if permute:
        # Apply permutation from (L, *, *) -> (*, *, L) for selected tensors
        x = torch.permute(
            x,
            dims=(1, 0),
        )

    return x


def preprocess_tensor(
    x: torch.Tensor,
    permute: bool,
    transpose: bool = False,
) -> torch.Tensor:
    if x.dim() == 2:
        x = torch.unsqueeze(x, dim=0)
    if x.dim() != 3:
        raise ValueError("Input must be 3D")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    if transpose:
        # e.g., (K, N) -> (N, K) or (L, K, N) -> (L, N, K)
        x = x.mT

    if permute:
        # Apply permutation from (L, *, *) -> (*, *, L) for selected tensors
        x = torch.permute(
            x,
            dims=(1, 2, 0),
        )

    return x


def gemm_epilogue(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    epi_cls: Callable[[object], EpilogueVisitorTree],
    epi_args: EpilogueVisitorTree.EpilogueArguments,
    epi_keys: tuple,
    pingpong: bool,
    tile_shape_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    add_to_output: bool,
) -> None:

    M, N, L = C.shape
    _, K, _ = A.shape
    assert A.shape == (M, K, L)
    assert B.shape == (N, K, L)
    assert C.shape == (M, N, L)
    assert A.dtype == B.dtype

    compile_key = (
        M,
        N,
        K,
        L,
        A.shape,
        B.shape,
        C.shape,
        A.stride(),
        B.stride(),
        C.stride(),
        A.dtype,
        B.dtype,
        C.dtype,
        ACC_DTYPE,
        tile_shape_mn,
        cluster_shape_mn,
        USE_QUACK,
        pingpong,
        add_to_output,
        *epi_keys,
    )

    if IS_PERSISTENT:
        max_active_clusters = get_max_active_clusters(
            cluster_shape_mn[0] *
            cluster_shape_mn[1]
        )
    else:
        max_active_clusters = 0

    if IS_PERSISTENT and DYNAMIC_SCHEDULER:
        tile_count_semaphore = torch.zeros(
            1,
            dtype=torch.int32,
            device=A.device,
        )
    else:
        tile_count_semaphore = None

    if PERMUTE_BATCH:
        raise NotImplementedError
    else:
        batch_idx_permute = None

    if compile_key not in gemm_epilogue.compile_cache:
        warnings.warn(
            f"Kernel compilation cache miss for compile_key={compile_key}. "
            f"This will trigger JIT compilation and may be slow.",
        )
        if USE_QUACK:
            _gemm = gemm_quack.HopperWgmmaGemmKernelQuack(
                acc_dtype=ACC_DTYPE,
                tile_shape_mn=tile_shape_mn,
                cluster_shape_mnk=(*cluster_shape_mn, 1),
                epilogue_visitor_tree_cls=epi_cls,
                pingpong=pingpong,
                is_persistent=IS_PERSISTENT,
                add_to_output=add_to_output,
            )
        else:
            raise NotImplementedError

        A_cute = to_cute_tensor(A)
        B_cute = to_cute_tensor(B)
        C_cute = to_cute_tensor(C)
        epi_args_cute = pytree.tree_map_only(
            torch.Tensor,
            func=to_cute_tensor_or_vector,
            tree=epi_args,
        )

        if not is_valid_dtypes(
            a_dtype=get_dtype(A_cute),
            b_dtype=get_dtype(B_cute),
            acc_dtype=ACC_DTYPE,
            c_dtype=get_dtype(C_cute),
            a_major=get_major(A_cute, dims=("m", "k", "l")),
            b_major=get_major(B_cute, dims=("n", "k", "l")),
        ):
            raise TypeError

        if not is_valid_tensor_alignment(
            m=M,
            n=N,
            k=K,
            l=L,
            ab_dtype=get_dtype(A_cute),
            c_dtype=get_dtype(C_cute),
            a_major=get_major(A_cute, dims=("m", "k", "l")),
            b_major=get_major(B_cute, dims=("n", "k", "l")),
            c_major=get_major(C_cute, dims=("m", "n", "l")),
        ):
            raise TypeError

        fake_scheduler_args = make_fake_scheduler_args(
            has_semaphore=(tile_count_semaphore is not None),
            has_batch_idx_permute=(batch_idx_permute is not None),
            l_sym=1,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm_epilogue.compile_cache[compile_key] = cute.compile(
            _gemm,
            A_cute,
            B_cute,
            C_cute,
            epi_args_cute,
            fake_scheduler_args,
            stream,
            options="--enable-tvm-ffi",
        )

    scheduler_args = make_scheduler_args(
        max_active_clusters=max_active_clusters,
        max_swizzle_size=8,
        tile_count_semaphore=tile_count_semaphore,
        batch_idx_permute=batch_idx_permute,
    )
    gemm_epilogue.compile_cache[compile_key](
        A,
        B,
        C,
        epi_args,
        scheduler_args,
    )


gemm_epilogue.compile_cache = {}


class GemmConfig(NamedTuple):
    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int
    pingpong: bool


GemmConfigOptions = [
    AutotuneConfig(
        config=GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            pingpong=pingpong,
        ),
    )
    for tile_m in [64, 128, 192, 256]
    for tile_n in [64, 128, 192, 256]
    for cluster_m in [1, 2]
    for cluster_n in [1, 2]
    for pingpong in [True, False]
]


GemmConfigOptions2 = [
    gemm_config
    for gemm_config in GemmConfigOptions
    if not all([
        gemm_config.kwargs["config"].tile_m == 192,
        gemm_config.kwargs["config"].tile_n >= 192,
        gemm_config.kwargs["config"].pingpong is False,
    ])
]


def _gemm_epilogue_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    epi_cls: Callable[[object], EpilogueVisitorTree],
    epi_args: EpilogueVisitorTree.EpilogueArguments,
    epi_keys: tuple,
    add_to_output: bool,
    config: GemmConfig,
) -> None:
    return gemm_epilogue(
        A=A,
        B=B,
        C=C,
        epi_cls=epi_cls,
        epi_args=epi_args,
        epi_keys=epi_keys,
        pingpong=config.pingpong,
        tile_shape_mn=(config.tile_m, config.tile_n),
        cluster_shape_mn=(config.cluster_m, config.cluster_n),
        add_to_output=add_to_output,
    )


gemm_epilogue_tuned = (
    autotune(
        configs=GemmConfigOptions2,
        key=["epi_keys"],
        cache_results=False,
    )(_gemm_epilogue_tuned)
)


gemm_epilogue_tuned_restricted = (
    autotune(
        configs=GemmConfigOptions2,
        key=["epi_keys"],
        cache_results=False,
    )(_gemm_epilogue_tuned)
)


gemm_epilogue_tuned_dict_m = {
    tile_m: autotune(
        configs=[
            gemm_config
            for gemm_config in GemmConfigOptions
            if gemm_config.kwargs["config"].tile_m == tile_m
        ],
        key=["epi_keys"],
        cache_results=False,
    )(_gemm_epilogue_tuned)
    for tile_m in [64, 128, 192, 256]
}


gemm_epilogue_tuned_restricted_dict_m = {
    tile_m: autotune(
        configs=[
            gemm_config
            for gemm_config in GemmConfigOptions2
            if gemm_config.kwargs["config"].tile_m == tile_m
        ],
        key=["epi_keys"],
        cache_results=False,
    )(_gemm_epilogue_tuned)
    for tile_m in [64, 128, 192, 256]
}


gemm_epilogue_tuned_dict_n = {
    tile_n: autotune(
        configs=[
            gemm_config
            for gemm_config in GemmConfigOptions
            if gemm_config.kwargs["config"].tile_n == tile_n
        ],
        key=["epi_keys"],
        cache_results=False,
    )(_gemm_epilogue_tuned)
    for tile_n in [64, 128, 192, 256]
}


gemm_epilogue_tuned_restricted_dict_n = {
    tile_n: autotune(
        configs=[
            gemm_config
            for gemm_config in GemmConfigOptions2
            if gemm_config.kwargs["config"].tile_n == tile_n
        ],
        key=["epi_keys"],
        cache_results=False,
    )(_gemm_epilogue_tuned)
    for tile_n in [64, 128, 192, 256]
}
