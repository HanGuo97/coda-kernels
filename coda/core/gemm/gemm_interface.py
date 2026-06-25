import torch
import cutlass
import cutlass.cute as cute

from quack.cache import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, torch2cute_dtype_map
from quack.gemm_sm90 import GemmSm90

from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    get_dtypes,
    get_majors,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
    perm3d,
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
    GemmCls: type,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor | None,
    C: torch.Tensor | None,
    tile_count_semaphore: torch.Tensor | None,
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None,
    pingpong: bool,
    persistent: bool,
    is_dynamic_persistent: bool,
    max_swizzle_size: int,
) -> None:

    A_p = perm3d_single(A, varlen_m=False)
    B_p = perm3d_single(B, varlen_m=False)
    D_p = perm3d_single(D, varlen_m=False)
    C_p = perm3d_single(C, varlen_m=False)

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    if rounding_mode == RoundingMode.RS:
        assert device_capacity[0] == 10, "Stochastic rounding (RoundingMode.RS) requires SM100"

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    compiled_fn = _compile(
        a_dtype=torch2cute_dtype_map[A.dtype],
        b_dtype=torch2cute_dtype_map[B.dtype],
        d_dtype=torch2cute_dtype_map[D.dtype] if D is not None else None,
        c_dtype=torch2cute_dtype_map[C.dtype] if C is not None else None,
        a_major=get_major(A_p, "m", "k"),
        b_major=get_major(B_p, "n", "k"),
        d_major=get_major(D_p, "m", "n") if D_p is not None else None,
        c_major=get_major(C_p, "m", "n") if C_p is not None else None,
        tile_M=tile_M,
        tile_N=tile_N,
        tile_K=tile_K,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
        pingpong=pingpong,
        persistent=persistent,
        is_dynamic_persistent=is_dynamic_persistent,
        device_capacity=device_capacity,
        gemm_cls_name=GemmCls.__name__,
    )

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0

    epi_args = GemmCls.EpilogueArguments(
        add_to_output=None,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, None, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, None)
