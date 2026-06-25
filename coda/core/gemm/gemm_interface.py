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
    cluster_K: int,
    tile_K: int | None,
    pingpong: bool,
    persistent: bool,
    is_dynamic_persistent: bool,
    max_swizzle_size: int,
    batch_idx_permute: torch.Tensor | None,
    add_to_output: bool,
    rounding_mode: RoundingMode,
) -> None:

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    if rounding_mode == RoundingMode.RS:
        raise NotImplementedError
    if is_dynamic_persistent and device_capacity[0] <= 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler for SM8x and SM90 requires a semaphore in GMEM"
        )
    if device_capacity[0] == 8:
        if add_to_output:
            C = D
            add_to_output = False

    A_p, B_p, D_p, C_p = perm3d(
        A=A,
        B=B,
        D=D,
        C=C,
        varlen_m=False,
        varlen_k=False,
    )

    compiled_fn = _compile_gemm(
        a_dtype=torch2cute_dtype_map[A.dtype],
        b_dtype=torch2cute_dtype_map[B.dtype],
        d_dtype=torch2cute_dtype_map[D.dtype] if D is not None else None,
        c_dtype=torch2cute_dtype_map[C.dtype] if C is not None else None,
        a_major=get_major(A_p, "m", "k"),
        b_major=get_major(B_p, "n", "k"),
        d_major=get_major(D_p, "m", "n") if D_p is not None else None,
        c_major=get_major(C_p, "m", "n") if C_p is not None else None,
        tile_shape_mnk=(tile_M, tile_N) if tile_K is None else (tile_M, tile_N, tile_K),
        cluster_shape_mnk=(cluster_M, cluster_N, cluster_K),
        pingpong=pingpong,
        persistent=persistent,
        is_dynamic_persistent=is_dynamic_persistent,
        add_to_output=add_to_output,
        concat_layout=None,
        varlen_m=False,
        varlen_k=False,
        gather_A=False,
        use_tma_gather=False,
        has_batch_idx_permute=batch_idx_permute is not None,
        device_capacity=device_capacity,
        rounding_mode=rounding_mode,
        sr_seed_mode=None,
        num_warps=None,
        gemm_cls_name=GemmCls.__name__,
    )

    cluster_size = cluster_M * cluster_N * cluster_K
    max_active_clusters = (
        get_max_active_clusters(
            cluster_size=cluster_size,
            device_capacity=device_capacity,
        )
        if persistent else 0
    )

    epi_args = GemmCls.EpilogueArguments(
        add_to_output=None,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters=max_active_clusters,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=tile_count_semaphore,
        batch_idx_permute=batch_idx_permute,
    )
    varlen_args = make_varlen_args(
        cu_seqlens_m=None,
        cu_seqlens_k=None,
        A_idx=None,
    )

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args)
