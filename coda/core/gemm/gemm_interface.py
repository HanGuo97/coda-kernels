import torch
import functools
import cutlass
import cutlass.cute as cute
from typing import Callable

import quack.cache
from quack.cache import jit_cache
from quack.rounding import RoundingMode
from quack.gemm_interface import (
    default_config,
    prune_invalid_gemm_configs,
)
from quack.autotuner import (
    autotune,
    AutotuneConfig,
)
from quack.gemm_config import (
    GemmConfig,
    get_all_configs,
)
from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from quack.gemm_tvm_ffi_utils import (
    get_major,
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from coda.core.ops.torch_utils import preprocess_tensor
from coda.core.epilogue.utils import compile_epi_args, process_epi_args


@jit_cache
def _compile_gemm(
    a_dtype: type[cute.Numeric],
    b_dtype: type[cute.Numeric],
    d_dtype: type[cute.Numeric],
    c_dtype: type[cute.Numeric],
    a_major: str,
    b_major: str,
    d_major: str,
    c_major: str,
    tile_shape_mnk: tuple[int, ...],
    cluster_shape_mnk: tuple[int, ...],
    pingpong: bool,
    persistent: bool,
    is_dynamic_persistent: bool,
    add_to_output: bool,
    concat_layout: tuple | None,
    varlen_m: bool,
    varlen_k: bool,
    gather_A: bool,
    use_tma_gather: bool,
    has_batch_idx_permute: bool,
    device_capacity: tuple[int, int],
    rounding_mode: RoundingMode,
    sr_seed_mode: int | None,
    num_warps: int | None,
    GemmCls: type,
    epi_keys: tuple,
) -> Callable:
    assert use_tma_gather is False
    assert sr_seed_mode is None
    assert num_warps is None
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        varlen_k=varlen_k,
        gather_A=gather_A,
    )

    epi_args = compile_epi_args(
        GemmCls=GemmCls,
        epi_keys=epi_keys,
        add_to_output=add_to_output,
        rounding_mode=rounding_mode,
        sr_seed=None,
        m=m,
        n=n,
        k=k,
        l=l,
    )
    scheduler_args = make_fake_scheduler_args(
        has_semaphore=(is_dynamic_persistent and device_capacity[0] <= 9),
        has_batch_idx_permute=has_batch_idx_permute,
        l_sym=l,
    )
    aidx_len = m if varlen_m else (k if varlen_k else None)
    varlen_args = make_fake_varlen_args(
        varlen_m=varlen_m,
        varlen_k=varlen_k,
        gather_A=gather_A,
        aidx_len=aidx_len,
    )
    if device_capacity[0] == 9:
        extra_kwargs = {"pingpong": pingpong, "is_persistent": persistent}
    else:
        raise NotImplementedError

    _gemm = GemmCls(
        acc_dtype=cute.Float32,
        a_dtype=a_dtype,
        tile_shape_mnk=tile_shape_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        gather_A=gather_A,
        concat_layout=concat_layout,
        **extra_kwargs,
    )

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _gemm,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        stream,
        options="--enable-tvm-ffi",
    )


def _gemm_epilogue(
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
    epi_args: dict,
    epi_keys: tuple,
) -> None:

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    if is_dynamic_persistent and device_capacity[0] <= 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler for SM8x and SM90 requires a semaphore in GMEM"
        )
    if device_capacity[0] == 8:
        if add_to_output:
            C = D
            add_to_output = False

    M, K, L = A.shape
    N, _, _ = B.shape
    assert A.shape == (M, K, L)
    assert B.shape == (N, K, L)
    assert D is None or D.shape == (M, N, L)
    assert C is None or C.shape == (M, N, L)

    compiled_fn = _compile_gemm(
        a_dtype=torch2cute_dtype_map[A.dtype],
        b_dtype=torch2cute_dtype_map[B.dtype],
        d_dtype=torch2cute_dtype_map[D.dtype] if D is not None else None,
        c_dtype=torch2cute_dtype_map[C.dtype] if C is not None else None,
        a_major=get_major(A, "m", "k"),
        b_major=get_major(B, "n", "k"),
        d_major=get_major(D, "m", "n") if D is not None else None,
        c_major=get_major(C, "m", "n") if C is not None else None,
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
        rounding_mode=RoundingMode.RN,
        sr_seed_mode=None,
        num_warps=None,
        GemmCls=GemmCls,
        epi_keys=epi_keys,
    )

    cluster_size = cluster_M * cluster_N * cluster_K
    max_active_clusters = (
        get_max_active_clusters(
            cluster_size=cluster_size,
            device_capacity=device_capacity,
        )
        if persistent else 0
    )

    processed_epi_args = process_epi_args(
        GemmCls=GemmCls,
        epi_args=epi_args,
        add_to_output=None,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters=max_active_clusters,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=tile_count_semaphore if (is_dynamic_persistent and device_capacity[0] <= 9) else None,
        batch_idx_permute=batch_idx_permute,
    )
    varlen_args = make_varlen_args(
        cu_seqlens_m=None,
        cu_seqlens_k=None,
        A_idx=None,
    )

    if device_capacity[0] in [10, 11]:
        compiled_fn(A, B, D, C, processed_epi_args, scheduler_args, varlen_args, None, None)
    else:
        compiled_fn(A, B, D, C, processed_epi_args, scheduler_args, varlen_args)


def prune_gemm_configs(configs: list[AutotuneConfig], named_args: dict, **kwargs) -> list[AutotuneConfig]:
    kwargs = named_args | kwargs
    pin_tile_M = kwargs.get("pin_tile_M", None)
    pin_tile_N = kwargs.get("pin_tile_N", None)
    configs = prune_invalid_gemm_configs(
        configs=configs,
        named_args=named_args,
        **kwargs,
    )
    configs = [conf for conf in configs if not conf.kwargs["config"].swap_ab]
    if pin_tile_M is not None:
        configs = [conf for conf in configs if conf.kwargs["config"].tile_m == pin_tile_M]
    if pin_tile_N is not None:
        configs = [conf for conf in configs if conf.kwargs["config"].tile_n == pin_tile_N]
    return configs


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["GemmCls", "epi_keys", "pin_tile_M", "pin_tile_N", "add_to_output"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_epilogue_tuned(
    GemmCls: type,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor | None,
    epi_args: dict,
    epi_keys: tuple,
    pin_tile_M: int | None,
    pin_tile_N: int | None,
    batch_idx_permute: torch.Tensor | None,
    add_to_output: bool,
    config: GemmConfig | None,
) -> None:
    if config is None:
        config = default_config(A.device)

    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device)
        if config.is_dynamic_persistent and get_device_capacity(A.device)[0] == 9
        else None
    )

    if config.swap_ab:
        raise NotImplementedError

    _gemm_epilogue(
        GemmCls=GemmCls,
        A=A,
        B=B,
        D=D,
        C=C,
        tile_count_semaphore=tile_count_semaphore,
        tile_M=config.tile_m,
        tile_N=config.tile_n,
        tile_K=config.tile_k,
        cluster_M=config.cluster_m,
        cluster_N=config.cluster_n,
        cluster_K=config.cluster_k,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.is_dynamic_persistent,
        max_swizzle_size=config.max_swizzle_size,
        batch_idx_permute=batch_idx_permute,
        add_to_output=add_to_output,
        epi_args=epi_args,
        epi_keys=epi_keys,
    )


def _kernel_op(
    name: str,
    mutates_args: tuple[str, ...],
) -> Callable[[Callable], Callable]:

    def decorator(fn: Callable) -> Callable:

        @torch.library.custom_op(
            name,
            mutates_args=mutates_args,
            device_types="cuda",
        )
        @functools.wraps(fn)
        def op(*args, **kwargs) -> None:
            return fn(*args, **kwargs)

        @torch.library.register_fake(name)
        def _(*args, **kwargs) -> None:
            pass

        return op

    return decorator


def _dispatch(
    GemmCls: type,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor | None,
    C: torch.Tensor | None = None,
    *,
    epi_args: dict,
    epi_keys: tuple,
    pin_tile_M: int | None = None,
    pin_tile_N: int | None = None,
    batch_idx_permute: torch.Tensor | None = None,
    add_to_output: bool = False,
    tuned: bool = True,
) -> None:
    # Preprocess A: (M, K) -> (M, K, L) with permute
    A = preprocess_tensor(A, permute=True, transpose=False)
    # Preprocess B: (K, N) -> (N, K, L) with transpose + permute
    B = preprocess_tensor(B, permute=True, transpose=True)
    # Preprocess D: (M, N) -> (M, N, L) with permute
    D = preprocess_tensor(D, permute=True, transpose=False) if D is not None else None
    # Preprocess C: (M, N) -> (M, N, L) with permute
    C = preprocess_tensor(C, permute=True, transpose=False) if C is not None else None

    if tuned:
        return _gemm_epilogue_tuned(
            GemmCls=GemmCls,
            A=A,
            B=B,
            D=D,
            C=C,
            epi_args=epi_args,
            epi_keys=epi_keys,
            pin_tile_M=pin_tile_M,
            pin_tile_N=pin_tile_N,
            batch_idx_permute=batch_idx_permute,
            add_to_output=add_to_output,
        )
    else:
        return _gemm_epilogue_tuned.fn(
            GemmCls=GemmCls,
            A=A,
            B=B,
            D=D,
            C=C,
            epi_args=epi_args,
            epi_keys=epi_keys,
            pin_tile_M=pin_tile_M,
            pin_tile_N=pin_tile_N,
            batch_idx_permute=batch_idx_permute,
            add_to_output=add_to_output,
            config=None,
        )
