import torch
import triton
import functools
from types import ModuleType
from typing import Callable
from rapier.gemm.gemm_interface import (
    DEFAULT_PINGPONG,
    DEFAULT_TILE_M,
    DEFAULT_TILE_N,
    DEFAULT_CLUSTER_M,
    DEFAULT_CLUSTER_N,
    gemm_epilogue,
    gemm_epilogue_tuned,
    gemm_epilogue_tuned_dict_m,
    gemm_epilogue_tuned_dict_n,
    gemm_epilogue_tuned_restricted,
    gemm_epilogue_tuned_restricted_dict_m,
    gemm_epilogue_tuned_restricted_dict_n,
    preprocess_vector,
    preprocess_tensor,
)
from kernels.gens.epilogue import (
    kernel_0,
    kernel_1,
    kernel_2,
    kernel_3,
    kernel_4,
    kernel_5,
    kernel_6,
    kernel_7,
    kernel_8,
    kernel_9,
)

USE_TUNED = False


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
    kernel: ModuleType,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    add_to_output: bool,
    block_size_m: int | None,
    block_size_n: int | None,
    restricted: bool = False,
    **epi_kwargs,
) -> None:

    M, K = A.shape
    _, N = B.shape
    epi_cls, epi_args, epi_outs, epi_keys = kernel.prepare_epilogue(
        shape_mnkl=(M, N, K, 1),
        tile_shape_mn=(block_size_m, block_size_n),
        **epi_kwargs,
    )

    # Preprocess A: (M, K) -> (M, K, L) with permute
    A = preprocess_tensor(A, permute=True, transpose=False)
    # Preprocess B: (K, N) -> (N, K, L) with transpose + permute
    B = preprocess_tensor(B, permute=True, transpose=True)
    # Preprocess out: (M, N) -> (M, N, L) with permute
    out = preprocess_tensor(out, permute=True, transpose=False)

    if not USE_TUNED:
        if block_size_m is None:
            block_size_m = DEFAULT_TILE_M
        if block_size_n is None:
            block_size_n = DEFAULT_TILE_N

        gemm_epilogue(
            A=A,
            B=B,
            C=out,
            epi_cls=epi_cls,
            epi_args=epi_args,
            epi_keys=epi_keys,
            pingpong=DEFAULT_PINGPONG,
            tile_shape_mn=(block_size_m, block_size_n),
            cluster_shape_mn=(DEFAULT_CLUSTER_M, DEFAULT_CLUSTER_N),
            add_to_output=add_to_output,
        )
    else:
        if (block_size_m is None) and (block_size_n is None):
            if not restricted:
                gemm_epilogue_tuned_fn = gemm_epilogue_tuned
            else:
                gemm_epilogue_tuned_fn = gemm_epilogue_tuned_restricted
        elif (block_size_m is not None) and (block_size_n is None):
            if not restricted:
                gemm_epilogue_tuned_fn = gemm_epilogue_tuned_dict_m[block_size_m]
            else:
                gemm_epilogue_tuned_fn = gemm_epilogue_tuned_restricted_dict_m[block_size_m]
        elif (block_size_m is None) and (block_size_n is not None):
            if not restricted:
                gemm_epilogue_tuned_fn = gemm_epilogue_tuned_dict_n[block_size_n]
            else:
                gemm_epilogue_tuned_fn = gemm_epilogue_tuned_restricted_dict_n[block_size_n]
        else:
            raise NotImplementedError

        gemm_epilogue_tuned_fn(
            A=A,
            B=B,
            C=out,
            epi_cls=epi_cls,
            epi_args=epi_args,
            epi_keys=epi_keys,
            add_to_output=add_to_output,
        )


@_kernel_op("rapier::_k0_dispatch", mutates_args=("D", "S", "O"))
def _k0_dispatch(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, C: torch.Tensor, S: torch.Tensor, W: torch.Tensor, O: torch.Tensor, block_size: int) -> None:
    _dispatch(kernel=kernel_0, A=A, B=B, out=D, add_to_output=False, C=C, S=S, W=W, O=O, block_size_m=None, block_size_n=block_size, restricted=True)


@_kernel_op("rapier::_k1_dispatch", mutates_args=("D",))
def _k1_dispatch(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, R: torch.Tensor) -> None:
    _dispatch(kernel=kernel_1, A=A, B=B, out=D, add_to_output=False, R=R, block_size_m=None, block_size_n=None)


@_kernel_op("rapier::_k2_dispatch", mutates_args=("D", "O"))
def _k2_dispatch(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, O: torch.Tensor, R: torch.Tensor) -> None:
    _dispatch(kernel=kernel_2, A=A, B=B, out=D, add_to_output=False, O=O, R=R, block_size_m=None, block_size_n=None)


@_kernel_op("rapier::_k3_dispatch", mutates_args=("D", "O"))
def _k3_dispatch(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, O: torch.Tensor, cos_sin: torch.Tensor) -> None:
    _dispatch(kernel=kernel_3, A=A, B=B, out=D, add_to_output=False, O=O, cos_sin=cos_sin, block_size_m=None, block_size_n=None)


@_kernel_op("rapier::_k4_dispatch", mutates_args=("D", "O"))
def _k4_dispatch(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, O: torch.Tensor, R: torch.Tensor, cos_sin: torch.Tensor) -> None:
    _dispatch(kernel=kernel_4, A=A, B=B, out=D, add_to_output=False, O=O, R=R, cos_sin=cos_sin, block_size_m=None, block_size_n=None)


@_kernel_op("rapier::_k5_dispatch", mutates_args=("logits", "logits_tgt", "logits_lse"))
def _k5_dispatch(A: torch.Tensor, B: torch.Tensor, logits: torch.Tensor, R: torch.Tensor, targets: torch.Tensor, logits_tgt: torch.Tensor, logits_lse: torch.Tensor, block_size: int) -> None:
    _dispatch(kernel=kernel_5, A=A, B=B, out=logits, add_to_output=False, R=R, targets=targets, logits_tgt=logits_tgt, logits_lse=logits_lse, block_size_m=None, block_size_n=block_size, restricted=True)


@_kernel_op("rapier::_k6_dispatch", mutates_args=("O", "C_out", "dW"))
def _k6_dispatch(A: torch.Tensor, B_T: torch.Tensor, O: torch.Tensor, C: torch.Tensor, W: torch.Tensor, R: torch.Tensor, ZdZ: torch.Tensor, C_out: torch.Tensor, dW: torch.Tensor, block_size: int) -> None:
    _dispatch(kernel=kernel_6, A=A, B=B_T, out=O, add_to_output=True, C=C, W=W, R=R, ZdZ=ZdZ, C_out=C_out, dW=dW, block_size_m=block_size, block_size_n=None, restricted=True)


@_kernel_op("rapier::_k7_dispatch", mutates_args=("O", "dZ", "ZdZ"))
def _k7_dispatch(A: torch.Tensor, B_T: torch.Tensor, O: torch.Tensor, Z: torch.Tensor, dZ: torch.Tensor, ZdZ: torch.Tensor, block_size: int) -> None:
    _dispatch(kernel=kernel_7, A=A, B=B_T, out=O, add_to_output=False, Z=Z, dZ=dZ, ZdZ=ZdZ, block_size_m=None, block_size_n=block_size, restricted=True)


@_kernel_op("rapier::_k8_dispatch", mutates_args=("D", "O"))
def _k8_dispatch(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, O: torch.Tensor) -> None:
    _dispatch(kernel=kernel_8, A=A, B=B, out=D, add_to_output=False, O=O, block_size_m=None, block_size_n=None)


@_kernel_op("rapier::_k9_dispatch", mutates_args=("logits", "logits_tgt", "logits_lse"))
def _k9_dispatch(A: torch.Tensor, B: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, logits_tgt: torch.Tensor, logits_lse: torch.Tensor, block_size: int) -> None:
    _dispatch(kernel=kernel_9, A=A, B=B, out=logits, add_to_output=False, targets=targets, logits_tgt=logits_tgt, logits_lse=logits_lse, block_size_m=None, block_size_n=block_size)


def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GEMM with residual, partial RMSNorm reduction, and fused RMSNorm-weight scaling.

    Computes:
        1. GEMM with residual: D = A @ B + C
        2. Partial RMSNorm reduction: S = mean(D ** 2, blocks of size block_size)
        3. RMSNorm-weight scaling: O = D * W (broadcast across M)

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N) where N = num_blocks * block_size
        C: Residual matrix of shape (M, N)
        W: RMSNorm weight of shape (N,) broadcast along M
        block_size: Block size for partial reduction

    Returns:
        Tuple of (D, S, O) where:
            D: GEMM output with residual of shape (M, N) with same dtype as inputs
            S: Partial mean of squares of shape (M, cdiv(N, block_size)) in fp32
            O: Weight-scaled output of shape (M, N) with same dtype as inputs
    """

    M, _ = A.shape
    _, N = B.shape
    assert N % block_size == 0
    num_blocks = triton.cdiv(N, block_size)

    D = torch.empty((M, N), dtype=A.dtype, device=A.device)
    S = torch.empty((M, num_blocks), dtype=torch.float32, device=A.device)
    O = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Preprocess C: (M, N) -> (M, N, L) with permute
    C = preprocess_tensor(C, permute=True)
    # Preprocess S: (M, num_blocks) -> (L, M, num_blocks) without permute
    S = preprocess_tensor(S, permute=False)
    # Preprocess W: (N,) -> (L, N)
    W = preprocess_vector(W, permute=False)
    # Preprocess O: (M, N) -> (M, N, L) with permute (matches the main D path)
    O = preprocess_tensor(O, permute=True)

    _k0_dispatch(A=A, B=B, D=D, C=C, S=S, W=W, O=O, block_size=block_size)

    S = S.squeeze(dim=0)
    O = O.squeeze(dim=-1)
    return D, S, O


def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    GEMM with RMSNorm normalization.

    Computes:
        D = A @ B
        D = D * R

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N)
        R: RMSNorm reciprocal standard deviation of shape (M,) in fp32

    Returns:
        D: Normalized output matrix of shape (M, N) with same dtype as inputs
    """

    M, _ = A.shape
    _, N = B.shape

    D = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Preprocess R: (M,) -> (L, M)
    R = preprocess_vector(R, permute=False)

    _k1_dispatch(A=A, B=B, D=D, R=R)

    return D


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GEMM with SwiGLU activation.

    Computes:
        D = A @ B
        G, U = interleaved_split(D) where D = [g0, u0, g1, u1, ...]
        O = silu(G) * U

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N) where N must be even for the split

    Returns:
        Tuple of (D, O) where:
            D: GEMM output before activation of shape (M, N) with same dtype as inputs
            O: SwiGLU activated output of shape (M, cdiv(N, 2)) with same dtype as inputs
    """

    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0

    D = torch.empty((M, N), dtype=A.dtype, device=A.device)
    O = torch.empty((M, triton.cdiv(N, 2)), dtype=A.dtype, device=A.device)

    # Preprocess O: (M, cdiv(N, 2)) -> (M, cdiv(N, 2), L) with permute
    O = preprocess_tensor(O, permute=True)

    _k8_dispatch(A=A, B=B, D=D, O=O)

    O = O.squeeze(dim=-1)
    return D, O


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GEMM with RMSNorm and SwiGLU activation.

    Computes:
        D = A @ B
        D = D * R
        G, U = interleaved_split(D) where D = [g0, u0, g1, u1, ...]
        O = silu(G) * U

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N) where N must be even for the split
        R: RMSNorm reciprocal standard deviation of shape (M,) in fp32

    Returns:
        Tuple of (D, O) where:
            D: Normalized output before activation of shape (M, N) with same dtype as inputs
            O: SwiGLU activated output of shape (M, cdiv(N, 2)) with same dtype as inputs
    """

    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0

    D = torch.empty((M, N), dtype=A.dtype, device=A.device)
    O = torch.empty((M, triton.cdiv(N, 2)), dtype=A.dtype, device=A.device)

    # Preprocess O: (M, cdiv(N, 2)) -> (M, cdiv(N, 2), L) with permute
    O = preprocess_tensor(O, permute=True)
    # Preprocess R: (M,) -> (L, M)
    R = preprocess_vector(R, permute=False)

    _k2_dispatch(A=A, B=B, D=D, O=O, R=R)

    O = O.squeeze(dim=-1)
    return D, O


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GEMM with RoPE positional encoding.

    Computes:
        1. GEMM: D = A @ B
        2. RoPE rotation: [x', y'] = [x*cos + y*sin, -x*sin + y*cos] on pairs of values

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N) where N must be even
        cos_sin: Rotation tensor of shape (M, N) with interleaved [cos, sin] pairs

    Returns:
        Tuple of (D, O) where:
            D: GEMM output before RoPE of shape (M, N) with same dtype as inputs
            O: Output with RoPE applied of shape (M, N) with same dtype as inputs
    """

    M, _ = A.shape
    _, N = B.shape

    D = torch.empty((M, N), dtype=A.dtype, device=A.device)
    O = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Preprocess cos_sin: (M, N) -> (M, N, L) with permute
    cos_sin = preprocess_tensor(cos_sin, permute=True)
    # Preprocess O: (M, N) -> (M, N, L) with permute
    O = preprocess_tensor(O, permute=True)

    _k3_dispatch(A=A, B=B, D=D, O=O, cos_sin=cos_sin)

    O = O.squeeze(dim=-1)
    return D, O


def gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GEMM with RMSNorm and RoPE positional encoding.

    Computes:
        D = A @ B
        D = D * R
        O = RoPE(D, cos_sin) where [x', y'] = [x*cos + y*sin, -x*sin + y*cos]

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N) where N must be even
        R: RMSNorm reciprocal standard deviation of shape (M,) in fp32
        cos_sin: Rotation tensor of shape (M, N) with interleaved [cos, sin] pairs

    Returns:
        Tuple of (D, O) where:
            D: Normalized output before RoPE of shape (M, N) with same dtype as inputs
            O: Output with RoPE applied of shape (M, N) with same dtype as inputs
    """

    M, _ = A.shape
    _, N = B.shape

    D = torch.empty((M, N), dtype=A.dtype, device=A.device)
    O = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Preprocess R: (M,) -> (L, M)
    R = preprocess_vector(R, permute=False)
    # Preprocess cos_sin: (M, N) -> (M, N, L) with permute
    cos_sin = preprocess_tensor(cos_sin, permute=True)
    # Preprocess O: (M, N) -> (M, N, L) with permute
    O = preprocess_tensor(O, permute=True)

    _k4_dispatch(A=A, B=B, D=D, O=O, R=R, cos_sin=cos_sin)

    O = O.squeeze(dim=-1)
    return D, O


def gemm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GEMM with target logit selection and partial LSE.

    Computes:
        1. GEMM: logits = A @ B
        2. Target logit selection: logits_tgt = logits[arange(M), targets]
        3. Partial LSE reduction: logits_lse = logsumexp(logits, blocks of size block_size)

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N) where N = num_blocks * block_size
        targets: Target indices of shape (M,)
        block_size: Block size for partial reductions

    Returns:
        Tuple of (logits, logits_tgt, logits_lse) where:
            logits: Full logits matrix of shape (M, N) with same dtype as inputs
            logits_tgt: Target logits of shape (M,) with same dtype as inputs
            logits_lse: Partial LSE values of shape (M, cdiv(N, block_size)) in fp32
    """

    M, _ = A.shape
    _, N = B.shape
    assert N % block_size == 0
    num_blocks = triton.cdiv(N, block_size)

    logits = torch.empty((M, N), dtype=A.dtype, device=A.device)
    logits_tgt = torch.empty((M,), dtype=A.dtype, device=A.device)
    logits_lse = torch.empty((M, num_blocks), dtype=torch.float32, device=A.device)

    # Preprocess targets: (M,) -> (L, M)
    targets = preprocess_vector(targets, permute=False)
    # Preprocess logits_tgt: (M,) -> (L, M)
    logits_tgt = preprocess_vector(logits_tgt, permute=False)
    # Preprocess logits_lse: (M, num_blocks) -> (L, M, num_blocks) without permute
    logits_lse = preprocess_tensor(logits_lse, permute=False)

    _k9_dispatch(A=A, B=B, logits=logits, targets=targets, logits_tgt=logits_tgt, logits_lse=logits_lse, block_size=block_size)

    logits_tgt = logits_tgt.squeeze(dim=0)
    logits_lse = logits_lse.squeeze(dim=0)
    return logits, logits_tgt, logits_lse


def gemm_rmsnorm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GEMM with RMSNorm and partial LSE.

    Computes:
        1. GEMM: logits = A @ B
        2. RMSNorm scaling: logits = logits * R
        3. Target logit selection: logits_tgt = logits[arange(M), targets]
        4. Partial LSE reduction: logits_lse = logsumexp(logits, blocks of size block_size)

    Args:
        A: Input matrix of shape (M, K)
        B: Weight matrix of shape (K, N) where N = num_blocks * block_size
        R: RMSNorm reciprocal standard deviation of shape (M,) in fp32
        targets: Target indices of shape (M,)
        block_size: Block size for partial reductions

    Returns:
        Tuple of (logits, logits_tgt, logits_lse) where:
            logits: Full logits matrix of shape (M, N) with same dtype as inputs
            logits_tgt: Target logits of shape (M,) with same dtype as inputs
            logits_lse: Partial LSE values of shape (M, cdiv(N, block_size)) in fp32
    """

    M, _ = A.shape
    _, N = B.shape
    assert N % block_size == 0
    num_blocks = triton.cdiv(N, block_size)

    logits = torch.empty((M, N), dtype=A.dtype, device=A.device)
    logits_tgt = torch.empty((M,), dtype=A.dtype, device=A.device)
    logits_lse = torch.empty((M, num_blocks), dtype=torch.float32, device=A.device)

    # Preprocess R: (M,) -> (L, M)
    R = preprocess_vector(R, permute=False)
    # Preprocess targets: (M,) -> (L, M)
    targets = preprocess_vector(targets, permute=False)
    # Preprocess logits_tgt: (M,) -> (L, M)
    logits_tgt = preprocess_vector(logits_tgt, permute=False)
    # Preprocess logits_lse: (M, num_blocks) -> (L, M, num_blocks) without permute
    logits_lse = preprocess_tensor(logits_lse, permute=False)

    _k5_dispatch(A=A, B=B, logits=logits, R=R, targets=targets, logits_tgt=logits_tgt, logits_lse=logits_lse, block_size=block_size)

    logits_tgt = logits_tgt.squeeze(dim=0)
    logits_lse = logits_lse.squeeze(dim=0)
    return logits, logits_tgt, logits_lse


def gemm_residual_partial_rmsnorm_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    ZdZ: torch.Tensor,
    O: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for GEMM with residual, partial RMSNorm reduction, and fused
    RMSNorm-weight scaling.

    Computes:
        D = A @ B.T
        C_norm = C * R
        O = O + (D * W - C_norm * ZdZ) * R
        C_out = C_norm * W
        dW = sum(D * C_norm, blocks of size block_size)

    Args:
        A: Gradient from previous layer of shape (M, K)
        B: Weight matrix of shape (N, K)
        C: RMSNorm input matrix of shape (M, N)
        W: RMSNorm weight of shape (N,) broadcast along M
        R: RMSNorm reciprocal standard deviation of shape (M,) in fp32
        ZdZ: Per-row preactivation-gradient mean of shape (M,) in fp32
        O: Incoming gradient w.r.t. residual of shape (M, N)
        block_size: Block size for partial reduction

    Returns:
        Tuple of (O, C, dW) where:
            O: Updated gradient w.r.t. residual of shape (M, N) with same dtype as inputs
            C: RMSNorm output of shape (M, N) with same dtype as inputs
            dW: Partially-reduced RMSNorm-weight gradient of shape (N, cdiv(M, block_size)) in fp32
    """

    M, _ = A.shape
    N, _ = B.shape
    assert M % block_size == 0
    num_blocks = triton.cdiv(M, block_size)

    # Transpose B for GEMM (we want A @ B.T)
    B_T = B.mT

    C_out = torch.empty((M, N), dtype=A.dtype, device=A.device)
    dW = torch.empty((N, num_blocks), dtype=torch.float32, device=A.device)

    # Preprocess C (residual matrix): (M, N) -> (M, N, L) with permute
    C = preprocess_tensor(C, permute=True)
    # Preprocess W: (N,) -> (L, N)
    W = preprocess_vector(W, permute=False)
    # Preprocess R: (M,) -> (L, M)
    R = preprocess_vector(R, permute=False)
    # Preprocess ZdZ: (M,) -> (L, M)
    ZdZ = preprocess_vector(ZdZ, permute=False)
    # Preprocess C_out: (M, N) -> (M, N, L) with permute
    C_out = preprocess_tensor(C_out, permute=True)
    # Preprocess dW: (N, num_blocks) -> (L, N, num_blocks) without permute
    dW = preprocess_tensor(dW, permute=False)

    _k6_dispatch(A=A, B_T=B_T, O=O, C=C, W=W, R=R, ZdZ=ZdZ, C_out=C_out, dW=dW, block_size=block_size)

    C_out = C_out.squeeze(dim=-1)
    dW = dW.squeeze(dim=0)
    return O, C_out, dW


def gemm_partial_swiglu_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for GEMM with SwiGLU activation.

    Computes:
        D = A @ B.T
        G, U = interleaved_split(Z) where Z = [g0, u0, g1, u1, ...]
        O = silu(G) * U
        dG = D * U * (sigmoid(G) + silu(G) * (1 - sigmoid(G)))
        dU = D * silu(G)
        dZ = interleaved_concat([dG, dU]) where dZ = [dg0, du0, dg1, du1, ...]
        ZdZ = mean(G * dG + U * dU, over bs blocks of size block_size)

    Args:
        A: Gradient from previous layer of shape (M, K)
        B: Weight matrix of shape (N, K)
        Z: Preactivation in interleaved format of shape (M, 2N)
        block_size: Block size for partial reduction

    Returns:
        Tuple of (dZ, ZdZ, O) where:
            dZ: Concatenated SwiGLU gradients of shape (M, 2N) with same dtype as inputs
            ZdZ: Partially-reduced preactivation-gradient products of shape (M, cdiv(N, block_size)) in fp32
            O: SwiGLU forward output silu(G) * U of shape (M, N) with same dtype as inputs
    """

    M, _ = A.shape
    N, _ = B.shape
    assert N % block_size == 0
    num_blocks = triton.cdiv(N, block_size)

    # Transpose B for GEMM (we want A @ B.T)
    B_T = B.mT

    dZ = torch.empty((M, N * 2), dtype=A.dtype, device=A.device)
    ZdZ = torch.empty((M, num_blocks), dtype=torch.float32, device=A.device)
    O = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Preprocess Z: (M, 2N) -> (M, 2N, L) with permute
    Z = preprocess_tensor(Z, permute=True)
    # Preprocess dZ: (M, 2N) -> (M, 2N, L) with permute
    dZ = preprocess_tensor(dZ, permute=True)
    # Preprocess ZdZ: (M, num_blocks) -> (L, M, num_blocks) without permute
    ZdZ = preprocess_tensor(ZdZ, permute=False)

    _k7_dispatch(A=A, B_T=B_T, O=O, Z=Z, dZ=dZ, ZdZ=ZdZ, block_size=block_size)

    dZ = dZ.squeeze(dim=-1)
    ZdZ = ZdZ.squeeze(dim=0)
    return dZ, ZdZ, O
