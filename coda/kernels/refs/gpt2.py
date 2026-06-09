import torch
from einops import rearrange, reduce


@torch.compile(fullgraph=True, dynamic=False)
def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GEMM with residual, partial RMSNorm reduction,
    and fused RMSNorm-weight scaling.

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
    D = torch.addmm(C, A, B)
    S = reduce(D ** 2, "m (nb bs) -> m nb", "mean", bs=block_size)
    O = D * rearrange(W, "n -> 1 n")
    return D.to(dtype=A.dtype), S, O.to(dtype=A.dtype)


@torch.compile(fullgraph=True, dynamic=False)
def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation for GEMM with RMSNorm normalization.

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
    D = torch.mm(A, B)
    D = D * rearrange(R, "m -> m 1")
    return D.to(dtype=A.dtype)


@torch.compile(fullgraph=True, dynamic=False)
def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GEMM with SwiGLU activation.

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
    D = torch.mm(A, B)
    D2 = rearrange(D, "... (n pair) -> ... n pair", pair=2)
    G = D2[..., 0]
    U = D2[..., 1]
    O = torch.nn.functional.silu(G) * U
    return D.to(dtype=A.dtype), O.to(dtype=A.dtype)


@torch.compile(fullgraph=True, dynamic=False)
def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GEMM with RMSNorm and SwiGLU activation.

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
    D = torch.mm(A, B)
    D = D * rearrange(R, "m -> m 1")
    D2 = rearrange(D, "... (n pair) -> ... n pair", pair=2)
    G = D2[..., 0]
    U = D2[..., 1]
    O = torch.nn.functional.silu(G) * U
    return D.to(dtype=A.dtype), O.to(dtype=A.dtype)


def rope(
    X: torch.Tensor,
    cos_sin: torch.Tensor,
    backward: bool = False,
) -> torch.Tensor:
    """
    Apply RoPE (Rotary Position Embedding) to input tensor.

    Applies rotation to pairs of values: [x', y'] = [x*cos + y*sin, -x*sin + y*cos]

    Args:
        X: Input tensor to rotate
        cos_sin: Rotation tensor with interleaved [cos, sin] pairs

    Returns:
        Rotated output tensor with same shape as X
    """
    X2 = rearrange(X, "... (n pair) -> ... n pair", pair=2)
    cos_sin = rearrange(cos_sin, "... (n pair) -> ... n pair", pair=2)

    if backward:
        sign = -1.
    else:
        sign = 1.

    # both `cos, sin` are broadcast, and zero-padded so we only rotate `q` and `k`
    O2 = X2.clone()
    O2[..., 0] = X2[..., 0] *   cos_sin[..., 0]         + X2[..., 1] * (cos_sin[..., 1] * sign)
    O2[..., 1] = X2[..., 0] * (-cos_sin[..., 1] * sign) + X2[..., 1] *  cos_sin[..., 0]
    O = rearrange(O2, "... n pair -> ... (n pair)", pair=2)
    return O


@torch.compile(fullgraph=True, dynamic=False)
def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GEMM with RoPE positional encoding.

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
    D = torch.mm(A, B)
    O = rope(D, cos_sin=cos_sin)
    return D.to(dtype=A.dtype), O.to(dtype=A.dtype)


@torch.compile(fullgraph=True, dynamic=False)
def gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GEMM with RMSNorm and RoPE positional encoding.

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
    D = torch.mm(A, B)
    D = D * rearrange(R, "m -> m 1")
    O = rope(D, cos_sin=cos_sin)
    return D.to(dtype=A.dtype), O.to(dtype=A.dtype)


@torch.compile(fullgraph=True, dynamic=False)
def gemm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GEMM with target logit selection and partial LSE.

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
    logits = torch.mm(A, B)
    # logits of the target
    logits_tgt = logits[torch.arange(logits.shape[0], device=logits.device), targets]
    # logits in blocks → fused logsumexp per block
    logits_blk = rearrange(logits, "m (nb bs) -> m nb bs", bs=block_size)
    logits_lse = torch.logsumexp(logits_blk, dim=-1)
    return logits.to(dtype=A.dtype), logits_tgt.to(dtype=A.dtype), logits_lse


@torch.compile(fullgraph=True, dynamic=False)
def gemm_rmsnorm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation for GEMM with RMSNorm and partial LSE.

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
    logits = torch.mm(A, B)
    logits = logits * rearrange(R, "m -> m 1")
    # logits of the target
    logits_tgt = logits[torch.arange(logits.shape[0], device=logits.device), targets]
    # logits in blocks → fused logsumexp per block
    logits_blk = rearrange(logits, "m (nb bs) -> m nb bs", bs=block_size)
    logits_lse = torch.logsumexp(logits_blk, dim=-1)
    return logits.to(dtype=A.dtype), logits_tgt.to(dtype=A.dtype), logits_lse


@torch.compile(fullgraph=True, dynamic=False)
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
    Reference implementation for backward pass of GEMM with residual, partial
    RMSNorm reduction, and fused RMSNorm-weight scaling.

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
    # `B` will be transposed
    D = torch.mm(A, B.T)
    ZdZ = rearrange(ZdZ, "m -> m 1")
    W = rearrange(W, "n -> 1 n")
    R = rearrange(R, "m -> m 1")
    C_norm = C * R
    O = O + (D * W - C_norm * ZdZ) * R
    C_out = C_norm * W
    dW = reduce(D * C_norm, "(nb bs) n -> n nb", "sum", bs=block_size)
    return O.to(dtype=A.dtype), C_out.to(dtype=A.dtype), dW


@torch.compile(fullgraph=True, dynamic=False)
def gemm_partial_swiglu_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference implementation for backward pass of GEMM with SwiGLU activation.

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
    assert Z.shape[1] == B.shape[0] * 2
    D = torch.mm(A, B.T)

    Z2 = rearrange(Z, "... (n pair) -> ... n pair", pair=2)
    G = Z2[..., 0]
    U = Z2[..., 1]

    G1 = torch.nn.functional.sigmoid(G)
    G2 = torch.nn.functional.silu(G)

    O = G2 * U
    dU = D * G2
    dG = D * U * (G1 + G2 * (1.0 - G1))

    dZ2 = Z2.clone()
    dZ2[..., 0] = dG
    dZ2[..., 1] = dU
    dZ = rearrange(dZ2, "... n pair -> ... (n pair)", pair=2)
    ZdZ = reduce(Z2 * dZ2, "m (nb bs) pair -> m nb", "sum", bs=block_size, pair=2) / block_size
    return dZ.to(dtype=A.dtype), ZdZ, O.to(dtype=A.dtype)
