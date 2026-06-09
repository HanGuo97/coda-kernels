import math
import torch
import pytest
import cutlass.cute.testing as testing
from einops import repeat, rearrange, reduce
from rapier.ops import benchmark_utils

from ..gens import gpt as gpt_gens
from ..refs import gpt as gpt_refs

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

MOptions = [2048, 4096, 8192]
NOptions = [2048, 4096, 8192]
KOptions = [2048, 4096, 8192]
NOptions3 = [N * 3 for N in NOptions]
BlockSizeOptions = [64, 128, 192, 256]
VocabSizeOptions = [32768]
DTypeOptions = [torch.float16, torch.bfloat16]


def _skip_if_not_divisible(size: int, block_size: int) -> None:
    if size % block_size != 0:
        pytest.skip(f"size={size} not divisible by block_size={block_size}")


def prepare_rope_inputs(
    seq_len: int,
    head_dim: int,
    batch_size: int | None,
    num_heads: int | None,
    base: int = 10000,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # stride the channels
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # stride the time steps
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)

    if batch_size is not None and num_heads is not None:
        cos = repeat(cos, "t d -> (b t) (h d)", b=batch_size, h=num_heads)
        sin = repeat(sin, "t d -> (b t) (h d)", b=batch_size, h=num_heads)
        cos = torch.stack([cos.clone(), cos.clone(), torch.ones_like(cos)], dim=-1)
        sin = torch.stack([sin.clone(), sin.clone(), torch.zeros_like(sin)], dim=-1)
        cos = rearrange(cos, "m n trio -> m (trio n)", trio=3)
        sin = rearrange(sin, "m n trio -> m (trio n)", trio=3)

    return cos, sin


def create_gpt_inputs(
    M: int,
    N: int,
    K: int,
    vocab_size: int | None,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
    rope: bool = False,
    backward: bool = False,
    transpose_B: bool = False,
    seed: int = 0,
) -> tuple[dict, dict, dict, dict]:
    if device is None:
        device = "cuda"
    if vocab_size is not None:
        assert N == vocab_size

    torch.random.manual_seed(seed)

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device) / math.sqrt(K)
    C = torch.randn(M, N, dtype=dtype, device=device)
    R = torch.rsqrt(reduce(A.float() ** 2, "m k -> m", "mean") + 1e-6)
    W = torch.randn(N, dtype=dtype, device=device)
    targets = torch.randint(vocab_size, (M,), device=device) if vocab_size is not None else None

    if rope:
        # just some reasonable defaults
        # 3 is for q, k, v
        batch_size = 4
        seq_length = int(M / batch_size)
        num_heads = 16
        head_dim = int(N / (num_heads * 3))
        assert head_dim != 2
        assert M == batch_size * seq_length
        assert N == num_heads * head_dim * 3
        cos, sin = prepare_rope_inputs(
            seq_len=seq_length,
            head_dim=head_dim,
            batch_size=None,
            num_heads=None,
            dtype=dtype,
            device=device,
        )
        cos_, sin_ = prepare_rope_inputs(
            seq_len=seq_length,
            head_dim=head_dim,
            batch_size=batch_size,
            num_heads=num_heads,
            dtype=dtype,
            device=device,
        )
        cos_sin = torch.stack([cos_, sin_], dim=-1)
        cos_sin = rearrange(cos_sin, "... n pair -> ... (n pair)", pair=2)
        positions = torch.arange(seq_length, device=device).repeat(batch_size)
    else:
        cos = None
        sin = None
        cos_sin = None
        positions = None

    if backward:
        Z = torch.randn(M, 2 * N, dtype=dtype, device=device)
        O = torch.randn(M, N, dtype=dtype, device=device)
        ZdZ = torch.randn(M, dtype=torch.float32, device=device)
    else:
        Z = None
        O = None
        ZdZ = None

    if transpose_B:
        B = B.mT.contiguous()

    A_ref = A.detach().clone()
    B_ref = B.detach().clone()
    C_ref = C.detach().clone()
    R_ref = R.detach().clone()
    W_ref = W.detach().clone()
    Z_ref = Z.detach().clone() if Z is not None else None
    O_ref = O.detach().clone() if O is not None else None
    ZdZ_ref = ZdZ.detach().clone() if ZdZ is not None else None
    cos_ref = cos.detach().clone() if cos is not None else None
    sin_ref = sin.detach().clone() if sin is not None else None
    cos_sin_ref = cos_sin.detach().clone() if cos_sin is not None else None

    A_ref_fp32 = A.detach().clone().float()
    B_ref_fp32 = B.detach().clone().float()
    C_ref_fp32 = C.detach().clone().float()
    R_ref_fp32 = R.detach().clone().float()
    W_ref_fp32 = W.detach().clone().float()
    Z_ref_fp32 = Z.detach().clone().float() if Z is not None else None
    O_ref_fp32 = O.detach().clone().float() if O is not None else None
    ZdZ_ref_fp32 = ZdZ.detach().clone().float() if ZdZ is not None else None
    cos_ref_fp32 = cos.detach().clone().float() if cos is not None else None
    sin_ref_fp32 = sin.detach().clone().float() if sin is not None else None
    cos_sin_ref_fp32 = cos_sin.detach().clone().float() if cos_sin is not None else None

    return (
        {
            "A": A,
            "B": B,
            "C": C,
            "R": R,
            "W": W,
            "Z": Z,
            "O": O,
            "ZdZ": ZdZ,
            "cos": cos,
            "sin": sin,
            "cos_sin": cos_sin,
        },
        {
            "A": A_ref,
            "B": B_ref,
            "C": C_ref,
            "R": R_ref,
            "W": W_ref,
            "Z": Z_ref,
            "O": O_ref,
            "ZdZ": ZdZ_ref,
            "cos": cos_ref,
            "sin": sin_ref,
            "cos_sin": cos_sin_ref,
        },
        {
            "A": A_ref_fp32,
            "B": B_ref_fp32,
            "C": C_ref_fp32,
            "R": R_ref_fp32,
            "W": W_ref_fp32,
            "Z": Z_ref_fp32,
            "O": O_ref_fp32,
            "ZdZ": ZdZ_ref_fp32,
            "cos": cos_ref_fp32,
            "sin": sin_ref_fp32,
            "cos_sin": cos_sin_ref_fp32,
        },
        {
            "targets": targets,
            "positions": positions,
        }
    )


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("block_size", BlockSizeOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_residual_partial_rmsnorm(M: int, N: int, K: int, block_size: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with residual, partial RMSNorm reduction, and fused RMSNorm-weight scaling.

    Compares the output of gemm_residual_partial_rmsnorm against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Shared dimension (columns of A, rows of B)
        block_size: Block size for partial reduction
        dtype: Data type for tensors
    """
    _skip_if_not_divisible(
        size=N,
        block_size=block_size,
    )
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_residual_partial_rmsnorm,
        A=inputs["A"],
        B=inputs["B"],
        C=inputs["C"],
        W=inputs["W"],
        block_size=block_size,
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    D, S, O = gpt_gens.gemm_residual_partial_rmsnorm(
        A=inputs["A"],
        B=inputs["B"],
        C=inputs["C"],
        W=inputs["W"],
        block_size=block_size,
    )
    D_ref, S_ref, O_ref = gpt_refs.gemm_residual_partial_rmsnorm(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        C=inputs_ref["C"],
        W=inputs_ref["W"],
        block_size=block_size,
    )
    D_ref_fp32, S_ref_fp32, O_ref_fp32 = gpt_refs.gemm_residual_partial_rmsnorm(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        C=inputs_ref_fp32["C"],
        W=inputs_ref_fp32["W"],
        block_size=block_size,
    )

    assert D.shape == (M, N)
    assert S.shape == (M, int(N / block_size))
    assert O.shape == (M, N)
    assert D.dtype == dtype
    assert S.dtype == torch.float32
    assert O.dtype == dtype
    assert (D - D_ref_fp32).abs().max() < 2 * (D_ref - D_ref_fp32).abs().max() + 1e-6
    assert (S - S_ref_fp32).abs().max() < 2 * (S_ref - S_ref_fp32).abs().max() + 1e-6
    assert (O - O_ref_fp32).abs().max() < 2 * (O_ref - O_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_rmsnorm(M: int, N: int, K: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with RMSNorm normalization.

    Compares the output of gemm_rmsnorm against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Shared dimension (columns of A, rows of B)
        dtype: Data type for tensors
    """
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_rmsnorm,
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    D = gpt_gens.gemm_rmsnorm(
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
    )
    D_ref = gpt_refs.gemm_rmsnorm(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        R=inputs_ref["R"],
    )
    D_ref_fp32 = gpt_refs.gemm_rmsnorm(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        R=inputs_ref_fp32["R"],
    )

    assert D.shape == (M, N)
    assert D.dtype == dtype
    assert (D - D_ref_fp32).abs().max() < 2 * (D_ref - D_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_swiglu(M: int, N: int, K: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with SwiGLU activation.

    Compares the output of gemm_swiglu against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Shared dimension (columns of A, rows of B)
        dtype: Data type for tensors
    """
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_swiglu,
        A=inputs["A"],
        B=inputs["B"],
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    D, O = gpt_gens.gemm_swiglu(
        A=inputs["A"],
        B=inputs["B"],
    )
    D_ref, O_ref = gpt_refs.gemm_swiglu(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
    )
    D_ref_fp32, O_ref_fp32 = gpt_refs.gemm_swiglu(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
    )

    assert D.shape == (M, N)
    assert O.shape == (M, N // 2)
    assert D.dtype == dtype
    assert O.dtype == dtype
    assert (D - D_ref_fp32).abs().max() < 2 * (D_ref - D_ref_fp32).abs().max() + 1e-6
    assert (O - O_ref_fp32).abs().max() < 2 * (O_ref - O_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_rmsnorm_swiglu(M: int, N: int, K: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with RMSNorm and SwiGLU activation.

    Compares the output of gemm_rmsnorm_swiglu against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Shared dimension (columns of A, rows of B)
        dtype: Data type for tensors
    """
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_rmsnorm_swiglu,
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    D, O = gpt_gens.gemm_rmsnorm_swiglu(
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
    )
    D_ref, O_ref = gpt_refs.gemm_rmsnorm_swiglu(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        R=inputs_ref["R"],
    )
    D_ref_fp32, O_ref_fp32 = gpt_refs.gemm_rmsnorm_swiglu(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        R=inputs_ref_fp32["R"],
    )

    assert D.shape == (M, N)
    assert O.shape == (M, N // 2)
    assert D.dtype == dtype
    assert O.dtype == dtype
    assert (D - D_ref_fp32).abs().max() < 2 * (D_ref - D_ref_fp32).abs().max() + 1e-6
    assert (O - O_ref_fp32).abs().max() < 2 * (O_ref - O_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions3)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_rope(M: int, N: int, K: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with RoPE positional encoding.

    Compares output of gemm_rope against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B (must be divisible by 3 for q,k,v heads)
        K: Shared dimension (columns of A, rows of B)
        dtype: Data type for tensors
    """
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        rope=True,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_rope,
        A=inputs["A"],
        B=inputs["B"],
        cos_sin=inputs["cos_sin"],
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    D, O = gpt_gens.gemm_rope(
        A=inputs["A"],
        B=inputs["B"],
        cos_sin=inputs["cos_sin"],
    )
    D_ref, O_ref = gpt_refs.gemm_rope(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        cos_sin=inputs_ref["cos_sin"],
    )
    D_ref_fp32, O_ref_fp32 = gpt_refs.gemm_rope(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        cos_sin=inputs_ref_fp32["cos_sin"],
    )

    assert D.shape == (M, N)
    assert O.shape == (M, N)
    assert D.dtype == dtype
    assert O.dtype == dtype
    assert (D - D_ref_fp32).abs().max() < 2 * (D_ref - D_ref_fp32).abs().max() + 1e-6
    assert (O - O_ref_fp32).abs().max() < 2 * (O_ref - O_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions3)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_rmsnorm_rope(M: int, N: int, K: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with RMSNorm and RoPE positional encoding.

    Compares output of gemm_rmsnorm_rope against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of columns in matrix B (must be divisible by 3 for q,k,v heads)
        K: Shared dimension (columns of A, rows of B)
        dtype: Data type for tensors
    """
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        rope=True,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_rmsnorm_rope,
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
        cos_sin=inputs["cos_sin"],
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    D, O = gpt_gens.gemm_rmsnorm_rope(
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
        cos_sin=inputs["cos_sin"],
    )
    D_ref, O_ref = gpt_refs.gemm_rmsnorm_rope(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        R=inputs_ref["R"],
        cos_sin=inputs_ref["cos_sin"],
    )
    D_ref_fp32, O_ref_fp32 = gpt_refs.gemm_rmsnorm_rope(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        R=inputs_ref_fp32["R"],
        cos_sin=inputs_ref_fp32["cos_sin"],
    )

    assert D.shape == (M, N)
    assert O.shape == (M, N)
    assert D.dtype == dtype
    assert O.dtype == dtype
    assert (D - D_ref_fp32).abs().max() < 2 * (D_ref - D_ref_fp32).abs().max() + 1e-6
    assert (O - O_ref_fp32).abs().max() < 2 * (O_ref - O_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("block_size", BlockSizeOptions)
@pytest.mark.parametrize("vocab_size", VocabSizeOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_partial_cross_entropy(M: int, K: int, block_size: int, vocab_size: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with target logit selection and partial LSE.

    Compares the output of gemm_partial_cross_entropy against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        K: Shared dimension (columns of A, rows of B)
        block_size: Block size for partial reduction
        vocab_size: Number of columns in matrix B
        dtype: Data type for tensors
    """
    _skip_if_not_divisible(
        size=vocab_size,
        block_size=block_size,
    )
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=vocab_size,
        K=K,
        vocab_size=vocab_size,
        dtype=dtype,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_partial_cross_entropy,
        A=inputs["A"],
        B=inputs["B"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    logits, logits_tgt, logits_lse = gpt_gens.gemm_partial_cross_entropy(
        A=inputs["A"],
        B=inputs["B"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )
    logits_ref, logits_tgt_ref, logits_lse_ref = gpt_refs.gemm_partial_cross_entropy(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )
    logits_ref_fp32, logits_tgt_ref_fp32, logits_lse_ref_fp32 = gpt_refs.gemm_partial_cross_entropy(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )

    assert logits.shape == (M, vocab_size)
    assert logits_tgt.shape == (M,)
    assert logits_lse.shape == (M, int(vocab_size / block_size))
    assert logits.dtype == dtype
    assert logits_tgt.dtype == dtype
    assert logits_lse.dtype == torch.float32
    assert (logits - logits_ref_fp32).abs().max() < 2 * (logits_ref - logits_ref_fp32).abs().max() + 1e-6
    assert (logits_tgt - logits_tgt_ref_fp32).abs().max() < 2 * (logits_tgt_ref - logits_tgt_ref_fp32).abs().max() + 1e-6
    assert (logits_lse - logits_lse_ref_fp32).abs().max() < 2 * (logits_lse_ref - logits_lse_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("block_size", BlockSizeOptions)
@pytest.mark.parametrize("vocab_size", VocabSizeOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_rmsnorm_partial_cross_entropy(M: int, K: int, block_size: int, vocab_size: int, dtype: torch.dtype) -> None:
    """
    Test GEMM with RMSNorm and partial LSE.

    Compares the output of gemm_rmsnorm_partial_cross_entropy against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        K: Shared dimension (columns of A, rows of B)
        block_size: Block size for partial reduction
        vocab_size: Number of columns in matrix B
        dtype: Data type for tensors
    """
    _skip_if_not_divisible(
        size=vocab_size,
        block_size=block_size,
    )
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=vocab_size,
        K=K,
        vocab_size=vocab_size,
        dtype=dtype,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_rmsnorm_partial_cross_entropy,
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    logits, logits_tgt, logits_lse = gpt_gens.gemm_rmsnorm_partial_cross_entropy(
        A=inputs["A"],
        B=inputs["B"],
        R=inputs["R"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )
    logits_ref, logits_tgt_ref, logits_lse_ref = gpt_refs.gemm_rmsnorm_partial_cross_entropy(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        R=inputs_ref["R"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )
    logits_ref_fp32, logits_tgt_ref_fp32, logits_lse_ref_fp32 = gpt_refs.gemm_rmsnorm_partial_cross_entropy(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        R=inputs_ref_fp32["R"],
        targets=auxiliary["targets"],
        block_size=block_size,
    )

    assert logits.shape == (M, vocab_size)
    assert logits_tgt.shape == (M,)
    assert logits_lse.shape == (M, int(vocab_size / block_size))
    assert logits.dtype == dtype
    assert logits_tgt.dtype == dtype
    assert logits_lse.dtype == torch.float32
    assert (logits - logits_ref_fp32).abs().max() < 2 * (logits_ref - logits_ref_fp32).abs().max() + 1e-6
    assert (logits_tgt - logits_tgt_ref_fp32).abs().max() < 2 * (logits_tgt_ref - logits_tgt_ref_fp32).abs().max() + 1e-6
    assert (logits_lse - logits_lse_ref_fp32).abs().max() < 2 * (logits_lse_ref - logits_lse_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("block_size", BlockSizeOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_residual_partial_rmsnorm_bwd(M: int, N: int, K: int, block_size: int, dtype: torch.dtype) -> None:
    """
    Test backward pass for GEMM with residual, partial RMSNorm reduction, and
    fused RMSNorm-weight scaling.

    Compares output of gemm_residual_partial_rmsnorm_bwd against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of rows in matrix B
        K: Shared dimension (columns of A, columns of B)
        block_size: Block size for partial reduction
        dtype: Data type for tensors
    """
    # Note that for backward, `B` has shape (N, K)
    _skip_if_not_divisible(
        size=M,
        block_size=block_size,
    )
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        backward=True,
        transpose_B=True,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_residual_partial_rmsnorm_bwd,
        A=inputs["A"],
        B=inputs["B"],
        C=inputs["C"],
        W=inputs["W"],
        R=inputs["R"],
        ZdZ=inputs["ZdZ"],
        # `O` is mutated in-place
        O=inputs["O"].clone(),
        block_size=block_size,
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    O, C, dW = gpt_gens.gemm_residual_partial_rmsnorm_bwd(
        A=inputs["A"],
        B=inputs["B"],
        C=inputs["C"],
        W=inputs["W"],
        R=inputs["R"],
        ZdZ=inputs["ZdZ"],
        O=inputs["O"],
        block_size=block_size,
    )
    O_ref, C_ref, dW_ref = gpt_refs.gemm_residual_partial_rmsnorm_bwd(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        C=inputs_ref["C"],
        W=inputs_ref["W"],
        R=inputs_ref["R"],
        ZdZ=inputs_ref["ZdZ"],
        O=inputs_ref["O"],
        block_size=block_size,
    )
    O_ref_fp32, C_ref_fp32, dW_ref_fp32 = gpt_refs.gemm_residual_partial_rmsnorm_bwd(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        C=inputs_ref_fp32["C"],
        W=inputs_ref_fp32["W"],
        R=inputs_ref_fp32["R"],
        ZdZ=inputs_ref_fp32["ZdZ"],
        O=inputs_ref_fp32["O"],
        block_size=block_size,
    )

    assert O.shape == (M, N)
    assert C.shape == (M, N)
    assert dW.shape == (N, int(M / block_size))
    assert O.dtype == dtype
    assert C.dtype == dtype
    assert dW.dtype == torch.float32
    assert (O - O_ref_fp32).abs().max() < 2 * (O_ref - O_ref_fp32).abs().max() + 1e-6
    assert (C - C_ref_fp32).abs().max() < 2 * (C_ref - C_ref_fp32).abs().max() + 1e-6
    assert (dW - dW_ref_fp32).abs().max() < 2 * (dW_ref - dW_ref_fp32).abs().max() + 1e-6


@pytest.mark.parametrize("M", MOptions)
@pytest.mark.parametrize("N", NOptions)
@pytest.mark.parametrize("K", KOptions)
@pytest.mark.parametrize("block_size", BlockSizeOptions)
@pytest.mark.parametrize("dtype", DTypeOptions)
def test_gemm_partial_swiglu_bwd(M: int, N: int, K: int, block_size: int, dtype: torch.dtype) -> None:
    """
    Test backward pass for GEMM with SwiGLU activation.

    Compares output of gemm_partial_swiglu_bwd against reference implementation
    in both original dtype and fp32 for numerical accuracy validation.

    Args:
        M: Number of rows in matrix A
        N: Number of rows in matrix B
        K: Shared dimension (columns of A, columns of B)
        block_size: Block size for partial reduction
        dtype: Data type for tensors
    """
    # Note that for backward, `B` has shape (N, K)
    _skip_if_not_divisible(
        size=N,
        block_size=block_size,
    )
    inputs, inputs_ref, inputs_ref_fp32, auxiliary = create_gpt_inputs(
        M=M,
        N=N,
        K=K,
        vocab_size=None,
        dtype=dtype,
        backward=True,
        transpose_B=True,
    )

    total_launches, _ = benchmark_utils.count_cuda_launch_calls(
        gpt_gens.gemm_partial_swiglu_bwd,
        A=inputs["A"],
        B=inputs["B"],
        Z=inputs["Z"],
        block_size=block_size,
    )
    assert total_launches == 1, f"Expected 1 kernel launch, got {total_launches}"

    dZ, ZdZ, O = gpt_gens.gemm_partial_swiglu_bwd(
        A=inputs["A"],
        B=inputs["B"],
        Z=inputs["Z"],
        block_size=block_size,
    )
    dZ_ref, ZdZ_ref, O_ref = gpt_refs.gemm_partial_swiglu_bwd(
        A=inputs_ref["A"],
        B=inputs_ref["B"],
        Z=inputs_ref["Z"],
        block_size=block_size,
    )
    dZ_ref_fp32, ZdZ_ref_fp32, O_ref_fp32 = gpt_refs.gemm_partial_swiglu_bwd(
        A=inputs_ref_fp32["A"],
        B=inputs_ref_fp32["B"],
        Z=inputs_ref_fp32["Z"],
        block_size=block_size,
    )

    assert O.shape == (M, N)
    assert dZ.shape == (M, 2 * N)
    assert ZdZ.shape == (M, int(N / block_size))
    assert O.dtype == dtype
    assert dZ.dtype == dtype
    assert ZdZ.dtype == torch.float32
    assert (O - O_ref_fp32).abs().max() < 2 * (O_ref - O_ref_fp32).abs().max() + 1e-6
    assert (dZ - dZ_ref_fp32).abs().max() < 2 * (dZ_ref - dZ_ref_fp32).abs().max() + 1e-6
    assert (ZdZ - ZdZ_ref_fp32).abs().max() < 2 * (ZdZ_ref - ZdZ_ref_fp32).abs().max() + 1e-6


def gemm_residual_partial_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    W: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    test_gemm_residual_partial_rmsnorm(
        M=A.shape[0],
        N=B.shape[1],
        K=A.shape[1],
        block_size=block_size,
        dtype=A.dtype,
    )
    return gpt_gens.gemm_residual_partial_rmsnorm(
        A=A,
        B=B,
        C=C,
        W=W,
        block_size=block_size,
    )


def gemm_rmsnorm(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    test_gemm_rmsnorm(
        M=A.shape[0],
        N=B.shape[1],
        K=A.shape[1],
        dtype=A.dtype,
    )
    return gpt_gens.gemm_rmsnorm(
        A=A,
        B=B,
        R=R,
    )


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    test_gemm_swiglu(
        M=A.shape[0],
        N=B.shape[1],
        K=A.shape[1],
        dtype=A.dtype,
    )
    return gpt_gens.gemm_swiglu(
        A=A,
        B=B,
    )


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    test_gemm_rmsnorm_swiglu(
        M=A.shape[0],
        N=B.shape[1],
        K=A.shape[1],
        dtype=A.dtype,
    )
    return gpt_gens.gemm_rmsnorm_swiglu(
        A=A,
        B=B,
        R=R,
    )


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    test_gemm_rope(
        M=A.shape[0],
        N=B.shape[1],
        K=A.shape[1],
        dtype=A.dtype,
    )
    return gpt_gens.gemm_rope(
        A=A,
        B=B,
        cos_sin=cos_sin,
    )


def gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    test_gemm_rmsnorm_rope(
        M=A.shape[0],
        N=B.shape[1],
        K=A.shape[1],
        dtype=A.dtype,
    )
    return gpt_gens.gemm_rmsnorm_rope(
        A=A,
        B=B,
        R=R,
        cos_sin=cos_sin,
    )


def gemm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    test_gemm_partial_cross_entropy(
        M=A.shape[0],
        K=A.shape[1],
        block_size=block_size,
        vocab_size=B.shape[1],
        dtype=A.dtype,
    )
    return gpt_gens.gemm_partial_cross_entropy(
        A=A,
        B=B,
        targets=targets,
        block_size=block_size,
    )


def gemm_rmsnorm_partial_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    targets: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    test_gemm_rmsnorm_partial_cross_entropy(
        M=A.shape[0],
        K=A.shape[1],
        block_size=block_size,
        vocab_size=B.shape[1],
        dtype=A.dtype,
    )
    return gpt_gens.gemm_rmsnorm_partial_cross_entropy(
        A=A,
        B=B,
        R=R,
        targets=targets,
        block_size=block_size,
    )


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
    # `B` will be transposed
    test_gemm_residual_partial_rmsnorm_bwd(
        M=A.shape[0],
        N=B.shape[0],
        K=A.shape[1],
        block_size=block_size,
        dtype=A.dtype,
    )
    return gpt_gens.gemm_residual_partial_rmsnorm_bwd(
        A=A,
        B=B,
        C=C,
        W=W,
        R=R,
        ZdZ=ZdZ,
        O=O,
        block_size=block_size,
    )


def gemm_partial_swiglu_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # `B` will be transposed
    test_gemm_partial_swiglu_bwd(
        M=A.shape[0],
        N=B.shape[0],
        K=A.shape[1],
        block_size=block_size,
        dtype=A.dtype,
    )
    return gpt_gens.gemm_partial_swiglu_bwd(
        A=A,
        B=B,
        Z=Z,
        block_size=block_size,
    )
