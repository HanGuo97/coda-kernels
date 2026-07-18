import torch
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm as quack_gemm
from quack.cross_entropy import cross_entropy_fwd_out
from quack.autotuner import autotune, AutotuneConfig

from coda.core.epilogue.utils import preprocess_epi_args, make_epi_keys
from coda.core.gemm.gemm_interface import (
    _kernel_op,
    _gemm_epilogue_tuned,
    _preprocess_gemm_operands,
    default_config,
    prune_gemm_configs,
    GEMM_CONFIGS,
)

from coda.core.ops import misc_utils
from coda.core.gemm.registry import (
    GemmLSE,
    GemmRoPE,
    GemmSwiGLU,
    GemmQKVSqSum,
    GemmLSESelectLogits,
)


torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)


@autotune(
    configs=[
        AutotuneConfig(backend="quack"),
        AutotuneConfig(backend="cublas"),
    ],
    cache_results=False,
)
def _gemm_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    alpha: torch.Tensor | None,
    backend: str,
) -> None:
    if backend == "quack":
        if alpha is None:
            quack_gemm(A=A, B=B, out=out, tuned=True)
        else:
            quack_gemm(A=A, B=B, out=out, alpha=alpha, tuned=True)
    else:
        torch.matmul(A, B, out=out)
        if alpha is not None:
            out.mul_(alpha)


@_kernel_op("coda::_gemm", mutates_args=("out",))
def _gemm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor | None) -> None:
    _gemm_tuned(A=A, B=B, out=out, alpha=alpha)


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    _gemm(A=A, B=B, out=out, alpha=alpha)
    return out


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_swiglu_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    post_act: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mAuxOut": post_act,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmSwiGLU,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmSwiGLU, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_swiglu", mutates_args=("D", "post_act"))
def _gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    post_act: torch.Tensor,
) -> None:
    _gemm_swiglu_tuned(
        A=A,
        B=B,
        D=D,
        post_act=post_act,
    )


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    pre_act: torch.Tensor | None = None,
    post_act: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0, f"swiglu needs an even gate||up width, got N={N}"
    if pre_act is None:
        pre_act = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if post_act is None:
        post_act = torch.empty(M, N // 2, dtype=A.dtype, device=A.device)
    A, B, pre_act, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=pre_act,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmSwiGLU,
        epi_args={
            "mAuxOut": post_act,
        },
    )
    _gemm_swiglu(
        A=A,
        B=B,
        D=pre_act,
        post_act=epi_args["mAuxOut"],
    )
    return pre_act, post_act


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_rope_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mPos": pos,
        "mFreq": freq,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmRoPE,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmRoPE, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_rope", mutates_args=("D",))
def _gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
) -> None:
    _gemm_rope_tuned(
        A=A,
        B=B,
        D=D,
        pos=pos,
        freq=freq,
    )


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    A, B, out, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmRoPE,
        epi_args={
            "mPos": pos,
            "mFreq": freq,
        },
    )
    _gemm_rope(
        A=A,
        B=B,
        D=out,
        pos=epi_args["mPos"],
        freq=epi_args["mFreq"],
    )
    return out


# a head spans at most ceil(head_dim / tile_n) + 1 tiles; size ssq for the narrowest sm90 tile
_SQSUM_MIN_TILE_N = min(
    c.tile_n
    for c in GEMM_CONFIGS
    if c.device_capacity == 9
)


def _sqsum_num_segments(head_dim: int) -> int:
    return misc_utils.ceil_div(head_dim, _SQSUM_MIN_TILE_N) + 1


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    key=["head_dim", "num_segments"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_qkv_sqsum_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ssq: torch.Tensor,
    head_dim: int,
    num_segments: int,
    config: GemmConfig,
) -> None:
    epi_args = {
        "mSqSumVec": ssq,
        "head_dim": head_dim,
        "num_segments": num_segments,
    }
    _gemm_epilogue_tuned(
        GemmCls=GemmQKVSqSum,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmQKVSqSum, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )


@_kernel_op("coda::_gemm_qkv_sqsum", mutates_args=("D", "ssq"))
def _gemm_qkv_sqsum(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    ssq: torch.Tensor,
    head_dim: int,
    num_segments: int,
) -> None:
    _gemm_qkv_sqsum_tuned(
        A=A,
        B=B,
        D=D,
        ssq=ssq,
        head_dim=head_dim,
        num_segments=num_segments,
    )


def gemm_qkv_sqsum(
    A: torch.Tensor,
    B: torch.Tensor,
    head_dim: int,
    num_segments: int,
    out: torch.Tensor | None = None,
    ssq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    if out is None:
        out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    if ssq is None:
        # zero-init as heads whose segments a tile never writes must read 0
        ssq = torch.zeros(M, (N // head_dim) * num_segments, dtype=torch.float32, device=A.device)
    A, B, out, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=out,
        C=None,
    )
    epi_args = preprocess_epi_args(
        GemmCls=GemmQKVSqSum,
        epi_args={
            "mSqSumVec": ssq,
            "head_dim": head_dim,
            "num_segments": num_segments,
        },
    )
    _gemm_qkv_sqsum(
        A=A,
        B=B,
        D=out,
        ssq=epi_args["mSqSumVec"],
        head_dim=head_dim,
        num_segments=num_segments,
    )
    return out, ssq


@torch.compile(fullgraph=True, dynamic=False)
def _lse_reduce_compiled(lses: torch.Tensor, lse_partial: torch.Tensor) -> None:
    lses.copy_(torch.logsumexp(lse_partial, dim=1))


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    key=["vocab_size"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_lse_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    lses: torch.Tensor,
    vocab_size: int,
    config: GemmConfig,
) -> None:
    M, _, _ = A.shape
    n_tiles = misc_utils.ceil_div(vocab_size, config.tile_n)
    lse_partial = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epi_args = preprocess_epi_args(
        GemmCls=GemmLSE,
        epi_args={
            "mLSEVec": lse_partial,
            "vocab_size": vocab_size,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmLSE,
        A=A,
        B=B,
        D=D,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmLSE, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )
    _lse_reduce_compiled(
        lses=lses,
        lse_partial=lse_partial,
    )


@_kernel_op("coda::_gemm_lse", mutates_args=("D", "lses"))
def _gemm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    lses: torch.Tensor,
    vocab_size: int,
) -> None:
    _gemm_lse_tuned(
        A=A,
        B=B,
        D=D,
        lses=lses,
        vocab_size=vocab_size,
    )


def gemm_lse(
    A: torch.Tensor,
    B: torch.Tensor,
    logits: torch.Tensor | None = None,
    lses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, vocab_size = B.shape
    if logits is None:
        logits = torch.empty(M, vocab_size, dtype=A.dtype, device=A.device)
    if lses is None:
        lses = torch.empty(M, dtype=torch.float32, device=A.device)
    A, B, logits, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=logits,
        C=None,
    )
    _gemm_lse(
        A=A,
        B=B,
        D=logits,
        lses=lses,
        vocab_size=vocab_size,
    )
    return logits, lses


@autotune(
    configs=[AutotuneConfig(config=c) for c in GEMM_CONFIGS],
    key=["vocab_size", "ignore_index"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    cache_results=False,
)
def _gemm_lse_select_logits_tuned(
    A: torch.Tensor,
    B: torch.Tensor,
    lses: torch.Tensor | None,
    target: torch.Tensor,
    losses: torch.Tensor,
    target_logits: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
    config: GemmConfig,
) -> None:
    M, _, _ = A.shape
    n_tiles = misc_utils.ceil_div(vocab_size, config.tile_n)
    lse_partial = torch.empty(M, n_tiles, dtype=torch.float32, device=A.device)
    epi_args = preprocess_epi_args(
        GemmCls=GemmLSESelectLogits,
        epi_args={
            "mLSEVec": lse_partial,
            "mTarget": target,
            "mLogits": target_logits,
            "vocab_size": vocab_size,
        },
    )
    _gemm_epilogue_tuned(
        GemmCls=GemmLSESelectLogits,
        A=A,
        B=B,
        D=None,
        C=None,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmLSESelectLogits, epi_args),
        pin_tile_M=None,
        pin_tile_N=None,
        batch_idx_permute=None,
        add_to_output=False,
        config=config,
    )
    cross_entropy_fwd_out(
        x=lse_partial,
        target=target,
        target_logit=target_logits,
        loss=losses,
        lse=lses,
        dx=None,
        weight=None,
        ignore_index=ignore_index,
    )


@_kernel_op("coda::_gemm_lse_select_logits", mutates_args=("lses", "losses", "target_logits"))
def _gemm_lse_select_logits(
    A: torch.Tensor,
    B: torch.Tensor,
    lses: torch.Tensor | None,
    target: torch.Tensor,
    losses: torch.Tensor,
    target_logits: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
) -> None:
    _gemm_lse_select_logits_tuned(
        A=A,
        B=B,
        lses=lses,
        target=target,
        losses=losses,
        target_logits=target_logits,
        vocab_size=vocab_size,
        ignore_index=ignore_index,
    )


def gemm_lse_select_logits(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    return_lse: bool,
    losses: torch.Tensor | None = None,
    target_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    assert target.dtype == torch.int32
    M, _ = A.shape
    _, vocab_size = B.shape
    if losses is None:
        losses = torch.empty(M, dtype=torch.float32, device=A.device)
    if target_logits is None:
        target_logits = torch.empty(M, dtype=A.dtype, device=A.device)
    if return_lse:
        lses = torch.empty(M, dtype=torch.float32, device=A.device)
    else:
        lses = None
    A, B, _, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=None,
        C=None,
    )
    _gemm_lse_select_logits(
        A=A,
        B=B,
        lses=lses,
        target=target,
        losses=losses,
        target_logits=target_logits,
        vocab_size=vocab_size,
        ignore_index=ignore_index,
    )
    return losses, lses, target_logits
