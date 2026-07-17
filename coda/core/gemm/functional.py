import torch
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm as quack_gemm
from quack.cross_entropy import cross_entropy_fwd_out
from quack.autotuner import autotune, AutotuneConfig

from coda.core.epilogue.utils import preprocess_epi_args, make_epi_keys
from coda.core.gemm.gemm_interface import (
    _dispatch,
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


@_kernel_op("coda::gemm", mutates_args=("out",))
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


@_kernel_op("coda::gemm_swiglu", mutates_args=("pre_act", "post_act"))
def _gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    pre_act: torch.Tensor,
    post_act: torch.Tensor,
) -> None:
    epi_args = {"mAuxOut": post_act}
    _dispatch(
        GemmCls=GemmSwiGLU,
        A=A,
        B=B,
        D=pre_act,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmSwiGLU, epi_args),
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
    epi_args = preprocess_epi_args(GemmCls=GemmSwiGLU, epi_args={"mAuxOut": post_act})
    _gemm_swiglu(A=A, B=B, pre_act=pre_act, post_act=epi_args["mAuxOut"])
    return pre_act, post_act


@_kernel_op("coda::gemm_rope", mutates_args=("D",))
def _gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
) -> None:
    epi_args = {
        "mPos": pos,
        "mFreq": freq,
    }
    _dispatch(
        GemmCls=GemmRoPE,
        A=A,
        B=B,
        D=D,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmRoPE, epi_args),
    )


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    D: torch.Tensor | None = None,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
    if D is None:
        D = torch.empty(M, N, dtype=A.dtype, device=A.device)
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
        D=D,
        pos=epi_args["mPos"],
        freq=epi_args["mFreq"],
    )
    return D


def gemm_lse(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, vocab_size = B.shape
    logits = torch.empty(M, vocab_size, dtype=A.dtype, device=A.device)
    lses = torch.empty(M, dtype=torch.float32, device=A.device)
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
    config: GemmConfig | None,
) -> None:
    if config is None:
        config = default_config(A.device)

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
    _gemm_epilogue_tuned.fn(
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


@_kernel_op("coda::gemm_lse_select_logits", mutates_args=("lses", "losses", "target_logits"))
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
    A, B, _, _ = _preprocess_gemm_operands(
        A=A,
        B=B,
        D=None,
        C=None,
    )
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
