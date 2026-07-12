import torch
from quack.gemm_interface import gemm as quack_gemm
from quack.autotuner import autotune, AutotuneConfig

from coda.core.epilogue.utils import preprocess_epi_args, make_epi_keys
from coda.core.gemm.gemm_interface import (
    _dispatch,
    _kernel_op,
    _gemm_epilogue_tuned,
    prune_gemm_configs,
)
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


@_kernel_op("coda::gemm_rope", mutates_args=("D",))
def _gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
) -> None:
    epi_args = {"mPos": pos, "mFreq": freq}
    _dispatch(
        GemmCls=GemmRoPE,
        A=A,
        B=B,
        D=D,
        epi_args=epi_args,
        epi_keys=make_epi_keys(GemmRoPE, epi_args),
    )


def gemm_swiglu(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0, f"swiglu needs an even gate||up width, got N={N}"
    pre_act = torch.empty(M, N, dtype=A.dtype, device=A.device)
    post_act = torch.empty(M, N // 2, dtype=A.dtype, device=A.device)
    epi_args = preprocess_epi_args(GemmCls=GemmSwiGLU, epi_args={"mAuxOut": post_act})
    _gemm_swiglu(A=A, B=B, pre_act=pre_act, post_act=epi_args["mAuxOut"])
    return pre_act, post_act


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
) -> torch.Tensor:
    M, _ = A.shape
    _, N = B.shape
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
