import torch
from coda.core.gemm.gemm_interface import _dispatch, _kernel_op, make_epi_keys
from coda.core.gemm.registry import GemmSwiGLU


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
        epi_keys=make_epi_keys(epi_args, pre_act.shape[-1]),
    )


def gemm_swiglu(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, _ = A.shape
    _, N = B.shape
    assert N % 2 == 0
    pre_act = torch.empty(M, N, dtype=A.dtype, device=A.device)
    post_act = torch.empty(M, N // 2, dtype=A.dtype, device=A.device)
    _gemm_swiglu(A=A, B=B, pre_act=pre_act, post_act=post_act)
    return pre_act, post_act
