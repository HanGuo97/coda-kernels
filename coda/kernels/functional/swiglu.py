import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.elementwise.functional import dswiglu_backward
from coda.core.gemm.functional import gemm, gemm_swiglu


class LinearSwiGLU(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        pre_act, out = gemm_swiglu(x, weight.mT)
        ctx.save_for_backward(x, weight, pre_act)
        return out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, weight, pre_act = ctx.saved_tensors
        grad_pre = dswiglu_backward(pre_act, dout)
        dx = gemm(grad_pre, weight)
        dweight = gemm(grad_pre.mT, x)
        return dx, dweight


def linear_swiglu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return LinearSwiGLU.apply(x, weight)
