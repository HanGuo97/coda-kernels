import torch
from quack.cross_entropy import cross_entropy_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.elementwise.functional import cross_entropy_dlogits
from coda.core.gemm.functional import gemm, gemm_lse


def _forward_dlogits(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gemm(x, weight.mT)
    return cross_entropy_fwd(
        x=logits,
        target=target,
        ignore_index=ignore_index,
        return_dx=True,
        inplace_backward=True,
    )


def _forward_lse(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, lses = gemm_lse(x, weight.mT)
    return cross_entropy_dlogits(
        lses=lses,
        logits=logits,
        target=target,
        ignore_index=ignore_index,
    )


def _backward_dlogits(
    x: torch.Tensor,
    weight: torch.Tensor,
    dlogits: torch.Tensor,
    dloss: torch.Tensor,
    need_dx: bool,
    need_dweight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if need_dx:
        dx = gemm(dlogits, weight, alpha=dloss)
    else:
        dx = None
    if need_dweight:
        dweight = gemm(dlogits.mT, x, alpha=dloss)
    else:
        dweight = None
    return dx, dweight


class LinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        reduction: str,
        fused_lse: bool,
    ) -> torch.Tensor:

        if fused_lse:
            losses, dlogits = _forward_lse(
                x=x,
                weight=weight,
                target=target,
                ignore_index=ignore_index,
            )
        else:
            losses, dlogits = _forward_dlogits(
                x=x,
                weight=weight,
                target=target,
                ignore_index=ignore_index,
            )

        if reduction == "mean":
            scale = 1.0 / (target != ignore_index).sum().float()
            loss = losses.sum() * scale
        else:
            scale = None
            loss = losses.sum()

        ctx.save_for_backward(x, weight, dlogits, scale)
        return loss

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dloss: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        x, weight, dlogits, scale = ctx.saved_tensors
        if scale is not None:
            dloss_scaled = dloss * scale
        else:
            dloss_scaled = dloss
        dx, dweight = _backward_dlogits(
            x=x,
            weight=weight,
            dlogits=dlogits,
            dloss=dloss_scaled,
            need_dx=ctx.needs_input_grad[0],
            need_dweight=ctx.needs_input_grad[1],
        )
        return dx, dweight, None, None, None, None


def linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    fused_lse: bool = True,
) -> torch.Tensor:
    assert reduction in ("mean", "sum")
    return LinearCrossEntropy.apply(
        x,
        weight,
        target,
        ignore_index,
        reduction,
        fused_lse,
    )
