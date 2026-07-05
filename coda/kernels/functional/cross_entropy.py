import torch
from quack.cross_entropy import cross_entropy_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.gemm.functional import gemm, gemm_add_inplace, gemm_ce_grad, gemm_linear_ce


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
    losses, lses, _ = gemm_linear_ce(
        x,
        weight.mT,
        target=target,
        ignore_index=ignore_index,
    )
    return losses, lses


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
        chunk_size: int | None,
    ) -> torch.Tensor:

        if chunk_size is None:
            losses, dlogits = _forward_dlogits(
                x=x,
                weight=weight,
                target=target,
                ignore_index=ignore_index,
            )
        else:
            losses, lses = _forward_lse(
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

        if chunk_size is None:
            ctx.save_for_backward(x, weight, dlogits, scale)
        else:
            ctx.save_for_backward(x, weight, target, lses, scale)

        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        return loss

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dloss: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        if ctx.chunk_size is None:
            x, weight, dlogits, scale = ctx.saved_tensors
            dx, dweight = _backward_dlogits(
                x=x,
                weight=weight,
                dlogits=dlogits,
                dloss=(dloss * scale),
                need_dx=ctx.needs_input_grad[0],
                need_dweight=ctx.needs_input_grad[1],
            )
        else:
            x, weight, target, lses, scale = ctx.saved_tensors
            dx, dweight = linear_ce_backward(
                x=x,
                weight=weight,
                target=target,
                lses=lses,
                dloss=(dloss * scale),
                chunk_size=ctx.chunk_size,
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
    chunk_size: int | None = None,
) -> torch.Tensor:
    assert reduction in ("mean", "sum")
    return LinearCrossEntropy.apply(
        x,
        weight,
        target,
        ignore_index,
        reduction,
        chunk_size,
    )
