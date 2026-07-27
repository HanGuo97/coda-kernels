import torch
from quack.cross_entropy import cross_entropy_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.elementwise.functional import cross_entropy_fwd_bwd
from coda.core.gemm.functional import gemm, gemm_lse, gemm_lse_select_logits, gemm_scalar_scale


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
    use_cutedsl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_cutedsl:
        logits, lses = gemm_lse(x, weight.mT)
        losses, _ = cross_entropy_fwd_bwd(
            logits=logits,
            lses=lses,
            target=target,
            ignore_index=ignore_index,
        )
        return losses, logits
    else:
        return _forward_lse_torch(
            x=x,
            weight=weight,
            target=target,
            ignore_index=ignore_index,
        )


@torch.compile(dynamic=False, fullgraph=True)
def _forward_lse_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, lses = gemm_lse(x, weight.mT)
    ignore = (target == ignore_index)
    ignore_ = ignore[:, None].to(dtype=logits.dtype)
    # mask out, say, target = -100 which would crash gather/scatter
    safe_target = torch.where(ignore, 0, target)[:, None]
    target_logits = torch.gather(logits, dim=1, index=safe_target)
    target_logits = torch.squeeze(target_logits, dim=1)
    losses = torch.where(ignore, 0.0, lses - target_logits)
    # backward
    logits.sub_(lses[:, None])
    logits.exp_()
    logits.mul_(1.0 - ignore_)
    logits.scatter_add_(dim=1, index=safe_target, src=ignore_ - 1.0)
    return losses, logits


def _backward_dlogits(
    x: torch.Tensor,
    weight: torch.Tensor,
    dlogits: torch.Tensor,
    dloss: torch.Tensor,
    need_dx: bool,
    need_dweight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if need_dx:
        dx = gemm_scalar_scale(dlogits, weight, alpha=dloss)
    else:
        dx = None
    if need_dweight:
        dweight = gemm_scalar_scale(dlogits.mT, x, alpha=dloss)
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
    assert target.dtype == torch.int32
    return LinearCrossEntropy.apply(
        x,
        weight,
        target,
        ignore_index,
        reduction,
        fused_lse,
    )


def linear_cross_entropy_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    assert reduction in ("mean", "sum")
    assert target.dtype == torch.int32
    losses, _, _ = gemm_lse_select_logits(
        x,
        weight.mT,
        target=target,
        ignore_index=ignore_index,
        return_lse=False,
    )
    if reduction == "mean":
        scale = 1.0 / (target != ignore_index).sum().float()
        loss = losses.sum() * scale
    else:
        loss = losses.sum()
    return loss
