import torch
from quack.cross_entropy import cross_entropy_fwd

from coda.core.gemm.functional import gemm, gemm_ce_grad, gemm_linear_ce


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
