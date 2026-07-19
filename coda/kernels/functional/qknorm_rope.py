import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.elementwise.functional import qknorm_rope_fwd, qknorm_rope_bwd
from coda.core.gemm.functional import _sqsum_num_segments, gemm, gemm_qkv_sqsum


class LinearQKNormRoPE(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        gamma: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None]:


def linear_qknorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return LinearQKNormRoPE.apply(
        x,
        weight,
        gamma,
        positions,
        frequencies,
        num_heads_q,
        num_heads_k,
        head_dim,
        eps,
    )
