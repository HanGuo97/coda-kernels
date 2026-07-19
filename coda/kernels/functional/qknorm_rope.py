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
        num_heads_qk = num_heads_q + num_heads_k
        num_segments = _sqsum_num_segments(head_dim)
        size_q = head_dim * num_heads_q
        size_qk = head_dim * num_heads_qk
        size_ssq_qk = num_heads_qk * num_segments
        pre, ssq = gemm_qkv_sqsum(
            x,
            weight.mT,
            head_dim=head_dim,
            num_segments=num_segments,
        )
        out_qk = qknorm_rope_fwd(
            x=pre[:, :size_qk],
            ssq=ssq[:, :size_ssq_qk],
            gamma=gamma,
            pos=positions,
            freq=frequencies,
            head_dim=head_dim,
            num_heads_qk=num_heads_qk,
            num_segments=num_segments,
            eps=eps,
        )
        ctx.save_for_backward(x, weight, gamma, positions, frequencies, pre, ssq)
        ctx.num_heads_q = num_heads_q
        ctx.num_heads_k = num_heads_k
        ctx.num_segments = num_segments
        ctx.size_qk = size_qk
        ctx.size_ssq = size_ssq_qk
        ctx.head_dim = head_dim
        ctx.eps = eps
        return out_qk[:, :size_q], out_qk[:, size_q:], pre[:, size_qk:]

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None]:
        x, weight, gamma, positions, frequencies, pre, ssq = ctx.saved_tensors

        _, dgamma = qknorm_rope_bwd(
            dq=dq,
            dk=dk,
            x=pre,
            ssq=ssq,
            gamma=gamma,
            pos=positions,
            freq=frequencies,
            head_dim=ctx.head_dim,
            num_heads_q=ctx.num_heads_q,
            num_heads_k=ctx.num_heads_k,
            num_segments=ctx.num_segments,
            eps=ctx.eps,
            dx=grad_pre[:, :ctx.size_qk],
        )
        dx = gemm(grad_pre, weight)
        dweight = gemm(grad_pre.mT, x)
        return dx, dweight, dgamma, None, None, None, None, None, None


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
    assert gamma.shape == (head_dim,)
    assert gamma.dtype == x.dtype
    assert positions.shape == (x.shape[0],)
    assert positions.dtype == torch.int32
    assert frequencies.shape == (head_dim * (num_heads_q + num_heads_k) // 2,)
    assert frequencies.dtype == torch.float32
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
