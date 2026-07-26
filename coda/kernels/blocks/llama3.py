def block_pre_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert wn.dtype == x.dtype
    _, n = x.shape
    h = x * rearrange(wn, "d -> 1 d")
    rstd = quack_rms_final_reduce(
        x=x.float() ** 2,
        scale=1.0 / n,
        eps=eps,
    )
    qkv = gemm_rmsnorm_rope(
        A=h,
        B=w.mT,
        rstd=rstd,
        positions=positions,
        frequencies=frequencies,
    )
    return qkv, rstd


def block_pre_backward(
    dx: torch.Tensor,
    dy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    rstd: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # y = R z and dz = R^T dy, hence
    # sum(z * dz) = z^T R^T dy = (R z)^T dy = sum(y * dy).
    dz, zdz = rope_bwd_zdz(
        y=qkv,
        dy=dy,
        pos=positions,
        freq=frequencies,
    )
    dx_out, dwn, x_out = gemm_residual_partial_rmsnorm_bwd(
        A=dz,
        # TODO: the function itself also transposes `B`
        # maybe we can directly pass in `BT` there
        B=w.mT,
        pre=x,
        W=wn,
        rstd=rstd,
        ZdZ=zdz,
        dX=dx,
    )
    dw = gemm(x_out.mT, dz)
    return dx_out, dw, dwn


class BlockPre(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        wn: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
        num_heads: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv, rstd = block_pre_forward(
            x=x,
            w=w,
            wn=wn,
            positions=positions,
            frequencies=frequencies,
            eps=eps,
        )
        ctx.save_for_backward(
            x,
            w,
            wn,
            qkv,
            rstd,
            positions,
            frequencies,
        )
        return x, qkv

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        x, w, wn, qkv, rstd, positions, frequencies = ctx.saved_tensors
        dx_out, dw, dwn = block_pre_backward(
            dx=dx,
            dy=dy,
            x=x,
            w=w,
            wn=wn,
            qkv=qkv,
            rstd=rstd,
            positions=positions,
            frequencies=frequencies,
        )
        return (
            dx_out,
            dw,
            dwn,
            None,  # positions
            None,  # frequencies
            None,  # num_heads
            None,  # eps
        )
