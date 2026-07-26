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


class LayerPre(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        wn: torch.Tensor,
        cos_sin: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        head_dim: int,
        eps: float,
        transpose: bool,
        backend: str,
        use_compile: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        y, z, rstd = layer_pre_forward_tunable(
            x=x,
            w=w,
            wn=wn,
            cos_sin=cos_sin,
            eps=eps,
            transpose=transpose,
            backend=backend,
            use_compile=use_compile,
        )

        ctx.save_for_backward(w, wn, x, z, rstd, cos_sin, cos, sin)
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.transpose = transpose
        ctx.backend = backend
        ctx.use_compile = use_compile
        return x, y

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None, None]:
        w, wn, x, z, rstd, cos_sin, cos, sin = ctx.saved_tensors

        dx_out, dw, dwn = layer_pre_backward_tunable(
            dx=dx,
            dy=dy,
            w=w,
            wn=wn,
            x=x,
            z=z,
            rstd=rstd,
            cos_sin=cos_sin,
            cos=cos,
            sin=sin,
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
            transpose=ctx.transpose,
            backend=ctx.backend,
            use_compile=ctx.use_compile,
        )

        return (
            dx_out,
            dw,
            dwn,
            None,  # cos_sin
            None,  # cos
            None,  # sin
            None,  # num_heads
            None,  # head_dim
            None,  # eps
            None,  # transpose
            None,  # backend
            None,  # use_compile
        )
