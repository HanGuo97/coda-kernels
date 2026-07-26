@torch.library.triton_op("rapier_ops::attn_bwd_rope_patch_quack", mutates_args={})
def _attn_bwd_rope_patch_quack(
    z: torch.Tensor,
    dy: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `zdz` is already reduced, and we rotates by -theta, hence `-sin`
    return trainstation_utils.rope_bwd_zdz(
        z=z,
        dy=dy,
        cos=cos,
        sin=-sin,
        num_heads=num_heads,
        head_dim=head_dim,
    )


@torch.compile(fullgraph=True, dynamic=False)
def _attn_bwd_rope_patch_torch(
    z: torch.Tensor,
    dy: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = dy.shape
    dy = rearrange(dy, "b t d -> (b t) d", b=B, t=T, d=D)
    dz = rope_interleaved(
        dy.to(dtype=torch.float32),
        cos_sin=cos_sin.to(dtype=torch.float32),
        backward=True,
    )
    dz = rearrange(dz, "(b t) d -> b t d", b=B, t=T, d=D)
    assert dz.dtype == torch.float32
    zdz = reduce(z.to(dtype=torch.float32) * dz, "b t d -> b t", "sum", b=B, t=T, d=D)
    assert zdz.dtype == torch.float32
    return dz.to(dtype=dy.dtype), zdz


def attn_bwd_rope_patch(
    z: torch.Tensor,
    dy: torch.Tensor,
    cos_sin: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    use_quack: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_quack:
        return _attn_bwd_rope_patch_quack(
            z=z,
            dy=dy,
            cos=cos,
            sin=sin,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    else:
        return _attn_bwd_rope_patch_torch(
            z=z,
            dy=dy,
            cos_sin=cos_sin,
        )


@torch.compile(fullgraph=True, dynamic=False)
def _compute_rstd_torch(s: torch.Tensor, eps: float) -> torch.Tensor:
    s = reduce(s, "... n -> ...", "mean")
    r = torch.rsqrt(s + eps)
    return r


def compute_rstd(s: torch.Tensor, eps: float, use_quack: bool) -> torch.Tensor:
    # `s` is partially (mean) reduced tensor squared
    # so we first reduce it globally before computing `rstd`
    if use_quack:
        assert s.ndim == 2
        n = s.shape[-1]
        return quack_rms_final_reduce(
            x=s,
            scale=1.0 / n,
            eps=eps,
        )
    else:
        return _compute_rstd_torch(
            s=s,
            eps=eps,
        )


def rmsnorm_gemm_rope_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    w_n: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    backend: str,
    use_quack: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, _ = x.shape
    D0, D1 = w.shape
    assert x.shape == (B, T, D0)
    assert w.shape == (D0, D1)
    assert w_n.shape == (D0,)
    x = rearrange(x, "b t d -> (b t) d", b=B, t=T, d=D0)

    h = x * rearrange(w_n, "d -> 1 d")
    rstd_out = compute_rstd(
        s=x ** 2,
        eps=eps,
        use_quack=use_quack,
    )
    z_out, y_out = _KERNELS[backend].gemm_rmsnorm_rope(
        A=h,
        B=w,
        R=rstd_out,
        cos_sin=cos_sin,
    )

    y_out = rearrange(y_out, "(b t) d -> b t d", b=B, t=T, d=D1)
    z_out = rearrange(z_out, "(b t) d -> b t d", b=B, t=T, d=D1)
    rstd_out = rearrange(rstd_out, "(b t) -> b t", b=B, t=T)
    return y_out, z_out, rstd_out


def rmsnorm_gemm_rope_bwd(
    x: torch.Tensor,
    dx: torch.Tensor,
    dz: torch.Tensor,
    w: torch.Tensor,
    w_n: torch.Tensor,
    rstd: torch.Tensor,
    zdz_prev: torch.Tensor,
    block_size: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    D0, D1 = w.shape
    B, T, D1 = dz.shape
    assert (B * T) % block_size == 0
    assert x.shape == (B, T, D0)
    assert dx.shape == (B, T, D0)
    assert dz.shape == (B, T, D1)
    assert w.shape == (D0, D1)
    assert w_n.shape == (D0,)
    assert rstd.shape == (B, T)
    assert zdz_prev.shape == (B, T)

    x = rearrange(x, "b t d -> (b t) d", b=B, t=T, d=D0)
    dx = rearrange(dx, "b t d -> (b t) d", b=B, t=T, d=D0)
    dz = rearrange(dz, "b t d -> (b t) d", b=B, t=T, d=D1)
    rstd = rearrange(rstd, "b t -> (b t)", b=B, t=T)
    zdz_prev = rearrange(zdz_prev, "b t -> (b t)", b=B, t=T) / D0

    # kernel 1
    dx_out, x_out, dwn = _KERNELS[backend].gemm_residual_partial_rmsnorm_bwd(
        A=dz,
        B=w,
        C=x,
        W=w_n,
        R=rstd,
        ZdZ=zdz_prev,
        O=dx,
        block_size=block_size,
    )
    dwn = reduce(dwn, "d nb -> d", "sum", d=D0, nb=triton.cdiv(B * T, block_size))

    # kernel 2
    dw = torch.mm(x_out.T, dz)

    dx_out = rearrange(dx_out, "(b t) d -> b t d", b=B, t=T, d=D0)
    return dx_out, dw, dwn


def _layer_pre_forward_tunable(
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    transpose: bool,
    backend: str,
    config: BlockSizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if transpose:
        w = w.mT
    y, z, rstd = rmsnorm_gemm_rope_fwd(
        x=x,
        w=w,
        w_n=wn,
        cos_sin=cos_sin,
        eps=eps,
        backend=backend,
        use_quack=config.use_quack,
    )
    return y, z, rstd


_layer_pre_forward_tunable_compiled = torch.compile(
    _layer_pre_forward_tunable,
    fullgraph=True,
    dynamic=False,
)


@autotune(
    configs=BlockSizeConfigOptions,
    key=["transpose", "backend", "use_compile"],
    cache_results=False,
)
def layer_pre_forward_tunable(
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    transpose: bool,
    backend: str,
    use_compile: bool,
    config: BlockSizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _layer_pre_forward_tunable
    else:
        fn = _layer_pre_forward_tunable_compiled
    return fn(
        x=x,
        w=w,
        wn=wn,
        cos_sin=cos_sin,
        eps=eps,
        transpose=transpose,
        backend=backend,
        config=config,
    )


def _layer_pre_backward_tunable(
    dx: torch.Tensor,
    dy: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    rstd: torch.Tensor,
    cos_sin: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    transpose: bool,
    backend: str,
    config: BlockSizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if transpose:
        w = w.mT
    dz, zdz = attn_bwd_rope_patch(
        z=z,
        dy=dy,
        cos_sin=cos_sin,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        head_dim=head_dim,
        use_quack=config.use_quack,
    )
    dx_out, dw, dwn = rmsnorm_gemm_rope_bwd(
        x=x,
        dx=dx,
        dz=dz,
        w=w,
        w_n=wn,
        rstd=rstd,
        zdz_prev=zdz,
        block_size=config.block_size,
        backend=backend,
    )
    if transpose:
        dw = dw.mT
    return dx_out, dw, dwn


_layer_pre_backward_tunable_compiled = torch.compile(
    _layer_pre_backward_tunable,
    fullgraph=True,
    dynamic=False,
)


@autotune(
    configs=BlockSizeConfigOptions,
    key=["transpose", "backend", "use_compile"],
    cache_results=False,
)
def layer_pre_backward_tunable(
    dx: torch.Tensor,
    dy: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    rstd: torch.Tensor,
    cos_sin: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    transpose: bool,
    backend: str,
    use_compile: bool,
    config: BlockSizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _layer_pre_backward_tunable
    else:
        fn = _layer_pre_backward_tunable_compiled
    return fn(
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
        num_heads=num_heads,
        head_dim=head_dim,
        transpose=transpose,
        backend=backend,
        config=config,
    )


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
