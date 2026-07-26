import torch
import triton
from typing import NamedTuple
from einops import rearrange, reduce
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from quack.autotuner import autotune, AutotuneConfig
from quack.cross_entropy import cross_entropy as quack_cross_entropy
from quack.rms_final_reduce import rms_final_reduce as quack_rms_final_reduce

from coda.kernels.refs.gpt import rope as rope_interleaved
# `gpt2` is more optimized and less precise
from coda.kernels.refs import gpt2 as kernels_torch
from coda.kernels.gens import gpt as kernels_rapier
from coda.kernels.tests import gpt as kernels_rapier_test
from coda.kernels.benchmarks import trainstation_utils


_KERNELS = {
    "torch": kernels_torch,
    "rapier": kernels_rapier,
    "rapier-test": kernels_rapier_test,
}


@torch.compile(fullgraph=True, dynamic=False)
def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logits_tgt: torch.Tensor,
    logits_lse: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, D = logits.shape
    assert D % block_size == 0
    assert logits.shape == (B, T, D)
    assert targets.shape == (B, T)
    assert logits_tgt.shape == (B, T)
    assert logits_lse.shape == (B, T, triton.cdiv(D, block_size))
    logits_lse_reduced = torch.logsumexp(logits_lse, dim=-1, keepdim=True)
    loss = rearrange(logits_lse_reduced, "b t 1 -> b t") - logits_tgt

    grad = (logits - logits_lse_reduced).exp()
    grad = rearrange(grad, "b t d -> (b t) d", b=B, t=T, d=D)
    targets = rearrange(targets, "b t -> (b t)", b=B, t=T)
    grad[torch.arange(grad.shape[0], device=grad.device), targets] = (
        grad[torch.arange(grad.shape[0], device=grad.device), targets] - 1
    )
    grad = rearrange(grad, "(b t) d -> b t d", b=B, t=T, d=D)
    zdz = reduce(logits * grad, "b t d -> b t", "sum")
    return loss, grad, zdz


@torch.compile(fullgraph=True, dynamic=False)
def _cross_entropy_forward_torch(
    logits_tgt: torch.Tensor,
    logits_lse: torch.Tensor,
) -> torch.Tensor:
    M, _ = logits_lse.shape
    assert logits_tgt.shape == (M,)
    logits_lse_reduced = torch.logsumexp(logits_lse, dim=-1)
    loss = logits_lse_reduced - logits_tgt
    return loss.mean()


def cross_entropy_forward(
    logits_tgt: torch.Tensor,
    logits_lse: torch.Tensor,
    targets: torch.Tensor,
    use_quack: bool,
) -> torch.Tensor:
    if use_quack:
        return quack_cross_entropy(
            x=logits_tgt,
            target=targets,
            lse_partial=logits_lse,
            reduction="mean",
        )
    else:
        return _cross_entropy_forward_torch(
            logits_tgt=logits_tgt,
            logits_lse=logits_lse,
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


def gemm_residual_rmsnorm_gemm_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_n: torch.Tensor,
    block_size_norm: int,
    block_size_loss: int | None,
    cos_sin: torch.Tensor | None,
    targets: torch.Tensor | None,
    eps: float,
    epilogue: str | None,
    backend: str,
    use_quack: bool,
) -> tuple[torch.Tensor,
           torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
           torch.Tensor,
           torch.Tensor]:

    B, T, D0 = y.shape
    D1, D2 = w_b.shape
    assert x.shape == (B, T, D1)
    assert y.shape == (B, T, D0)
    assert w_a.shape == (D0, D1)
    assert w_b.shape == (D1, D2)
    assert w_n.shape == (D1,)
    x = rearrange(x, "b t d -> (b t) d", b=B, t=T, d=D1)
    y = rearrange(y, "b t d -> (b t) d", b=B, t=T, d=D0)

    x_out, s, h = _KERNELS[backend].gemm_residual_partial_rmsnorm(
        A=y,
        B=w_a,
        C=x,
        W=w_n,
        block_size=block_size_norm,
    )

    rstd_out = compute_rstd(
        s=s,
        eps=eps,
        use_quack=use_quack,
    )

    if epilogue is None:
        y_out = None
        z_out = _KERNELS[backend].gemm_rmsnorm(
            A=h,
            B=w_b,
            R=rstd_out,
        )

    elif epilogue == "swiglu":
        assert D2 % 2 == 0
        z_out, y_out = _KERNELS[backend].gemm_rmsnorm_swiglu(
            A=h,
            B=w_b,
            R=rstd_out,
        )
        y_out = rearrange(y_out, "(b t) d -> b t d", b=B, t=T, d=triton.cdiv(D2, 2))

    elif epilogue == "rope":
        assert cos_sin is not None
        z_out, y_out = _KERNELS[backend].gemm_rmsnorm_rope(
            A=h,
            B=w_b,
            R=rstd_out,
            cos_sin=cos_sin,
        )
        y_out = rearrange(y_out, "(b t) d -> b t d", b=B, t=T, d=D2)

    elif epilogue == "cross-entropy":
        assert block_size_loss is not None
        assert D2 % block_size_loss == 0
        assert targets is not None
        assert targets.shape == (B, T)
        targets = rearrange(targets, "b t -> (b t)", b=B, t=T)
        logits, logits_tgt, logits_lse = _KERNELS[backend].gemm_rmsnorm_partial_cross_entropy(
            A=h,
            B=w_b,
            R=rstd_out,
            targets=targets,
            block_size=block_size_loss,
        )
        logits_tgt = rearrange(logits_tgt, "(b t) -> b t", b=B, t=T)
        logits_lse = rearrange(logits_lse, "(b t) nb -> b t nb", b=B, t=T, nb=triton.cdiv(D2, block_size_loss))

        z_out = logits
        y_out = (logits_tgt, logits_lse)

    x_out = rearrange(x_out, "(b t) d -> b t d", b=B, t=T, d=D1)
    z_out = rearrange(z_out, "(b t) d -> b t d", b=B, t=T, d=D2)
    # s = rearrange(s, "(b t) nb -> b t nb", b=B, t=T, nb=triton.cdiv(D1, block_size))
    rstd_out = rearrange(rstd_out, "(b t) -> b t", b=B, t=T)
    return x_out, y_out, z_out, rstd_out


def _gemm_residual_rmsnorm_gemm_fwd_tunable(
    x: torch.Tensor,
    y: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_n: torch.Tensor,
    cos_sin: torch.Tensor | None,
    targets: torch.Tensor | None,
    eps: float,
    epilogue: str | None,
    backend: str,
    config: BlockSizeConfig2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_out, y_out, z_out, rstd_out = gemm_residual_rmsnorm_gemm_fwd(
        x=x,
        y=y,
        w_a=w_a,
        w_b=w_b,
        w_n=w_n,
        block_size_norm=config.block_size0,
        block_size_loss=config.block_size1,
        cos_sin=cos_sin,
        targets=targets,
        eps=eps,
        epilogue=epilogue,
        backend=backend,
        use_quack=config.use_quack0,
    )
    if epilogue == "cross-entropy":
        logits_tgt, logits_lse = y_out
        B, T, NB = logits_lse.shape
        loss = cross_entropy_forward(
            logits_tgt=rearrange(logits_tgt, "b t -> (b t)", b=B, t=T),
            logits_lse=rearrange(logits_lse, "b t nb -> (b t) nb", b=B, t=T, nb=NB),
            targets=rearrange(targets, "b t -> (b t)", b=B, t=T),
            use_quack=config.use_quack1,
        )
        return x_out, loss, z_out, rstd_out
    else:
        return x_out, y_out, z_out, rstd_out


_gemm_residual_rmsnorm_gemm_fwd_tunable_compiled = torch.compile(
    _gemm_residual_rmsnorm_gemm_fwd_tunable,
    fullgraph=True,
    dynamic=False,
)


@autotune(
    configs=BlockSizeConfig2Options,
    key=["epilogue", "backend", "use_compile"],
    cache_results=False,
)
def gemm_residual_rmsnorm_gemm_fwd_tunable(
    x: torch.Tensor,
    y: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_n: torch.Tensor,
    cos_sin: torch.Tensor | None,
    targets: torch.Tensor | None,
    eps: float,
    epilogue: str | None,
    backend: str,
    use_compile: bool,
    config: BlockSizeConfig2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _gemm_residual_rmsnorm_gemm_fwd_tunable
    else:
        fn = _gemm_residual_rmsnorm_gemm_fwd_tunable_compiled
    return fn(
        x=x,
        y=y,
        w_a=w_a,
        w_b=w_b,
        w_n=w_n,
        cos_sin=cos_sin,
        targets=targets,
        eps=eps,
        epilogue=epilogue,
        backend=backend,
        config=config,
    )


def gemm_residual_rmsnorm_gemm_bwd(
    x: torch.Tensor,
    z: torch.Tensor,
    dx: torch.Tensor,
    dz: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_n: torch.Tensor,
    rstd: torch.Tensor,
    zdz_prev: torch.Tensor,
    block_size_prev: int | None,
    block_size_curr: int | None,
    block_size_norm: int,
    epilogue: str | None,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:

    B, T, D2 = dz.shape
    D0, D1 = w_a.shape
    assert (B * T) % block_size_norm == 0
    assert x.shape == (B, T, D1)
    assert dz.shape == (B, T, D2)
    assert w_a.shape == (D0, D1)
    assert w_b.shape == (D1, D2)
    assert w_n.shape == (D1,)
    assert rstd.shape == (B, T)
    assert zdz_prev.shape == (B, T)

    x = rearrange(x, "b t d -> (b t) d", b=B, t=T, d=D1)
    dx = rearrange(dx, "b t d -> (b t) d", b=B, t=T, d=D1)
    dz = rearrange(dz, "b t d -> (b t) d", b=B, t=T, d=D2)
    rstd = rearrange(rstd, "b t -> (b t)", b=B, t=T)
    if block_size_prev is None:
        assert epilogue is not None
        zdz_prev = rearrange(zdz_prev, "b t -> (b t)", b=B, t=T) / D1
    else:
        assert epilogue is None
        # we scale by `block_size_prev` because the partial `zdz`
        # technically should be sum, but we implement it via mean
        zdz_prev = rearrange(zdz_prev, "b t -> (b t)", b=B, t=T) * (block_size_prev / D1)

    # kernel 1
    dx_out, x_out, dwn = _KERNELS[backend].gemm_residual_partial_rmsnorm_bwd(
        A=dz,
        B=w_b,
        C=x,
        W=w_n,
        R=rstd,
        ZdZ=zdz_prev,
        O=dx,
        block_size=block_size_norm,
    )
    dwn = reduce(dwn, "d nb -> d", "sum", d=D1, nb=triton.cdiv(B * T, block_size_norm))

    # kernel 2
    dw_b = torch.mm(x_out.T, dz)

    # kernel 3
    if epilogue is None:
        assert z.shape == (B, T, D0)
        assert block_size_curr is None
        z = rearrange(z, "b t d -> (b t) d", b=B, t=T, d=D0)
        dz_out = torch.mm(dx_out, w_a.T)
        dz_out = rearrange(dz_out, "(b t) d -> b t d", b=B, t=T, d=D0)
        zdz_out = None
        y = z

    elif epilogue == "swiglu":
        assert z.shape == (B, T, D0 * 2)
        assert block_size_curr is not None
        assert D0 % block_size_curr == 0
        z = rearrange(z, "b t d -> (b t) d", b=B, t=T, d=D0 * 2)
        dz_out, zdz_out, y = _KERNELS[backend].gemm_partial_swiglu_bwd(
            A=dx_out,
            B=w_a,
            Z=z,
            block_size=block_size_curr,
        )

        dz_out = rearrange(dz_out, "(b t) d -> b t d", b=B, t=T, d=D0 * 2)
        # int(D0 * 2 / block_size_curr / 2): an extra `/ 2` due to summing `dU, dG`
        zdz_out = reduce(zdz_out, "(b t) nb -> b t", "sum", b=B, t=T, nb=triton.cdiv(D0, block_size_curr))

    else:
        raise NotImplementedError

    # kernel 4
    dw_a = torch.mm(y.T, dx_out)

    dx_out = rearrange(dx_out, "(b t) d -> b t d", b=B, t=T, d=D1)
    return dx_out, dz_out, dw_a, dw_b, dwn, zdz_out


def _layer_post_forward_tunable(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    transpose: bool,
    backend: str,
    config: BlockSizeConfig3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if transpose:
        w0 = w0.mT
        w1 = w1.mT
        w2 = w2.mT
        w3 = w3.mT
    x1, y1, z1, rstd1 = gemm_residual_rmsnorm_gemm_fwd(
        x=x0,
        y=y0,
        w_a=w0,
        w_b=w1,
        w_n=wn0,
        block_size_norm=config.block_size0,
        block_size_loss=None,
        cos_sin=None,
        targets=None,
        eps=eps,
        epilogue="swiglu",
        backend=backend,
        use_quack=config.use_quack0,
    )
    x2, (logits_tgt, logits_lse), logits, rstd2 = gemm_residual_rmsnorm_gemm_fwd(
        x=x1,
        y=y1,
        w_a=w2,
        w_b=w3,
        w_n=wn1,
        block_size_norm=config.block_size1,
        block_size_loss=config.block_size2,
        cos_sin=None,
        targets=targets,
        eps=eps,
        epilogue="cross-entropy",
        backend=backend,
        use_quack=config.use_quack1,
    )
    loss, dlogits, zdz2 = cross_entropy(
        logits=logits,
        targets=targets,
        logits_tgt=logits_tgt,
        logits_lse=logits_lse,
        block_size=config.block_size1,
        # use_quack=config.use_quack2,
    )
    assert isinstance(y1, torch.Tensor)
    return x1, z1, rstd1, x2, rstd2, loss, dlogits, zdz2


_layer_post_forward_tunable_compiled = torch.compile(
    _layer_post_forward_tunable,
    fullgraph=True,
    dynamic=False,
)


@autotune(
    configs=BlockSizeConfig3Options,
    key=["transpose", "backend", "use_compile"],
    cache_results=False,
)
def layer_post_forward_tunable(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    transpose: bool,
    backend: str,
    use_compile: bool,
    config: BlockSizeConfig3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _layer_post_forward_tunable
    else:
        fn = _layer_post_forward_tunable_compiled
    return fn(
        x0=x0,
        y0=y0,
        w0=w0,
        w1=w1,
        w2=w2,
        w3=w3,
        wn0=wn0,
        wn1=wn1,
        targets=targets,
        eps=eps,
        transpose=transpose,
        backend=backend,
        config=config,
    )


def _layer_post_backward_tunable(
    dloss: torch.Tensor,
    dlogits: torch.Tensor,
    zdz2: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    y0: torch.Tensor,
    z1: torch.Tensor,
    rstd1: torch.Tensor,
    rstd2: torch.Tensor,
    transpose: bool,
    backend: str,
    config: BlockSizeConfig3PostBwd,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if transpose:
        w0 = w0.mT
        w1 = w1.mT
        w2 = w2.mT
        w3 = w3.mT
    dx1, dz1, dw2, dw3, dwn1, zdz1 = gemm_residual_rmsnorm_gemm_bwd(
        x=x2,
        z=z1,
        dx=torch.zeros_like(x2),
        dz=dlogits * rearrange(dloss, "b t -> b t 1"),
        w_a=w2,
        w_b=w3,
        w_n=wn1,
        rstd=rstd2,
        zdz_prev=zdz2 * dloss,
        block_size_prev=None,
        block_size_curr=config.block_size0,
        block_size_norm=config.block_size2,
        epilogue="swiglu",
        backend=backend,
    )
    dx0, dz0, dw0, dw1, dwn0, _ = gemm_residual_rmsnorm_gemm_bwd(
        x=x1,
        z=y0,
        dx=dx1,
        dz=dz1,
        w_a=w0,
        w_b=w1,
        w_n=wn0,
        rstd=rstd1,
        zdz_prev=zdz1,
        block_size_prev=config.block_size0,
        block_size_curr=None,
        block_size_norm=config.block_size1,
        epilogue=None,
        backend=backend,
    )
    dy0 = dz0
    if transpose:
        dw0 = dw0.mT
        dw1 = dw1.mT
        dw2 = dw2.mT
        dw3 = dw3.mT
    return dx0, dy0, dw0, dw1, dw2, dw3, dwn0, dwn1


_layer_post_backward_tunable_compiled = torch.compile(
    _layer_post_backward_tunable,
    fullgraph=True,
    dynamic=False,
)


@autotune(
    configs=BlockSizeConfig3PostBwdOptions,
    key=["transpose", "backend", "use_compile"],
    cache_results=False,
)
def layer_post_backward_tunable(
    dloss: torch.Tensor,
    dlogits: torch.Tensor,
    zdz2: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    y0: torch.Tensor,
    z1: torch.Tensor,
    rstd1: torch.Tensor,
    rstd2: torch.Tensor,
    transpose: bool,
    backend: str,
    use_compile: bool,
    config: BlockSizeConfig3PostBwd,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _layer_post_backward_tunable
    else:
        fn = _layer_post_backward_tunable_compiled
    return fn(
        dloss=dloss,
        dlogits=dlogits,
        zdz2=zdz2,
        w0=w0,
        w1=w1,
        w2=w2,
        w3=w3,
        wn0=wn0,
        wn1=wn1,
        x1=x1,
        x2=x2,
        y0=y0,
        z1=z1,
        rstd1=rstd1,
        rstd2=rstd2,
        transpose=transpose,
        backend=backend,
        config=config,
    )


class LayerPost(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x0: torch.Tensor,
        y0: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        wn0: torch.Tensor,
        wn1: torch.Tensor,
        targets: torch.Tensor,
        eps: float,
        transpose: bool,
        backend: str,
        use_compile: bool,
    ) -> torch.Tensor:

        x1, z1, rstd1, x2, rstd2, loss, dlogits, zdz2 = layer_post_forward_tunable(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            targets=targets,
            eps=eps,
            transpose=transpose,
            backend=backend,
            use_compile=use_compile,
        )

        ctx.save_for_backward(w0, w1, w2, w3, wn0, wn1, x1, x2, y0, z1, rstd1, rstd2, dlogits, zdz2)
        ctx.transpose = transpose
        ctx.backend = backend
        ctx.use_compile = use_compile
        return loss

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dloss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None]:
        w0, w1, w2, w3, wn0, wn1, x1, x2, y0, z1, rstd1, rstd2, dlogits, zdz2 = ctx.saved_tensors

        dx0, dy0, dw0, dw1, dw2, dw3, dwn0, dwn1 = layer_post_backward_tunable(
            dloss=dloss,
            dlogits=dlogits,
            zdz2=zdz2,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            x1=x1,
            x2=x2,
            y0=y0,
            z1=z1,
            rstd1=rstd1,
            rstd2=rstd2,
            transpose=ctx.transpose,
            backend=ctx.backend,
            use_compile=ctx.use_compile,
        )

        return (
            dx0,
            dy0,
            dw0,
            dw1,
            dw2,
            dw3,
            dwn0,
            dwn1,
            None,  # targets
            None,  # eps
            None,  # transpose
            None,  # backend
            None,  # use_compile
        )


def _layer_forward_tunable(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    transpose: bool,
    backend: str,
    config: BlockSizeConfig2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if transpose:
        w0 = w0.mT
        w1 = w1.mT
        w2 = w2.mT
        w3 = w3.mT
    x1, y1, z1, rstd1 = gemm_residual_rmsnorm_gemm_fwd(
        x=x0,
        y=y0,
        w_a=w0,
        w_b=w1,
        w_n=wn0,
        block_size_norm=config.block_size0,
        block_size_loss=None,
        cos_sin=None,
        targets=None,
        eps=eps,
        epilogue="swiglu",
        backend=backend,
        use_quack=config.use_quack0,
    )
    x2, y2, z2, rstd2 = gemm_residual_rmsnorm_gemm_fwd(
        x=x1,
        y=y1,
        w_a=w2,
        w_b=w3,
        w_n=wn1,
        block_size_norm=config.block_size1,
        block_size_loss=None,
        cos_sin=cos_sin,
        targets=None,
        eps=eps,
        epilogue="rope",
        backend=backend,
        use_quack=config.use_quack1,
    )
    assert isinstance(y1, torch.Tensor)
    assert isinstance(y2, torch.Tensor)
    return x1, z1, rstd1, x2, y2, z2, rstd2


_layer_forward_tunable_compiled = torch.compile(
    _layer_forward_tunable,
    fullgraph=True,
    dynamic=False,
)


@autotune(
    configs=BlockSizeConfig2Options,
    key=["transpose", "backend", "use_compile"],
    cache_results=False,
)
def layer_forward_tunable(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    transpose: bool,
    backend: str,
    use_compile: bool,
    config: BlockSizeConfig2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _layer_forward_tunable
    else:
        fn = _layer_forward_tunable_compiled
    return fn(
        x0=x0,
        y0=y0,
        w0=w0,
        w1=w1,
        w2=w2,
        w3=w3,
        wn0=wn0,
        wn1=wn1,
        cos_sin=cos_sin,
        eps=eps,
        transpose=transpose,
        backend=backend,
        config=config,
    )


def _layer_backward_tunable(
    dx2: torch.Tensor,
    dy2: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    y0: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    rstd1: torch.Tensor,
    rstd2: torch.Tensor,
    cos_sin: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    transpose: bool,
    backend: str,
    config: BlockSizeConfig3Bwd,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if transpose:
        w0 = w0.mT
        w1 = w1.mT
        w2 = w2.mT
        w3 = w3.mT
    dz2, zdz2 = attn_bwd_rope_patch(
        z=z2,
        dy=dy2,
        cos_sin=cos_sin,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        head_dim=head_dim,
        use_quack=config.use_quack,
    )
    dx1, dz1, dw2, dw3, dwn1, zdz1 = gemm_residual_rmsnorm_gemm_bwd(
        x=x2,
        z=z1,
        dx=dx2,
        dz=dz2,
        w_a=w2,
        w_b=w3,
        w_n=wn1,
        rstd=rstd2,
        zdz_prev=zdz2,
        block_size_prev=None,
        block_size_curr=config.block_size0,
        block_size_norm=config.block_size2,
        epilogue="swiglu",
        backend=backend,
    )
    dx0, dz0, dw0, dw1, dwn0, _ = gemm_residual_rmsnorm_gemm_bwd(
        x=x1,
        z=y0,
        dx=dx1,
        dz=dz1,
        w_a=w0,
        w_b=w1,
        w_n=wn0,
        rstd=rstd1,
        zdz_prev=zdz1,
        block_size_prev=config.block_size0,
        block_size_curr=None,
        block_size_norm=config.block_size1,
        epilogue=None,
        backend=backend,
    )
    dy0 = dz0
    if transpose:
        dw0 = dw0.mT
        dw1 = dw1.mT
        dw2 = dw2.mT
        dw3 = dw3.mT
    return dx0, dy0, dw0, dw1, dw2, dw3, dwn0, dwn1


_layer_backward_tunable_compiled = torch.compile(
    _layer_backward_tunable,
    fullgraph=True,
    dynamic=False,
)


@autotune(
    configs=BlockSizeConfig3BwdOptions,
    key=["transpose", "backend", "use_compile"],
    cache_results=False,
)
def layer_backward_tunable(
    dx2: torch.Tensor,
    dy2: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    y0: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    rstd1: torch.Tensor,
    rstd2: torch.Tensor,
    cos_sin: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    transpose: bool,
    backend: str,
    use_compile: bool,
    config: BlockSizeConfig3Bwd,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _layer_backward_tunable
    else:
        fn = _layer_backward_tunable_compiled
    return fn(
        dx2=dx2,
        dy2=dy2,
        w0=w0,
        w1=w1,
        w2=w2,
        w3=w3,
        wn0=wn0,
        wn1=wn1,
        x1=x1,
        x2=x2,
        y0=y0,
        z1=z1,
        z2=z2,
        rstd1=rstd1,
        rstd2=rstd2,
        cos_sin=cos_sin,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        head_dim=head_dim,
        transpose=transpose,
        backend=backend,
        config=config,
    )


class Layer(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x0: torch.Tensor,
        y0: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        wn0: torch.Tensor,
        wn1: torch.Tensor,
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

        x1, z1, rstd1, x2, y2, z2, rstd2 = layer_forward_tunable(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            cos_sin=cos_sin,
            eps=eps,
            transpose=transpose,
            backend=backend,
            use_compile=use_compile,
        )

        ctx.save_for_backward(w0, w1, w2, w3, wn0, wn1, x1, x2, y0, z1, z2, rstd1, rstd2, cos_sin, cos, sin)
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.transpose = transpose
        ctx.backend = backend
        ctx.use_compile = use_compile
        return x2, y2

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx2: torch.Tensor,
        dy2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None, None]:
        w0, w1, w2, w3, wn0, wn1, x1, x2, y0, z1, z2, rstd1, rstd2, cos_sin, cos, sin = ctx.saved_tensors

        dx0, dy0, dw0, dw1, dw2, dw3, dwn0, dwn1 = layer_backward_tunable(
            dx2=dx2,
            dy2=dy2,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            x1=x1,
            x2=x2,
            y0=y0,
            z1=z1,
            z2=z2,
            rstd1=rstd1,
            rstd2=rstd2,
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
            dx0,
            dy0,
            dw0,
            dw1,
            dw2,
            dw3,
            dwn0,
            dwn1,
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
