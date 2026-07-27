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
