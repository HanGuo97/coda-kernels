import torch
from einops import rearrange
from quack.rms_final_reduce import rms_final_reduce as quack_rms_final_reduce
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from coda.core.elementwise.functional import (
    cross_entropy_fwd_bwd,
    rope_bwd_zdz,
)
from coda.core.gemm.functional import (
    gemm as coda_gemm,
    gemm_scalar_scale,
    gemm_lse,
    gemm_residual_partial_rmsnorm,
    gemm_residual_partial_rmsnorm_bwd,
    gemm_rmsnorm_rope,
    gemm_rmsnorm_swiglu,
    gemm_swiglu_bwd_zdz,
)

USE_CODA_GEMM = False
ALLOW_INPLACE_GRAD_OUTPUT = False


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if USE_CODA_GEMM:
        return coda_gemm(A=A, B=B)
    else:
        return torch.matmul(A, B)


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
    qkv: torch.Tensor,
    rstd: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # y = R z and dz = R^T dy, hence
    # sum(z * dz) = z^T R^T dy = (R z)^T dy = sum(y * dy).
    dz, zdz = rope_bwd_zdz(
        y=qkv,
        dy=dy,
        pos=positions,
        freq=frequencies,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=1.0 / x.shape[1],
    )
    dx_out = dx if ALLOW_INPLACE_GRAD_OUTPUT else dx.clone()
    dx_out, dwn, x_out = gemm_residual_partial_rmsnorm_bwd(
        A=dz,
        # TODO: the function itself also transposes `B`
        # maybe we can directly pass in `BT` there
        B=w.mT,
        W=wn,
        dX=dx_out,
        pre=x,
        ZdZ=zdz,
        rstd=rstd,
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
        head_dim: int,
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
            frequencies[:head_dim],
        )
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        return x, qkv

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None]:
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
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
        )
        return (
            dx_out,
            dw,
            dwn,
            None,  # positions
            None,  # frequencies
            None,  # num_heads
            None,  # head_dim
            None,  # eps
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


def block_pre(
    x: torch.Tensor,
    w: torch.Tensor,
    wn: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, _ = x.shape
    dim1, dim0 = w.shape
    assert x.shape == (batch, length, dim0)
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert w.shape == (dim1, dim0)
    assert w.dtype == x.dtype
    assert wn.shape == (dim0,)
    assert wn.dtype == x.dtype
    assert positions.shape == (batch * length,)
    assert positions.dtype in (torch.float32, torch.int32)
    assert frequencies.shape == (dim1,)
    assert frequencies.dtype == torch.float32
    residual, qkv = BlockPre.apply(
        rearrange(x, "b t d -> (b t) d"),
        w,
        wn,
        positions,
        frequencies,
        num_heads,
        head_dim,
        eps,
    )
    return (
        rearrange(residual, "(b t) d -> b t d", b=batch),
        rearrange(qkv, "(b t) d -> b t d", b=batch),
    )


def block_post(
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
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    batch, length, _ = x0.shape
    dim0, _ = w0.shape
    dim1, _ = w1.shape
    dim3, _ = w3.shape
    assert x0.shape == (batch, length, dim0)
    assert x0.dtype in (torch.float16, torch.bfloat16)
    assert y0.shape == (batch, length, dim0)
    assert y0.dtype == x0.dtype
    assert w0.shape == (dim0, dim0)
    assert w0.dtype == x0.dtype
    assert w1.shape == (dim1, dim0)
    assert w1.dtype == x0.dtype
    assert w2.shape == (dim0, dim1 // 2)
    assert w2.dtype == x0.dtype
    assert w3.shape == (dim3, dim0)
    assert w3.dtype == x0.dtype
    assert wn0.shape == (dim0,)
    assert wn0.dtype == x0.dtype
    assert wn1.shape == (dim0,)
    assert wn1.dtype == x0.dtype
    assert targets.shape == (batch * length,)
    assert targets.dtype == torch.int32
    assert reduction in ("mean", "sum")
    return BlockPost.apply(
        rearrange(x0, "b t d -> (b t) d"),
        rearrange(y0, "b t d -> (b t) d"),
        w0,
        w1,
        w2,
        w3,
        wn0,
        wn1,
        targets,
        eps,
        ignore_index,
        reduction,
    )


def block(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, _ = x0.shape
    dim0, _ = w0.shape
    dim1, _ = w1.shape
    dim3, _ = w3.shape
    assert x0.shape == (batch, length, dim0)
    assert x0.dtype in (torch.float16, torch.bfloat16)
    assert y0.shape == (batch, length, dim0)
    assert y0.dtype == x0.dtype
    assert w0.shape == (dim0, dim0)
    assert w0.dtype == x0.dtype
    assert w1.shape == (dim1, dim0)
    assert w1.dtype == x0.dtype
    assert w2.shape == (dim0, dim1 // 2)
    assert w2.dtype == x0.dtype
    assert w3.shape == (dim3, dim0)
    assert w3.dtype == x0.dtype
    assert wn0.shape == (dim0,)
    assert wn0.dtype == x0.dtype
    assert wn1.shape == (dim0,)
    assert wn1.dtype == x0.dtype
    assert positions.shape == (batch * length,)
    assert positions.dtype in (torch.float32, torch.int32)
    assert frequencies.shape == (dim3,)
    assert frequencies.dtype == torch.float32
    residual, qkv = Block.apply(
        rearrange(x0, "b t d -> (b t) d"),
        rearrange(y0, "b t d -> (b t) d"),
        w0,
        w1,
        w2,
        w3,
        wn0,
        wn1,
        positions,
        frequencies,
        num_heads,
        head_dim,
        eps,
    )
    return (
        rearrange(residual, "(b t) d -> b t d", b=batch),
        rearrange(qkv, "(b t) d -> b t d", b=batch),
    )
