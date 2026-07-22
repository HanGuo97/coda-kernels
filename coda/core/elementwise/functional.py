import torch
import cutlass
import cutlass.cute as cute

from quack.activation import dswiglu
from quack.autotuner import autotune, AutotuneConfig

from coda.core.ops.misc_utils import static_assert, ceil_div
from coda.core.gemm.gemm_interface import _kernel_op
from coda.core.elementwise.rope import qknorm_rope_, qknorm_rope_bwd_
from coda.core.elementwise.cross_entropy import cross_entropy_fwd_bwd_
from coda.core.elementwise.templates import ElementwiseConfig, _elementwise_op_tuned


_ELEMENTWISE_CONFIGS = tuple(
    ElementwiseConfig(
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    for thr_m, thr_n, val_m in (
        (4, 32, 4),
        (8, 64, 4),
        (16, 16, 4),
        (1, 128, 4),
        (8, 128, 4),
        (4, 256, 4),
        (2, 512, 4),
    )
)

_CE_ELEMENTWISE_CONFIGS = tuple(
    ElementwiseConfig(
        thr_m=thr_m,
        thr_n=thr_n,
        val_m=val_m,
    )
    for thr_m, thr_n, val_m in (
        (1, 512, 2),
        (4, 128, 1),
    )
)


def _prune_rope_configs(configs: list[AutotuneConfig], named_args: dict, **kwargs) -> list[AutotuneConfig]:
    kwargs = named_args | kwargs
    x = kwargs["x"]
    assert x.ndim == 2
    packed_cols = x.shape[1] // 2
    dtype_width = x.element_size() * 8
    vector_size = 128 // (2 * dtype_width)
    configs_pruned = [
        c for c in configs
        if packed_cols % (c.kwargs["config"].thr_n * vector_size) == 0
    ]

    # bwd only (fwd passes no dq)
    dq = kwargs.get("dq", None)
    interleaved = kwargs["interleaved"]
    if dq is not None and not interleaved:
        assert dq.ndim == 2
        packed_cols_dq = dq.shape[1] // 2
        configs_pruned = [
            c for c in configs_pruned
            if packed_cols_dq % (c.kwargs["config"].thr_n * vector_size) == 0
        ]

    return configs_pruned


@cute.jit
def _dswiglu_op(tX: cute.Tensor, tY: cute.Tensor, tZ: cute.Tensor) -> None:
    static_assert(tX.dtype == cute.Int32)
    static_assert(tZ.dtype == cute.Int32)
    static_assert(tY.dtype in (cute.Float16, cute.BFloat16))
    dtype = tY.dtype
    tX_pair = cute.recast_tensor(tX, dtype=dtype)
    tZ_pair = cute.recast_tensor(tZ, dtype=dtype)
    for i in cutlass.range_constexpr(cute.size(tY)):
        g = tX_pair[2 * i].to(dtype=cutlass.Float32)
        u = tX_pair[2 * i + 1].to(dtype=cutlass.Float32)
        dout = tY[i].to(dtype=cutlass.Float32)
        dg, du, _ = dswiglu(x=g, y=u, dout=dout)
        tZ_pair[2 * i] = dg.to(dtype=dtype)
        tZ_pair[2 * i + 1] = du.to(dtype=dtype)


@_kernel_op("coda::_dswiglu_backward", mutates_args=("Z",))
def _dswiglu_backward(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> None:
    return _elementwise_op_tuned(op=_dswiglu_op, X=X, Y=Y, Z=Z)


def dswiglu_backward(
    pre_act: torch.Tensor,
    grad_out: torch.Tensor,
    grad_pre: torch.Tensor | None = None,
) -> torch.Tensor:
    assert pre_act.dtype in (torch.bfloat16, torch.float16)
    assert grad_out.dtype == pre_act.dtype
    assert pre_act.is_contiguous()
    assert grad_out.is_contiguous()
    if grad_pre is None:
        grad_pre = torch.empty_like(pre_act)
    _dswiglu_backward(
        X=pre_act.view(dtype=torch.int32),
        Y=grad_out,
        Z=grad_pre.view(dtype=torch.int32),
    )
    return grad_pre


@autotune(
    configs=[AutotuneConfig(config=c) for c in _CE_ELEMENTWISE_CONFIGS],
    key=["ignore_index"],
    cache_results=False,
)
def _cross_entropy_fwd_bwd_tuned(
    logits: torch.Tensor,
    lses: torch.Tensor,
    target: torch.Tensor,
    losses: torch.Tensor,
    ignore_index: int,
    config: ElementwiseConfig | None,
) -> None:
    if config is None:
        config = ElementwiseConfig(thr_m=4, thr_n=32, val_m=4)

    cross_entropy_fwd_bwd_(
        logits=logits,
        lses=lses,
        target=target,
        losses=losses,
        ignore_index=ignore_index,
        thr_m=config.thr_m,
        thr_n=config.thr_n,
        val_m=config.val_m,
    )


@_kernel_op("coda::_cross_entropy_fwd_bwd", mutates_args=("logits", "losses"))
def _cross_entropy_fwd_bwd(
    logits: torch.Tensor,
    lses: torch.Tensor,
    target: torch.Tensor,
    losses: torch.Tensor,
    ignore_index: int,
) -> None:
    _cross_entropy_fwd_bwd_tuned(
        logits=logits,
        lses=lses,
        target=target,
        losses=losses,
        ignore_index=ignore_index,
    )


def cross_entropy_fwd_bwd(
    logits: torch.Tensor,
    lses: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    losses: torch.Tensor | None = None,
) -> torch.Tensor:
    if losses is None:
        # zero-init as the kernel never writes ignored rows' losses
        losses = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    _cross_entropy_fwd_bwd(
        logits=logits,
        lses=lses,
        target=target,
        losses=losses,
        ignore_index=ignore_index,
    )
    return losses


@autotune(
    configs=[AutotuneConfig(config=c) for c in _ELEMENTWISE_CONFIGS],
    key=["head_dim", "num_heads_q", "num_heads_k", "num_segments", "eps", "interleaved"],
    prune_configs_by={"early_config_prune": _prune_rope_configs},
    cache_results=False,
)
def _qknorm_rope_fwd_tuned(
    x: torch.Tensor,
    y: torch.Tensor,
    ssq: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_segments: int,
    eps: float,
    interleaved: bool,
    config: ElementwiseConfig | None,
) -> None:
    if config is None:
        config = ElementwiseConfig(thr_m=4, thr_n=32, val_m=4)

    qknorm_rope_(
        x=x,
        y=y,
        ssq=ssq,
        gamma_q=gamma_q,
        gamma_k=gamma_k,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_segments=num_segments,
        eps=eps,
        interleaved=interleaved,
        thr_m=config.thr_m,
        thr_n=config.thr_n,
        val_m=config.val_m,
    )


@_kernel_op("coda::_qknorm_rope_fwd", mutates_args=("y",))
def _qknorm_rope_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    ssq: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_segments: int,
    eps: float,
    interleaved: bool,
) -> None:
    _qknorm_rope_fwd_tuned(
        x=x,
        y=y,
        ssq=ssq,
        gamma_q=gamma_q,
        gamma_k=gamma_k,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_segments=num_segments,
        eps=eps,
        interleaved=interleaved,
    )


def qknorm_rope_fwd(
    x: torch.Tensor,
    ssq: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_segments: int,
    eps: float,
    interleaved: bool,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    if y is None:
        y = torch.empty_like(x)
    _qknorm_rope_fwd(
        x=x,
        y=y,
        ssq=ssq,
        gamma_q=gamma_q,
        gamma_k=gamma_k,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_segments=num_segments,
        eps=eps,
        interleaved=interleaved,
    )
    return y


@autotune(
    configs=[AutotuneConfig(config=c) for c in _ELEMENTWISE_CONFIGS],
    key=["head_dim", "num_heads_q", "num_heads_k", "num_segments", "eps", "interleaved"],
    prune_configs_by={"early_config_prune": _prune_rope_configs},
    cache_results=False,
)
def _qknorm_rope_bwd_tuned(
    dx: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor | None,
    dgamma_q: torch.Tensor,
    dgamma_k: torch.Tensor,
    x: torch.Tensor,
    ssq: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_segments: int,
    eps: float,
    interleaved: bool,
    config: ElementwiseConfig | None,
) -> None:
    if config is None:
        config = ElementwiseConfig(thr_m=4, thr_n=32, val_m=4)

    tile_m = config.thr_m * config.val_m
    num_m_tiles = ceil_div(x.shape[0], tile_m)
    num_heads_qk = num_heads_q + num_heads_k
    dgamma_partials = torch.empty(
        num_m_tiles,
        x.shape[1],
        dtype=torch.float32,
        device=x.device,
    )
    qknorm_rope_bwd_(
        dx=dx,
        dq=dq,
        dk=dk,
        dv=dv,
        dgamma=dgamma_partials,
        x=x,
        ssq=ssq,
        gamma_q=gamma_q,
        gamma_k=gamma_k,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_segments=num_segments,
        eps=eps,
        interleaved=interleaved,
        thr_m=config.thr_m,
        thr_n=config.thr_n,
        val_m=config.val_m,
    )
    if interleaved:
        raise NotImplementedError
    else:
        dgamma_partials = dgamma_partials.view(num_m_tiles, num_heads_qk, head_dim)
        torch.sum(
            dgamma_partials[:, :num_heads_q, :],
            dim=(0, 1),
            out=dgamma_q,
        )
        torch.sum(
            dgamma_partials[:, num_heads_q:, :],
            dim=(0, 1),
            out=dgamma_k,
        )


@_kernel_op("coda::_qknorm_rope_bwd", mutates_args=("dx", "dgamma_q", "dgamma_k"))
def _qknorm_rope_bwd(
    dx: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor | None,
    dgamma_q: torch.Tensor,
    dgamma_k: torch.Tensor,
    x: torch.Tensor,
    ssq: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_segments: int,
    eps: float,
    interleaved: bool,
) -> None:
    _qknorm_rope_bwd_tuned(
        dx=dx,
        dq=dq,
        dk=dk,
        dv=dv,
        dgamma_q=dgamma_q,
        dgamma_k=dgamma_k,
        x=x,
        ssq=ssq,
        gamma_q=gamma_q,
        gamma_k=gamma_k,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_segments=num_segments,
        eps=eps,
        interleaved=interleaved,
    )


def qknorm_rope_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    x: torch.Tensor,
    ssq: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    pos: torch.Tensor,
    freq: torch.Tensor,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_segments: int,
    eps: float,
    interleaved: bool,
    dx: torch.Tensor | None = None,
    dgamma_q: torch.Tensor | None = None,
    dgamma_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if interleaved:
        # x = [q, k, v]
        if dx is None:
            dx = torch.empty_like(x)
        kernel_dx = dx
        kernel_dv = dv
    else:
        # x = [q, k]
        size_qk = x.shape[1]
        if dx is None:
            dx = torch.empty(
                x.shape[0],
                size_qk + dv.shape[1],
                dtype=x.dtype,
                device=x.device,
            )
        dx[:, size_qk:].copy_(dv)
        kernel_dx = dx[:, :size_qk]
        kernel_dv = None
    if dgamma_q is None:
        dgamma_q = torch.empty_like(gamma_q)
    if dgamma_k is None:
        dgamma_k = torch.empty_like(gamma_k)
    _qknorm_rope_bwd(
        dx=kernel_dx,
        dq=dq,
        dk=dk,
        dv=kernel_dv,
        dgamma_q=dgamma_q,
        dgamma_k=dgamma_k,
        x=x,
        ssq=ssq,
        gamma_q=gamma_q,
        gamma_k=gamma_k,
        pos=pos,
        freq=freq,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_segments=num_segments,
        eps=eps,
        interleaved=interleaved,
    )
    return dx, dgamma_q, dgamma_k
