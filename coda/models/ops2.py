import torch
import triton
import flashinfer
from einops import reduce, rearrange
from liger_kernel import ops as liger_ops
from liger_kernel.ops.rope import _triton_rope
from liger_kernel.ops import cross_entropy as liger_cross_entropy_ops
from liger_kernel.ops import fused_linear_cross_entropy as liger_fused_linear_cross_entropy_ops
from liger_kernel.transformers import functional as liger_functional

from coda.kernels.refs.gpt2 import gemm_rmsnorm


def liger_fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ignore_index=-100,
    reduction="mean",
):
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(liger_fused_linear_cross_entropy_ops.MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # when doing matmul, use the original precision
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,

        # ensure _input and target are contiguous
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        liger_fused_linear_cross_entropy_ops.liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            weight_ptr=None,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=None,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            token_accuracy_ptr=None,
            token_accuracy_stride=0,
            predicted_tokens_ptr=None,
            predicted_tokens_stride=0,
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_n_non_ignore,
            weight_sum=0.0,
            ignore_index=ignore_index,
            lse_square_scale=0.0,
            label_smoothing=0.0,
            reduction=reduction,
            softcap=None,
            RETURN_Z_LOSS=False,
            RETURN_TOKEN_ACCURACY=False,
            RETURN_PREDICTED_TOKENS=False,
            HAS_WEIGHT=False,
            HAS_SOFTCAPPING=False,
            HAS_GRADIENTS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not liger_fused_linear_cross_entropy_ops.is_hip() else 16,
        )

        loss_1d[start_idx:end_idx] = loss_1d_slice

    if reduction == "none":
        loss = loss_1d
    else:
        loss = torch.sum(loss_1d)

    return loss


def liger_cross_entropy_forward(
    _input,
    target,
    ignore_index=-100,
    reduction="mean",
):
    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = min(liger_cross_entropy_ops.MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)

    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    # commenting them out as they break `torch.compile`
    # assert (target * target_mask).max() < _input.shape[-1], (
    #     f"Target {target.max()} is out of bounds. Expected < {_input.shape[-1]}"
    # )
    # assert (target * target_mask).min() >= 0, f"Target {target.min()} is out of bounds. Expected >= 0"
    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0

    # ensure _input and target are contiguous in the last dimension
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    liger_cross_entropy_ops.liger_cross_entropy_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),  # always 1
        weight_ptr=None,
        loss_ptr=loss_1d,
        z_loss_ptr=None,
        loss_stride=loss_1d.stride(-1),  # always 1
        token_accuracy_ptr=None,
        token_accuracy_stride=0,
        predicted_tokens_ptr=None,
        predicted_tokens_stride=0,
        n_cols=V,
        n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=sum_non_ignore_weight,
        ignore_index=ignore_index,
        weight_sum=weight_sum,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction=reduction,
        softcap=None,
        RETURN_Z_LOSS=False,
        RETURN_TOKEN_ACCURACY=False,
        RETURN_PREDICTED_TOKENS=False,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=False,
        HAS_SOFTCAPPING=False,
        HAS_GRADIENTS=False,
        num_warps=32 if not liger_cross_entropy_ops.is_hip() else 16,
    )

    if reduction == "none":
        loss = loss_1d
    else:
        loss = torch.sum(loss_1d)

    return loss


def rope(
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    bwd: bool = False,
    interleave: bool = False,
) -> torch.Tensor:
    assert cos.shape == sin.shape
    _, T, _, D = cos.shape
    cos = rearrange(cos, "b t h d -> b t 1 h d", b=1, t=T, h=1, d=D)
    sin = rearrange(sin, "b t h d -> b t 1 h d", b=1, t=T, h=1, d=D)
    if bwd:
        sin = -sin
    if interleave:
        pattern = "d pair"
    else:
        pattern = "pair d"

    y = rearrange(y, f"(b t) (trio h {pattern}) -> b t trio h pair d", t=T, trio=3, pair=2, d=D)
    y_rope = y.clone()
    # rotate pairs of dims, `:2` since we only rotate `q` and `k`
    y_rope[:, :, :2, :, 0, :] = y[:, :, :2, :, 0, :] *   cos  + y[:, :, :2, :, 1, :] * sin
    y_rope[:, :, :2, :, 1, :] = y[:, :, :2, :, 0, :] * (-sin) + y[:, :, :2, :, 1, :] * cos
    y_rope = rearrange(y_rope, f"b t trio h pair d -> (b t) (trio h {pattern})", t=T, trio=3, pair=2, d=D)
    return y_rope.to(dtype=y.dtype)


class SwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        g: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(g, u)
        return torch.nn.functional.silu(g) * u

    @staticmethod
    def backward(
        ctx,
        dout: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors
        smg = torch.nn.functional.sigmoid(g)
        slg = torch.nn.functional.silu(g)
        du = dout * slg
        dg = dout * u * (smg + slg * (1. - smg))
        return dg, du


class RoPEFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return rope(y=y, cos=cos, sin=sin)

    @staticmethod
    def backward(
        ctx,
        dout: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        return rope(y=dout, cos=cos, sin=sin, bwd=True), None, None


def rope_forward_liger_(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = rearrange(cos, "t d -> 1 t d")
    sin = rearrange(sin, "t d -> 1 t d")
    q = rearrange(q, "(b t) (h d) -> b t h d", t=cos.shape[1], d=cos.shape[2] * 2)
    k = rearrange(k, "(b t) (h d) -> b t h d", t=cos.shape[1], d=cos.shape[2] * 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_batch_size = cos.shape[0]

    _triton_rope[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=False,
    )

    q = rearrange(q, "b t h d -> (b t) (h d)", t=cos.shape[1], d=cos.shape[2] * 2)
    k = rearrange(k, "b t h d -> (b t) (h d)", t=cos.shape[1], d=cos.shape[2] * 2)
    return q, k


def rope_forward_flashinfer_(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q,
        key=k,
        head_size=cos_sin.shape[-1],
        cos_sin_cache=cos_sin,
    )
    return q, k


def rope_forward_flashinfer2(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions=positions,
        query=q,
        key=k,
        head_size=cos_sin.shape[-1],
        cos_sin_cache=cos_sin,
    )


def _gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = gemm_rmsnorm(A=A, B=B, R=R)

    if backend == "torch":
        G, U = torch.tensor_split(D, 2, dim=-1)
        O = torch.nn.functional.silu(G) * U

    elif backend == "liger":
        G, U = torch.tensor_split(D, 2, dim=-1)
        _, _, O = liger_ops.swiglu_forward(G, U)

    elif backend == "flashinfer":
        O = flashinfer.activation.silu_and_mul(D)

    else:
        raise NotImplementedError

    return D, O


_gemm_rmsnorm_swiglu_compiled = torch.compile(
    _gemm_rmsnorm_swiglu,
    fullgraph=True,
    dynamic=False,
)
_gemm_rmsnorm_swiglu_compiled_nofullgraph = torch.compile(
    _gemm_rmsnorm_swiglu,
    fullgraph=False,
    dynamic=False,
)


def gemm_rmsnorm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    backend: str,
    use_compile: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _gemm_rmsnorm_swiglu
    else:
        if backend == "flashinfer":
            fn = _gemm_rmsnorm_swiglu_compiled_nofullgraph
        else:
            fn = _gemm_rmsnorm_swiglu_compiled
    return fn(
        A=A,
        B=B,
        R=R,
        backend=backend,
    )


def _gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = torch.mm(A, B)

    if backend == "torch":
        G, U = torch.tensor_split(D, 2, dim=-1)
        O = torch.nn.functional.silu(G) * U

    elif backend == "liger":
        G, U = torch.tensor_split(D, 2, dim=-1)
        _, _, O = liger_ops.swiglu_forward(G, U)

    elif backend == "flashinfer":
        O = flashinfer.activation.silu_and_mul(D)

    else:
        raise NotImplementedError

    return D, O


_gemm_swiglu_compiled = torch.compile(
    _gemm_swiglu,
    fullgraph=True,
    dynamic=False,
)
_gemm_swiglu_compiled_nofullgraph = torch.compile(
    _gemm_swiglu,
    fullgraph=False,
    dynamic=False,
)


def gemm_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    backend: str,
    use_compile: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _gemm_swiglu
    else:
        if backend == "flashinfer":
            fn = _gemm_swiglu_compiled_nofullgraph
        else:
            fn = _gemm_swiglu_compiled
    return fn(
        A=A,
        B=B,
        backend=backend,
    )


def _gemm_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    targets: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    if backend == "torch":
        logits = torch.mm(A, B)
        loss = torch.nn.functional.cross_entropy(
            input=logits,
            target=targets,
        )

    elif backend == "liger":
        # liger cross entropy used transposed weight
        loss = liger_fused_linear_cross_entropy_forward(
            _input=A,
            weight=B,
            target=targets,
        )

    elif backend == "liger2":
        logits = torch.mm(A, B)
        loss = liger_cross_entropy_forward(
            _input=logits,
            target=targets,
        )

    else:
        raise NotImplementedError

    return loss


_gemm_cross_entropy_compiled = torch.compile(
    _gemm_cross_entropy,
    fullgraph=True,
    dynamic=False,
)


def gemm_cross_entropy(
    A: torch.Tensor,
    B: torch.Tensor,
    targets: torch.Tensor,
    backend: str,
    use_compile: bool,
) -> torch.Tensor:
    if not use_compile:
        fn = _gemm_cross_entropy
    else:
        fn = _gemm_cross_entropy_compiled
    return fn(
        A=A,
        B=B,
        targets=targets,
        backend=backend,
    )


def _gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    D = torch.mm(A, B)
    Q, K, V = torch.tensor_split(D, 3, dim=-1)

    if backend == "liger":
        Q, K = rope_forward_liger_(Q, K, cos=cos, sin=sin)

    elif backend == "flashinfer":
        Q, K = rope_forward_flashinfer_(Q, K, cos_sin=cos_sin, positions=positions)

    elif backend == "flashinfer2":
        Q, K = rope_forward_flashinfer2(Q, K, cos_sin=cos_sin, positions=positions)

    else:
        raise NotImplementedError

    O = (Q, K, V)
    return D, O


_gemm_rope_compiled = torch.compile(
    _gemm_rope,
    fullgraph=True,
    dynamic=False,
)
_gemm_rope_compiled_nofullgraph = torch.compile(
    _gemm_rope,
    fullgraph=False,
    dynamic=False,
)


def gemm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    backend: str,
    use_compile: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not use_compile:
        fn = _gemm_rope
    else:
        if backend in ("flashinfer", "flashinfer2"):
            fn = _gemm_rope_compiled_nofullgraph
        else:
            fn = _gemm_rope_compiled
    return fn(
        A=A,
        B=B,
        cos=cos,
        sin=sin,
        cos_sin=cos_sin,
        positions=positions,
        backend=backend,
    )


def _gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    D = gemm_rmsnorm(A=A, B=B, R=R)
    Q, K, V = torch.tensor_split(D, 3, dim=-1)

    if backend == "liger":
        Q, K = rope_forward_liger_(Q, K, cos=cos, sin=sin)

    elif backend == "flashinfer":
        Q, K = rope_forward_flashinfer_(Q, K, cos_sin=cos_sin, positions=positions)

    elif backend == "flashinfer2":
        Q, K = rope_forward_flashinfer2(Q, K, cos_sin=cos_sin, positions=positions)

    else:
        raise NotImplementedError

    O = (Q, K, V)
    return D, O


_gemm_rmsnorm_rope_compiled = torch.compile(
    _gemm_rmsnorm_rope,
    fullgraph=True,
    dynamic=False,
)
_gemm_rmsnorm_rope_compiled_nofullgraph = torch.compile(
    _gemm_rmsnorm_rope,
    fullgraph=False,
    dynamic=False,
)


def gemm_rmsnorm_rope(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    backend: str,
    use_compile: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not use_compile:
        fn = _gemm_rmsnorm_rope
    else:
        if backend in ("flashinfer", "flashinfer2"):
            fn = _gemm_rmsnorm_rope_compiled_nofullgraph
        else:
            fn = _gemm_rmsnorm_rope_compiled
    return fn(
        A=A,
        B=B,
        R=R,
        cos=cos,
        sin=sin,
        cos_sin=cos_sin,
        positions=positions,
        backend=backend,
    )


def _gemm_partial_swiglu_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    block_size: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # B is expected pre-transposed by the caller (shape (K, N) rather than (N, K))
    assert Z.shape[1] == B.shape[1] * 2
    D = torch.mm(A, B)

    if backend == "liger":
        G, U = torch.tensor_split(Z, 2, dim=-1)
        dG, dU = liger_ops.swiglu_backward(G, U, D)
        dZ = torch.cat([dG, dU], dim=-1)
        ZdZ = reduce(Z * dZ, "m (nb bs 2) -> m nb", "sum", bs=block_size) / block_size

    else:
        raise NotImplementedError

    return dZ, ZdZ, None


_gemm_partial_swiglu_bwd_compiled = torch.compile(
    _gemm_partial_swiglu_bwd,
    fullgraph=True,
    dynamic=False,
)


def gemm_partial_swiglu_bwd(
    A: torch.Tensor,
    B: torch.Tensor,
    Z: torch.Tensor,
    block_size: int,
    backend: str,
    use_compile: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_compile:
        fn = _gemm_partial_swiglu_bwd
    else:
        if backend == "flashinfer":
            raise NotImplementedError
        else:
            fn = _gemm_partial_swiglu_bwd_compiled
    return fn(
        A=A,
        B=B,
        Z=Z,
        block_size=block_size,
        backend=backend,
    )


def _gemm_residual_rmsnorm_gemm(
    x: torch.Tensor,
    y: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_n: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    cos_sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    targets: torch.Tensor | None,
    eps: float,
    epilogue: str | None,
    transpose: bool,
    backend: str,
) -> tuple[torch.Tensor,
           torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
           torch.Tensor | None,
           torch.Tensor,
           torch.Tensor]:

    B, T, D0 = y.shape
    if transpose or all([
        epilogue == "cross-entropy",
        backend == "liger",
    ]):
        # liger cross entropy used transposed weight
        D2, D1 = w_b.shape
    else:
        D1, D2 = w_b.shape
    if transpose:
        assert w_a.shape == (D1, D0)
    else:
        assert w_a.shape == (D0, D1)

    assert x.shape == (B, T, D1)
    assert y.shape == (B, T, D0)
    x = rearrange(x, "b t d -> (b t) d", b=B, t=T, d=D1)
    y = rearrange(y, "b t d -> (b t) d", b=B, t=T, d=D0)

    if backend == "torch":
        if transpose:
            x_out = torch.nn.functional.linear(y, w_a) + x
        else:
            x_out = torch.addmm(x, y, w_a)
        h = torch.nn.functional.rms_norm(x_out, normalized_shape=(x_out.shape[-1],), weight=w_n, eps=eps)

    elif backend in ("liger", "liger2"):
        if transpose:
            z = torch.nn.functional.linear(y, w_a)
        else:
            z = torch.mm(y, w_a)
        h, x_out = liger_functional.liger_fused_add_rms_norm(X=z, R=x, W=w_n, eps=eps)

    elif backend in ("flashinfer", "flashinfer2"):
        if transpose:
            z = torch.nn.functional.linear(y, w_a)
        else:
            z = torch.mm(y, w_a)
        # this is modified in-place
        flashinfer.norm.fused_add_rmsnorm(input=z, residual=x, weight=w_n, eps=eps)
        h = z
        x_out = x

    else:
        raise NotImplementedError

    if all([
        epilogue == "cross-entropy",
        backend == "liger",
    ]):
        # liger's fused linear cross-entropy fuses the second gemm internally
        z_out = None
    elif transpose:
        z_out = torch.nn.functional.linear(h, w_b)
    else:
        z_out = torch.mm(h, w_b)

    if epilogue is None:
        y_out = None

    elif epilogue == "swiglu":
        assert z_out is not None

        if backend == "torch":
            g, u = torch.tensor_split(z_out, 2, dim=-1)
            y_out = SwiGLUFunction.apply(g, u)

        elif backend in ("liger", "liger2"):
            g, u = torch.tensor_split(z_out, 2, dim=-1)
            y_out = liger_ops.LigerSiLUMulFunction.apply(g, u)

        elif backend in ("flashinfer", "flashinfer2"):
            y_out = flashinfer.activation.silu_and_mul(z_out)

        else:
            raise NotImplementedError

        y_out = rearrange(y_out, "(b t) d -> b t d", b=B, t=T, d=D2 // 2)

    elif epilogue == "rope":
        assert z_out is not None

        if backend == "torch":
            y_out = RoPEFunction.apply(z_out, cos, sin)
            y_out = rearrange(y_out, "(b t) d -> b t d", b=B, t=T, d=D2)

        elif backend in ("liger", "liger2"):
            cos = rearrange(cos, "t d -> 1 t d")
            sin = rearrange(sin, "t d -> 1 t d")
            q, k, v = torch.tensor_split(z_out, 3, dim=-1)
            qt = rearrange(q, "(b t) (h d) -> b h t d", t=cos.shape[1], d=cos.shape[2] * 2)
            kt = rearrange(k, "(b t) (h d) -> b h t d", t=cos.shape[1], d=cos.shape[2] * 2)
            qt, kt = liger_ops.LigerRopeFunction.apply(qt, kt, cos, sin)
            q = rearrange(qt, "b h t d -> b t (h d)", t=cos.shape[1], d=cos.shape[2] * 2)
            k = rearrange(kt, "b h t d -> b t (h d)", t=cos.shape[1], d=cos.shape[2] * 2)
            y_out = (
                q,
                k,
                rearrange(v, "(b t) d -> b t d", b=B, t=T, d=D2 // 3),
            )

        elif backend == "flashinfer":
            q, k, v = torch.tensor_split(z_out, 3, dim=-1)
            q, k = rope_forward_flashinfer_(q, k, cos_sin=cos_sin, positions=positions)
            y_out = (
                rearrange(q, "(b t) d -> b t d", b=B, t=T, d=D2 // 3),
                rearrange(k, "(b t) d -> b t d", b=B, t=T, d=D2 // 3),
                rearrange(v, "(b t) d -> b t d", b=B, t=T, d=D2 // 3),
            )

        elif backend == "flashinfer2":
            q, k, v = torch.tensor_split(z_out, 3, dim=-1)
            q, k = rope_forward_flashinfer2(q, k, cos_sin=cos_sin, positions=positions)
            y_out = (
                rearrange(q, "(b t) d -> b t d", b=B, t=T, d=D2 // 3),
                rearrange(k, "(b t) d -> b t d", b=B, t=T, d=D2 // 3),
                rearrange(v, "(b t) d -> b t d", b=B, t=T, d=D2 // 3),
            )

        else:
            raise NotImplementedError

    elif epilogue == "cross-entropy":
        targets = rearrange(targets, "b t -> (b t)", b=B, t=T)

        if backend == "torch":
            assert z_out is not None
            loss = torch.nn.functional.cross_entropy(input=z_out, target=targets)

        elif backend == "liger":
            assert z_out is None
            loss = liger_fused_linear_cross_entropy_forward(_input=h, weight=w_b, target=targets)

        elif backend == "liger2":
            assert z_out is not None
            loss = liger_cross_entropy_forward(_input=z_out, target=targets)

        else:
            raise NotImplementedError

        y_out = loss
    else:
        raise NotImplementedError

    x_out = rearrange(x_out, "(b t) d -> b t d", b=B, t=T, d=D1)
    z_out = rearrange(z_out, "(b t) d -> b t d", b=B, t=T, d=D2) if z_out is not None else None
    return x_out, y_out, z_out, None, None


_gemm_residual_rmsnorm_gemm_compiled = torch.compile(
    _gemm_residual_rmsnorm_gemm,
    fullgraph=True,
    dynamic=False,
)
_gemm_residual_rmsnorm_gemm_compiled_nofullgraph = torch.compile(
    _gemm_residual_rmsnorm_gemm,
    fullgraph=False,
    dynamic=False,
)


def gemm_residual_rmsnorm_gemm(
    x: torch.Tensor,
    y: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    w_n: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    cos_sin: torch.Tensor | None,
    positions: torch.Tensor | None,
    targets: torch.Tensor | None,
    eps: float,
    epilogue: str | None,
    transpose: bool,
    backend: str,
    use_compile: bool,
) -> tuple[torch.Tensor,
           torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
           torch.Tensor,
           torch.Tensor,
           torch.Tensor]:

    if not use_compile:
        fn = _gemm_residual_rmsnorm_gemm
    else:
        if backend in ("flashinfer", "flashinfer2"):
            fn = _gemm_residual_rmsnorm_gemm_compiled_nofullgraph
        else:
            fn = _gemm_residual_rmsnorm_gemm_compiled
    return fn(
        x=x,
        y=y,
        w_a=w_a,
        w_b=w_b,
        w_n=w_n,
        cos=cos,
        sin=sin,
        cos_sin=cos_sin,
        positions=positions,
        targets=targets,
        eps=eps,
        epilogue=epilogue,
        transpose=transpose,
        backend=backend,
    )


def layer(
    x0: torch.Tensor,
    y0: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    wn0: torch.Tensor,
    wn1: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    transpose: bool,
    backend: str,
    use_compile: bool,
) -> tuple[torch.Tensor, ...]:
    x1, y1, _, _, _ = gemm_residual_rmsnorm_gemm(
        x=x0,
        y=y0,
        w_a=w0,
        w_b=w1,
        w_n=wn0,
        cos=None,
        sin=None,
        cos_sin=None,
        positions=None,
        targets=None,
        eps=eps,
        epilogue="swiglu",
        transpose=transpose,
        backend=backend,
        use_compile=use_compile,
    )
    x2, y2, _, _, _ = gemm_residual_rmsnorm_gemm(
        x=x1,
        y=y1,
        w_a=w2,
        w_b=w3,
        w_n=wn1,
        cos=cos,
        sin=sin,
        cos_sin=cos_sin,
        positions=positions,
        targets=None,
        eps=eps,
        epilogue="rope",
        transpose=transpose,
        backend=backend,
        use_compile=use_compile,
    )
    if backend == "torch":
        return x2, y2
    else:
        q, k, v = y2
        return x2, q, k, v
