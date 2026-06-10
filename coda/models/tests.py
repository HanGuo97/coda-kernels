import math
import torch
import pytest
from dataclasses import dataclass
from typing import cast
from einops import rearrange, reduce

from coda.models import ops
from coda.models import gpt
from coda.models import gpt_ref
from coda.kernels.benchmarks.block import GPTConfig2


@dataclass
class GPTConfigSmall:
    sequence_len: int = 512
    vocab_size: int = 2048
    n_layer: int = 2
    n_head: int = 2
    n_kv_head: int = 2
    n_embd: int = 128


@dataclass
class Tolerances:
    DW0: float
    DW1: float
    DW2: float
    DW3: float
    DWL: float
    DX: float
    DY: float
    DWN: float
    Y: float
    Z: float
    LOGITS: float


BLOCKSIZE = 128

# Default fp32 tolerances used by torch.testing.assert_close.
DEFAULT_FP32_RTOL = 1.3e-6
DEFAULT_FP32_ATOL = 1e-5

TIGHT_FP32_TOLS_SMALL = Tolerances(
    DW0=DEFAULT_FP32_ATOL * 1.5,   # max observed: 1.431e-5
    DW1=DEFAULT_FP32_ATOL,         # max observed: 8.821e-6 (below default)
    DW2=DEFAULT_FP32_ATOL * 1.8,   # max observed: 1.717e-5
    DW3=DEFAULT_FP32_ATOL * 1.1,   # max observed: 1.049e-5
    DWL=DEFAULT_FP32_ATOL * 2.5,   # max observed: 2.480e-5
    DX =DEFAULT_FP32_ATOL * 1.2,   # max observed: 1.168e-5 (varies run-to-run)
    DY =DEFAULT_FP32_ATOL,         # max observed: 3.338e-6 (below default)
    DWN=DEFAULT_FP32_ATOL,         # max observed: 4.292e-6 (below default)
    Y  =DEFAULT_FP32_ATOL,         # max observed: 2.980e-6 (below default)
    Z  =DEFAULT_FP32_ATOL,         # max observed: 2.384e-6 (below default)
    LOGITS=DEFAULT_FP32_ATOL,      # max observed: 2.861e-6 (below default)
)

TIGHT_FP32_TOLS_BENCH = Tolerances(
    DW0=DEFAULT_FP32_ATOL * 13.0,  # max observed: 1.221e-4
    DW1=DEFAULT_FP32_ATOL *  7.5,  # max observed: 7.498e-5
    DW2=DEFAULT_FP32_ATOL * 13.0,  # max observed: 1.225e-4
    DW3=DEFAULT_FP32_ATOL *  6.8,  # max observed: 6.771e-5
    DWL=DEFAULT_FP32_ATOL *  7.5,  # max observed: 7.474e-5
    DX =DEFAULT_FP32_ATOL * 11.0,  # max observed: 1.087e-4
    DY =DEFAULT_FP32_ATOL *  2.1,  # max observed: 2.050e-5
    DWN=DEFAULT_FP32_ATOL *  3.6,  # max observed: 3.517e-5
    Y  =DEFAULT_FP32_ATOL *  1.5,  # max observed: 1.431e-5
    Z  =DEFAULT_FP32_ATOL *  1.2,  # max observed: 1.144e-5
    LOGITS=DEFAULT_FP32_ATOL * 1.3,# max observed: 1.216e-5
)


def concat_qkv(module: gpt_ref.CausalSelfAttention) -> torch.Tensor:
    return torch.cat(
        [
            module.c_q.weight,
            module.c_k.weight,
            module.c_v.weight,
        ],
        dim=0,
    )


def concat_gateup(module: gpt_ref.MLP) -> torch.Tensor:
    return torch.cat(
        [
            module.c_gate.weight,
            module.c_up.weight,
        ],
        dim=0,
    )


def interleave_rope(
    tensor: torch.Tensor,
    num_heads: int,
    preprocess: bool = True,
) -> torch.Tensor:
    if preprocess:
        return rearrange(tensor, "... (trio h pair n) -> ... (trio h n pair)", trio=3, h=num_heads, pair=2)
    else:
        return rearrange(tensor, "... (trio h n pair) -> ... (trio h pair n)", trio=3, h=num_heads, pair=2)


def interleave_swiglu(
    tensor: torch.Tensor,
    preprocess: bool = True,
) -> torch.Tensor:
    if preprocess:
        return rearrange(tensor, "... (pair n) -> ... (n pair)", pair=2)
    else:
        return rearrange(tensor, "... (n pair) -> ... (pair n)", pair=2)


@torch.no_grad()
def convert_weights(
    model: gpt.GPT,
    model_ref: gpt_ref.GPT,
) -> None:
    model.block0.embedding.weight.copy_(model_ref.transformer.wte.weight)
    model.blockL.lm_head.weight.copy_(model_ref.lm_head.weight)

    ref_block0 = cast(gpt_ref.Block, model_ref.transformer.h[0])
    ref_blockL = cast(gpt_ref.Block, model_ref.transformer.h[-1])
    model.block0.proj_qkv.weight.copy_(concat_qkv(ref_block0.attn))
    model.blockL.proj_out.weight.copy_(ref_blockL.attn.c_proj.weight)
    model.blockL.proj_down.weight.copy_(ref_blockL.mlp.c_down.weight)
    model.blockL.proj_gateup.weight.copy_(concat_gateup(ref_blockL.mlp))

    for block, ref_block0, ref_block1 in zip(
        model.blocks,
        model_ref.transformer.h[:-1],
        model_ref.transformer.h[1:],
    ):
        assert isinstance(block, gpt.Block)
        assert isinstance(ref_block0, gpt_ref.Block)
        assert isinstance(ref_block1, gpt_ref.Block)
        block.proj_qkv.weight.copy_(concat_qkv(ref_block1.attn))
        block.proj_out.weight.copy_(ref_block0.attn.c_proj.weight)
        block.proj_down.weight.copy_(ref_block0.mlp.c_down.weight)
        block.proj_gateup.weight.copy_(concat_gateup(ref_block0.mlp))


def check_models_have_same_parameter_count(config: gpt.GPTConfig) -> None:
    """Test that both models have the same number of parameters."""
    model = gpt.GPT(config)
    model_ref = gpt_ref.GPT(config)
    params = sum(p.numel() for p in model.parameters())
    params_ref = sum(p.numel() for p in model_ref.parameters())
    assert params == params_ref, f"Parameter count mismatch: {params:,} vs {params_ref:,}"


@torch.compiler.disable
def check_layer_fwd_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cos_sin: tuple[torch.Tensor, torch.Tensor],
    block: gpt.Block,
    block_ref0: gpt_ref.Block,
    block_ref1: gpt_ref.Block,
    eps: float,
    tols: Tolerances,
) -> None:
    cos, sin = cos_sin
    B, T, _ = x.shape
    H = block_ref1.attn.n_head
    D = block_ref1.attn.head_dim
    assert block_ref1.attn.n_head == block_ref1.attn.n_kv_head

    wn0_ref = torch.randn(block.hidden_dim, dtype=x.dtype, device=x.device, requires_grad=True)
    wn1_ref = torch.randn(block.hidden_dim, dtype=x.dtype, device=x.device, requires_grad=True)

    # Ref
    r1_ref = x + block_ref0.attn.c_proj(y)
    x1_ref = r1_ref.clone()
    h1_ref = gpt_ref.norm(r1_ref, weight=wn0_ref, eps=eps)
    g_ref = block_ref0.mlp.c_gate(h1_ref)
    u_ref = block_ref0.mlp.c_up(h1_ref)
    y1_ref = torch.nn.functional.silu(g_ref) * u_ref
    r2_ref = x1_ref + block_ref0.mlp.c_down(y1_ref)
    x2_ref = r2_ref.clone()
    h2_ref = gpt_ref.norm(r2_ref, weight=wn1_ref, eps=eps)
    q_ref = block_ref1.attn.c_q(h2_ref)
    k_ref = block_ref1.attn.c_k(h2_ref)
    v_ref = block_ref1.attn.c_v(h2_ref)
    q_rope_ref = rearrange(gpt_ref.apply_rotary_emb(rearrange(q_ref, "b t (h d) -> b t h d", h=H, d=D), cos, sin), "b t h d -> b t (h d)")
    k_rope_ref = rearrange(gpt_ref.apply_rotary_emb(rearrange(k_ref, "b t (h d) -> b t h d", h=H, d=D), cos, sin), "b t h d -> b t (h d)")
    y2_ref = torch.cat([q_rope_ref, k_rope_ref, v_ref], dim=-1)
    z1_ref = torch.cat([g_ref, u_ref], dim=-1)
    z2_ref = torch.cat([q_ref, k_ref, v_ref], dim=-1)
    rstd1_ref = torch.rsqrt(reduce(r1_ref ** 2, "... d -> ...", "mean") + eps)
    rstd2_ref = torch.rsqrt(reduce(r2_ref ** 2, "... d -> ...", "mean") + eps)

    # Ours
    w0 = block.proj_out.weight.T
    w1 = interleave_swiglu(block.proj_gateup.weight.T)
    w2 = block.proj_down.weight.T
    w3 = interleave_rope(block.proj_qkv.weight.T, num_heads=H)
    wn0 = wn0_ref.detach().clone().requires_grad_(True)
    wn1 = wn1_ref.detach().clone().requires_grad_(True)
    cos_sin_preprocessed = gpt.preprocess_rope(cos=cos, sin=sin, batch_size=B, num_heads=H)
    cos_raw = rearrange(cos, "1 t 1 d -> t d")
    sin_raw = rearrange(sin, "1 t 1 d -> t d")
    x_out, y_out = ops.layer(
        x0=x,
        y0=y,
        w0=w0,
        w1=w1,
        w2=w2,
        w3=w3,
        wn0=wn0,
        wn1=wn1,
        cos_sin=cos_sin_preprocessed,
        cos=cos_raw,
        sin=sin_raw,
        num_heads=H,
        head_dim=D,
        eps=eps,
        transpose=False,
        backend="torch",
        use_compile=False,
    )
    y_out = interleave_rope(y_out, num_heads=H, preprocess=False)

    grad_x = torch.randn_like(x_out) / math.sqrt(block.hidden_dim)
    grad_y = torch.randn_like(y_out) / math.sqrt(block.hidden_dim)

    (
        dx_ref, dy_ref, dx1_ref, dg_ref, du_ref, dq_ref, dk_ref, dv_ref,
        dwn0_ref, dwn1_ref, dw0_ref, dw1g_ref, dw1u_ref, dw2_ref, dw3q_ref, dw3k_ref, dw3v_ref,
    ) = torch.autograd.grad(
        outputs=(x2_ref, y2_ref),
        inputs=(
            x,
            y,
            x1_ref,
            g_ref,
            u_ref,
            q_ref,
            k_ref,
            v_ref,
            wn0_ref,
            wn1_ref,
            block_ref0.attn.c_proj.weight,
            block_ref0.mlp .c_gate.weight,
            block_ref0.mlp .c_up  .weight,
            block_ref0.mlp .c_down.weight,
            block_ref1.attn.c_q   .weight,
            block_ref1.attn.c_k   .weight,
            block_ref1.attn.c_v   .weight,
        ),
        grad_outputs=(grad_x, grad_y),
    )
    dw1_ref = torch.cat([dw1g_ref, dw1u_ref], dim=0)
    dw3_ref = torch.cat([dw3q_ref, dw3k_ref, dw3v_ref], dim=0)
    dz1_ref = torch.cat([dg_ref, du_ref], dim=-1)
    dz2_ref = torch.cat([dq_ref, dk_ref, dv_ref], dim=-1)

    dx_out, dy_out, dwn0_out, dwn1_out, dw0_out, dw1_out, dw2_out, dw3_out = torch.autograd.grad(
        outputs=(x_out, y_out),
        inputs=(
            x,
            y,
            wn0,
            wn1,
            block.proj_out   .weight,
            block.proj_gateup.weight,
            block.proj_down  .weight,
            block.proj_qkv   .weight,
        ),
        grad_outputs=(grad_x, grad_y),
    )

    torch.testing.assert_close(x_out, x2_ref)
    torch.testing.assert_close(y_out, y2_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Y)
    torch.testing.assert_close(dx_out, dx_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DX)
    torch.testing.assert_close(dy_out, dy_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DY)
    torch.testing.assert_close(dwn0_out, dwn0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dwn1_out, dwn1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dw0_out, dw0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW0)
    torch.testing.assert_close(dw1_out, dw1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW1)
    torch.testing.assert_close(dw2_out, dw2_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW2)
    torch.testing.assert_close(dw3_out, dw3_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW3)

    x1_out2, y1_out2, z1_out2, rstd1_out2 = ops.gemm_residual_rmsnorm_gemm_fwd(
        x=x,
        y=y,
        w_a=w0,
        w_b=w1,
        w_n=wn0,
        block_size_norm=BLOCKSIZE,
        block_size_loss=None,
        cos_sin=None,
        targets=None,
        eps=eps,
        epilogue="swiglu",
        backend="torch",
        use_quack=False,
    )
    x2_out2, y2_out2, z2_out2, rstd2_out2 = ops.gemm_residual_rmsnorm_gemm_fwd(
        x=x1_out2,
        y=y1_out2,
        w_a=w2,
        w_b=w3,
        w_n=wn1,
        block_size_norm=BLOCKSIZE,
        block_size_loss=None,
        cos_sin=cos_sin_preprocessed,
        targets=None,
        eps=eps,
        epilogue="rope",
        backend="torch",
        use_quack=False,
    )
    z1_out2_ = interleave_swiglu(z1_out2, preprocess=False)
    z2_out2_ = interleave_rope(z2_out2, num_heads=H, preprocess=False)
    y2_out2_ = interleave_rope(y2_out2, num_heads=H, preprocess=False)
    torch.testing.assert_close(x1_out2, x1_ref)
    torch.testing.assert_close(x2_out2, x2_ref)
    torch.testing.assert_close(y1_out2, y1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Y)
    torch.testing.assert_close(y2_out2_, y2_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Y)
    torch.testing.assert_close(z1_out2_, z1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Z)
    torch.testing.assert_close(z2_out2_, z2_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Z)
    torch.testing.assert_close(rstd1_out2, rstd1_ref)
    torch.testing.assert_close(rstd2_out2, rstd2_ref)

    dz2_out2, zdz2_out2 = ops.attn_bwd_rope_patch(
        z=z2_out2,
        dy=interleave_rope(grad_y, num_heads=H),
        cos_sin=cos_sin_preprocessed,
        cos=cos_raw,
        sin=sin_raw,
        num_heads=H,
        head_dim=D,
        use_quack=False,
    )
    dx1_out2, dz1_out2, dw2_out2, dw3_out2, dwn1_out2, zdz1_out2 = ops.gemm_residual_rmsnorm_gemm_bwd(
        x=x2_out2,
        z=z1_out2,
        dx=grad_x,
        dz=dz2_out2,
        w_a=w2,
        w_b=w3,
        w_n=wn1,
        rstd=rstd2_out2,
        zdz_prev=zdz2_out2,
        block_size_prev=None,
        block_size_curr=BLOCKSIZE,
        block_size_norm=BLOCKSIZE,
        epilogue="swiglu",
        backend="torch",
    )
    dx_out2, dy_out2, dw0_out2, dw1_out2, dwn0_out2, _ = ops.gemm_residual_rmsnorm_gemm_bwd(
        x=x1_out2,
        z=y,
        dx=dx1_out2,
        dz=dz1_out2,
        w_a=w0,
        w_b=w1,
        w_n=wn0,
        rstd=rstd1_out2,
        zdz_prev=zdz1_out2,
        block_size_prev=BLOCKSIZE,
        block_size_curr=None,
        block_size_norm=BLOCKSIZE,
        epilogue=None,
        backend="torch",
    )
    dz1_out2_ = interleave_swiglu(dz1_out2, preprocess=False)
    dz2_out2_ = interleave_rope(dz2_out2, num_heads=H, preprocess=False)
    dw1_out2_ = interleave_swiglu(dw1_out2, preprocess=False)
    dw3_out2_ = interleave_rope(dw3_out2, num_heads=H, preprocess=False)
    torch.testing.assert_close(dx_out2, dx_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DX)
    torch.testing.assert_close(dy_out2, dy_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DY)
    torch.testing.assert_close(dx1_out2, dx1_ref)
    torch.testing.assert_close(dz1_out2_, dz1_ref)
    torch.testing.assert_close(dz2_out2_, dz2_ref)
    torch.testing.assert_close(dwn0_out2, dwn0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dwn1_out2, dwn1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dw0_out2.T, dw0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW0)
    torch.testing.assert_close(dw1_out2_.T, dw1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW1)
    torch.testing.assert_close(dw2_out2.T, dw2_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW2)
    torch.testing.assert_close(dw3_out2_.T, dw3_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW3)


@torch.compiler.disable
def check_layer_post_fwd_bwd(
    x: torch.Tensor,
    y: torch.Tensor,
    targets: torch.Tensor,
    block: gpt.BlockPost,
    block_ref: gpt_ref.Block,
    lm_head_ref: torch.nn.Linear,
    eps: float,
    tols: Tolerances,
) -> None:
    B, T, _ = x.shape

    wn0_ref = torch.randn(block.hidden_dim, dtype=x.dtype, device=x.device, requires_grad=True)
    wn1_ref = torch.randn(block.hidden_dim, dtype=x.dtype, device=x.device, requires_grad=True)

    # Ref
    r1_ref = x + block_ref.attn.c_proj(y)
    x1_ref = r1_ref.clone()
    h1_ref = gpt_ref.norm(r1_ref, weight=wn0_ref, eps=eps)
    g_ref = block_ref.mlp.c_gate(h1_ref)
    u_ref = block_ref.mlp.c_up(h1_ref)
    y1_ref = torch.nn.functional.silu(g_ref) * u_ref
    r2_ref = x1_ref + block_ref.mlp.c_down(y1_ref)
    x2_ref = r2_ref.clone()
    h2_ref = gpt_ref.norm(r2_ref, weight=wn1_ref, eps=eps)
    logits_ref = lm_head_ref(h2_ref)
    loss_ref = torch.nn.functional.cross_entropy(
        input=rearrange(logits_ref, "b t d -> (b t) d"),
        target=rearrange(targets, "b t -> (b t)"),
        ignore_index=-1,
        reduction="none",
    )
    loss_ref = rearrange(loss_ref, "(b t) -> b t", b=B, t=T)
    z1_ref = torch.cat([g_ref, u_ref], dim=-1)
    rstd1_ref = torch.rsqrt(reduce(r1_ref ** 2, "... d -> ...", "mean") + eps)
    rstd2_ref = torch.rsqrt(reduce(r2_ref ** 2, "... d -> ...", "mean") + eps)

    # Ours
    w0 = block.proj_out.weight.T
    w1 = interleave_swiglu(block.proj_gateup.weight.T)
    w2 = block.proj_down.weight.T
    wl = block.lm_head.weight.T
    wn0 = wn0_ref.detach().clone().requires_grad_(True)
    wn1 = wn1_ref.detach().clone().requires_grad_(True)
    loss_out = ops.layer_post(
        x0=x,
        y0=y,
        w0=w0,
        w1=w1,
        w2=w2,
        w3=wl,
        wn0=wn0,
        wn1=wn1,
        targets=targets,
        eps=eps,
        transpose=False,
        backend="torch",
        use_compile=False,
    )

    grad_loss = torch.randn_like(loss_out)

    (
        dx_ref, dy_ref, dx1_ref, dg_ref, du_ref, dlogits_ref,
        dwn0_ref, dwn1_ref, dw0_ref, dw1g_ref, dw1u_ref, dw2_ref, dwl_ref,
    ) = torch.autograd.grad(
        outputs=(loss_ref,),
        inputs=(
            x,
            y,
            x1_ref,
            g_ref,
            u_ref,
            logits_ref,
            wn0_ref,
            wn1_ref,
            block_ref.attn.c_proj.weight,
            block_ref.mlp .c_gate.weight,
            block_ref.mlp .c_up  .weight,
            block_ref.mlp .c_down.weight,
            lm_head_ref          .weight,
        ),
        grad_outputs=(grad_loss,),
    )
    dw1_ref = torch.cat([dw1g_ref, dw1u_ref], dim=0)
    dz1_ref = torch.cat([dg_ref, du_ref], dim=-1)

    dx_out, dy_out, dwn0_out, dwn1_out, dw0_out, dw1_out, dw2_out, dwl_out = torch.autograd.grad(
        outputs=(loss_out,),
        inputs=(
            x,
            y,
            wn0,
            wn1,
            block.proj_out   .weight,
            block.proj_gateup.weight,
            block.proj_down  .weight,
            block.lm_head    .weight,
        ),
        grad_outputs=(grad_loss,),
    )

    torch.testing.assert_close(loss_out, loss_ref)
    torch.testing.assert_close(dx_out, dx_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DX)
    torch.testing.assert_close(dy_out, dy_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DY)
    torch.testing.assert_close(dwn0_out, dwn0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dwn1_out, dwn1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dw0_out, dw0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW0)
    torch.testing.assert_close(dw1_out, dw1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW1)
    torch.testing.assert_close(dw2_out, dw2_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW2)
    torch.testing.assert_close(dwl_out, dwl_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWL)

    with torch.no_grad():
        x1_out2, y1_out2, z1_out2, rstd1_out2 = ops.gemm_residual_rmsnorm_gemm_fwd(
            x=x,
            y=y,
            w_a=w0,
            w_b=w1,
            w_n=wn0,
            block_size_norm=BLOCKSIZE,
            block_size_loss=None,
            cos_sin=None,
            targets=None,
            eps=eps,
            epilogue="swiglu",
            backend="torch",
            use_quack=False,
        )
        x2_out2, (logits_tgt_out2, logits_lse_out2), logits_out2, rstd2_out2 = ops.gemm_residual_rmsnorm_gemm_fwd(
            x=x1_out2,
            y=y1_out2,
            w_a=w2,
            w_b=wl,
            w_n=wn1,
            block_size_norm=BLOCKSIZE,
            block_size_loss=BLOCKSIZE,
            cos_sin=None,
            targets=targets,
            eps=eps,
            epilogue="cross-entropy",
            backend="torch",
            use_quack=False,
        )
        loss_out2, dlogits_out2, zdz2_out2 = ops.cross_entropy(
            logits=logits_out2,
            targets=targets,
            logits_tgt=logits_tgt_out2,
            logits_lse=logits_lse_out2,
            block_size=BLOCKSIZE,
        )
        # `ops.cross_entropy` returns the unit grad of loss w.r.t. logits;
        # autograd's `dlogits_ref` includes the `grad_loss` scaling.
        dlogits_out2_ = dlogits_out2 * rearrange(grad_loss, "b t -> b t 1")
        z1_out2_ = interleave_swiglu(z1_out2, preprocess=False)
        torch.testing.assert_close(loss_out2, loss_ref)
        torch.testing.assert_close(x1_out2, x1_ref)
        torch.testing.assert_close(x2_out2, x2_ref)
        torch.testing.assert_close(y1_out2, y1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Y)
        torch.testing.assert_close(z1_out2_, z1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Z)
        torch.testing.assert_close(logits_out2, logits_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.LOGITS)
        torch.testing.assert_close(dlogits_out2_, dlogits_ref)
        torch.testing.assert_close(rstd1_out2, rstd1_ref)
        torch.testing.assert_close(rstd2_out2, rstd2_ref)

        dx1_out2, dz1_out2, dw2_out2, dwl_out2, dwn1_out2, zdz1_out2 = ops.gemm_residual_rmsnorm_gemm_bwd(
            x=x2_out2,
            z=z1_out2,
            dx=torch.zeros_like(x2_out2),
            dz=dlogits_out2_,
            w_a=w2,
            w_b=wl,
            w_n=wn1,
            rstd=rstd2_out2,
            zdz_prev=zdz2_out2 * grad_loss,
            block_size_prev=None,
            block_size_curr=BLOCKSIZE,
            block_size_norm=BLOCKSIZE,
            epilogue="swiglu",
            backend="torch",
        )
        dx_out2, dy_out2, dw0_out2, dw1_out2, dwn0_out2, _ = ops.gemm_residual_rmsnorm_gemm_bwd(
            x=x1_out2,
            z=y,
            dx=dx1_out2,
            dz=dz1_out2,
            w_a=w0,
            w_b=w1,
            w_n=wn0,
            rstd=rstd1_out2,
            zdz_prev=zdz1_out2,
            block_size_prev=BLOCKSIZE,
            block_size_curr=None,
            block_size_norm=BLOCKSIZE,
            epilogue=None,
            backend="torch",
        )
        dz1_out2_ = interleave_swiglu(dz1_out2, preprocess=False)
        dw1_out2_ = interleave_swiglu(dw1_out2, preprocess=False)
        torch.testing.assert_close(dx_out2, dx_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DX)
        torch.testing.assert_close(dy_out2, dy_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DY)
        torch.testing.assert_close(dx1_out2, dx1_ref)
        torch.testing.assert_close(dz1_out2_, dz1_ref)
        torch.testing.assert_close(dwn0_out2, dwn0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
        torch.testing.assert_close(dwn1_out2, dwn1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
        torch.testing.assert_close(dw0_out2.T, dw0_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW0)
        torch.testing.assert_close(dw1_out2_.T, dw1_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW1)
        torch.testing.assert_close(dw2_out2.T, dw2_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DW2)
        torch.testing.assert_close(dwl_out2.T, dwl_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWL)


@torch.compiler.disable
def check_layer_pre_fwd_bwd(
    x: torch.Tensor,
    cos_sin: tuple[torch.Tensor, torch.Tensor],
    block: gpt.BlockPre,
    block_ref: gpt_ref.Block,
    eps: float,
    tols: Tolerances,
) -> None:
    cos, sin = cos_sin
    B, T, _ = x.shape
    H = block_ref.attn.n_head
    D = block_ref.attn.head_dim
    assert block_ref.attn.n_head == block_ref.attn.n_kv_head

    wn_ref = torch.randn(block.hidden_dim, dtype=x.dtype, device=x.device, requires_grad=True)

    # Ref
    x_ref = x.clone()
    h_ref = gpt_ref.norm(x, weight=wn_ref, eps=eps)
    q_ref = block_ref.attn.c_q(h_ref)
    k_ref = block_ref.attn.c_k(h_ref)
    v_ref = block_ref.attn.c_v(h_ref)
    q_rope_ref = rearrange(gpt_ref.apply_rotary_emb(rearrange(q_ref, "b t (h d) -> b t h d", h=H, d=D), cos, sin), "b t h d -> b t (h d)")
    k_rope_ref = rearrange(gpt_ref.apply_rotary_emb(rearrange(k_ref, "b t (h d) -> b t h d", h=H, d=D), cos, sin), "b t h d -> b t (h d)")
    y_ref = torch.cat([q_rope_ref, k_rope_ref, v_ref], dim=-1)
    z_ref = torch.cat([q_ref, k_ref, v_ref], dim=-1)
    rstd_ref = torch.rsqrt(reduce(x ** 2, "... d -> ...", "mean") + eps)

    # Ours
    w = interleave_rope(block.proj_qkv.weight.T, num_heads=H)
    wn = wn_ref.detach().clone().requires_grad_(True)
    cos_sin_preprocessed = gpt.preprocess_rope(cos=cos, sin=sin, batch_size=B, num_heads=H)
    cos_raw = rearrange(cos, "1 t 1 d -> t d")
    sin_raw = rearrange(sin, "1 t 1 d -> t d")
    x_out, y_out = ops.layer_pre(
        x=x,
        w=w,
        wn=wn,
        cos_sin=cos_sin_preprocessed,
        cos=cos_raw,
        sin=sin_raw,
        num_heads=H,
        head_dim=D,
        eps=eps,
        transpose=False,
        backend="torch",
        use_compile=False,
    )
    y_out = interleave_rope(y_out, num_heads=H, preprocess=False)

    grad_x = torch.randn_like(x_out) / math.sqrt(block.hidden_dim)
    grad_y = torch.randn_like(y_out) / math.sqrt(block.hidden_dim)

    (
        dx_ref, dq_ref, dk_ref, dv_ref,
        dwn_ref, dwq_ref, dwk_ref, dwv_ref,
    ) = torch.autograd.grad(
        outputs=(x_ref, y_ref),
        inputs=(
            x,
            q_ref,
            k_ref,
            v_ref,
            wn_ref,
            block_ref.attn.c_q.weight,
            block_ref.attn.c_k.weight,
            block_ref.attn.c_v.weight,
        ),
        grad_outputs=(grad_x, grad_y),
    )
    dw_ref = torch.cat([dwq_ref, dwk_ref, dwv_ref], dim=0)
    dz_ref = torch.cat([dq_ref, dk_ref, dv_ref], dim=-1)

    dx_out, dwn_out, dw_out = torch.autograd.grad(
        outputs=(x_out, y_out),
        inputs=(
            x,
            wn,
            block.proj_qkv.weight,
        ),
        grad_outputs=(grad_x, grad_y),
    )

    torch.testing.assert_close(x_out, x_ref)
    torch.testing.assert_close(y_out, y_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Y)
    torch.testing.assert_close(dx_out, dx_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DX)
    torch.testing.assert_close(dwn_out, dwn_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dw_out, dw_ref)

    y_out2, z_out2, rstd_out2 = ops.rmsnorm_gemm_rope_fwd(
        x=x,
        w=w,
        w_n=wn,
        cos_sin=cos_sin_preprocessed,
        eps=eps,
        backend="torch",
        use_quack=False,
    )
    z_out2_ = interleave_rope(z_out2, num_heads=H, preprocess=False)
    y_out2_ = interleave_rope(y_out2, num_heads=H, preprocess=False)
    torch.testing.assert_close(y_out2_, y_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Y)
    torch.testing.assert_close(z_out2_, z_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.Z)
    torch.testing.assert_close(rstd_out2, rstd_ref)

    dz_out2, zdz_out2 = ops.attn_bwd_rope_patch(
        z=z_out2,
        dy=interleave_rope(grad_y, num_heads=H),
        cos_sin=cos_sin_preprocessed,
        cos=cos_raw,
        sin=sin_raw,
        num_heads=H,
        head_dim=D,
        use_quack=False,
    )
    dx_out2, dw_out2, dwn_out2 = ops.rmsnorm_gemm_rope_bwd(
        x=x,
        dx=grad_x,
        dz=dz_out2,
        w=w,
        w_n=wn,
        rstd=rstd_out2,
        zdz_prev=zdz_out2,
        block_size=BLOCKSIZE,
        backend="torch",
    )
    dz_out2_ = interleave_rope(dz_out2, num_heads=H, preprocess=False)
    dw_out2_ = interleave_rope(dw_out2, num_heads=H, preprocess=False)
    torch.testing.assert_close(dx_out2, dx_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DX)
    torch.testing.assert_close(dz_out2_, dz_ref)
    torch.testing.assert_close(dwn_out2, dwn_ref, rtol=DEFAULT_FP32_RTOL, atol=tols.DWN)
    torch.testing.assert_close(dw_out2_.T, dw_ref)


def check_ops_fwd_bwd(
    config: gpt.GPTConfig,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype,
    eps: float,
    seed: int,
    tols: Tolerances,
) -> None:
    torch.manual_seed(seed)

    model = gpt.GPT(config).to(dtype=dtype, device="cuda")
    model_ref = gpt_ref.GPT(config).to(dtype=dtype, device="cuda")
    # model_ref.init_weights()
    convert_weights(model, model_ref)

    # model.eval()
    # model_ref.eval()

    cos_sin = (
        model.cos[:, :seq_length],
        model.sin[:, :seq_length],
    )

    block0 = model.block0
    block0_ref = model_ref.transformer.h[0]
    assert isinstance(block0, gpt.BlockPre)
    assert isinstance(block0_ref, gpt_ref.Block)
    assert block0.layer_idx == block0_ref.attn.layer_idx
    x = torch.randn((batch_size, seq_length, block0.hidden_dim), dtype=dtype, device="cuda") / math.sqrt(block0.hidden_dim)
    x = x.detach().requires_grad_(True)

    check_layer_pre_fwd_bwd(
        x=x,
        cos_sin=cos_sin,
        block=block0,
        block_ref=block0_ref,
        eps=eps,
        tols=tols,
    )

    for block, block_ref0, block_ref1 in zip(
        model.blocks,
        model_ref.transformer.h[:-1],
        model_ref.transformer.h[1:],
    ):
        assert isinstance(block, gpt.Block)
        assert isinstance(block_ref0, gpt_ref.Block)
        assert isinstance(block_ref1, gpt_ref.Block)
        assert block.layer_idx == block_ref0.attn.layer_idx + 1
        assert block.layer_idx == block_ref1.attn.layer_idx
        x = torch.randn((batch_size, seq_length, block.hidden_dim), dtype=dtype, device="cuda") / math.sqrt(block.hidden_dim)
        y = torch.randn((batch_size, seq_length, block.hidden_dim), dtype=dtype, device="cuda") / math.sqrt(block.hidden_dim)
        x = x.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)

        check_layer_fwd_bwd(
            x=x,
            y=y,
            cos_sin=cos_sin,
            block=block,
            block_ref0=block_ref0,
            block_ref1=block_ref1,
            eps=eps,
            tols=tols,
        )

    blockL = model.blockL
    blockL_ref = model_ref.transformer.h[-1]
    assert isinstance(blockL, gpt.BlockPost)
    assert isinstance(blockL_ref, gpt_ref.Block)
    assert blockL.layer_idx == blockL_ref.attn.layer_idx + 1
    x = torch.randn((batch_size, seq_length, blockL.hidden_dim), dtype=dtype, device="cuda") / math.sqrt(blockL.hidden_dim)
    y = torch.randn((batch_size, seq_length, blockL.hidden_dim), dtype=dtype, device="cuda") / math.sqrt(blockL.hidden_dim)
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    targets = torch.randint(model.config.vocab_size, (batch_size, seq_length), device="cuda")

    check_layer_post_fwd_bwd(
        x=x,
        y=y,
        targets=targets,
        block=blockL,
        block_ref=blockL_ref,
        lm_head_ref=model_ref.lm_head,
        eps=eps,
        tols=tols,
    )


def test_ops_fwd_bwd_small() -> None:
    check_models_have_same_parameter_count(
        config=GPTConfigSmall(),
    )
    check_ops_fwd_bwd(
        config=GPTConfigSmall(),
        batch_size=2,
        seq_length=512,
        dtype=torch.float32,
        eps=0.,
        seed=0,
        tols=TIGHT_FP32_TOLS_SMALL,
    )


def test_ops_fwd_bwd_bench() -> None:
    check_models_have_same_parameter_count(
        config=GPTConfig2(),
    )
    check_ops_fwd_bwd(
        config=GPTConfig2(),
        batch_size=2,
        seq_length=4096,
        dtype=torch.float32,
        eps=0.,
        seed=0,
        tols=TIGHT_FP32_TOLS_BENCH,
    )
