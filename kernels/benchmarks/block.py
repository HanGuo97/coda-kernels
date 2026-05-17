import gc
import time
import math
import torch
import triton
import pytest
import argparse
from typing import Callable
from einops import rearrange
from quack import cache_utils
from dataclasses import dataclass

from models import ops
from models import ops2
from models.gpt import (
    GPT,
    Block,
    BlockPost,
    preprocess_rope,
)
# just for the torch compile setting
from kernels.tests import gpt as tests
from kernels.benchmarks import trainstation_utils
from kernels.benchmarks import bench_utils

cache_utils.CACHE_ENABLED = False
torch._dynamo.config.capture_scalar_outputs = True


@dataclass
class GPTConfig0:
    sequence_len: int = 8192
    vocab_size: int = 32768
    n_layer: int = 2
    n_head: int = 64
    n_kv_head: int = 64
    n_embd: int = 8192


@dataclass
class GPTConfig1:
    sequence_len: int = 8192
    vocab_size: int = 32768
    n_layer: int = 2
    n_head: int = 32
    n_kv_head: int = 32
    n_embd: int = 4096


@dataclass
class GPTConfig2:
    sequence_len: int = 8192
    vocab_size: int = 32768
    n_layer: int = 2
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 2048


def benchmark_layer(
    name: str,
    config: GPTConfig0 | GPTConfig1 | GPTConfig2,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    bench_rapier: bool,
) -> dict:
    model = GPT(config)
    model = model.to(dtype=dtype, device="cuda")
    block = model.blocks[0]
    blockL = model.blockL
    assert isinstance(block, Block)
    assert isinstance(blockL, BlockPost)
    num_heads = block.num_heads
    head_dim = block.head_dim
    cos = model.cos[:, :seq_length]
    sin = model.sin[:, :seq_length]
    cos_sin = preprocess_rope(
        cos=cos,
        sin=sin,
        batch_size=batch_size,
        num_heads=num_heads,
    )
    cos_liger  = rearrange(cos, "1 t 1 d -> t d")
    sin_liger  = rearrange(sin, "1 t 1 d -> t d")
    cos_finfer = cos_liger.float()
    sin_finfer = sin_liger.float()
    positions  = torch.arange(seq_length, device="cuda").repeat(batch_size)

    w0 = torch.randn_like(block.proj_out   .weight) / math.sqrt(block.proj_out   .in_features)
    w1 = torch.randn_like(block.proj_gateup.weight) / math.sqrt(block.proj_gateup.in_features)
    w2 = torch.randn_like(block.proj_down  .weight) / math.sqrt(block.proj_down  .in_features)
    w3 = torch.randn_like(block.proj_qkv   .weight) / math.sqrt(block.proj_qkv   .in_features)
    wl = torch.randn_like(blockL.lm_head   .weight) / math.sqrt(blockL.lm_head   .in_features)
    wn0 = torch.randn(block.hidden_dim, dtype=dtype, device="cuda")
    wn1 = torch.randn(block.hidden_dim, dtype=dtype, device="cuda")
    w0t = w0.mT.contiguous()
    w1t = w1.mT.contiguous()
    w2t = w2.mT.contiguous()
    w3t = w3.mT.contiguous()
    wlt = wl.mT.contiguous()

    x0 = torch.randn((batch_size, seq_length, block.hidden_dim), dtype=dtype, device="cuda")
    x1 = torch.randn((batch_size, seq_length, block.hidden_dim), dtype=dtype, device="cuda")
    x2 = torch.randn((batch_size, seq_length, block.hidden_dim), dtype=dtype, device="cuda")

    y0 = torch.randn((batch_size, seq_length, block.hidden_dim), dtype=dtype, device="cuda")
    y1 = torch.randn((batch_size, seq_length, block.mlp_proj_dim), dtype=dtype, device="cuda")

    z1 = torch.randn((batch_size, seq_length, w1t.shape[1]), dtype=dtype, device="cuda")
    z2 = torch.randn((batch_size, seq_length, w3t.shape[1]), dtype=dtype, device="cuda")

    rstd1 = torch.randn((batch_size, seq_length), dtype=torch.float32, device="cuda")
    rstd2 = torch.randn((batch_size, seq_length), dtype=torch.float32, device="cuda")

    zdz1 = torch.randn((batch_size, seq_length), dtype=torch.float32, device="cuda")
    zdz2 = torch.randn((batch_size, seq_length), dtype=torch.float32, device="cuda")

    targets = torch.randint(config.vocab_size, (batch_size, seq_length), device="cuda")

    x0_g = x0.detach().requires_grad_(True)
    y0_g = y0.detach().requires_grad_(True)

    w0_g = w0.detach().requires_grad_(True)
    w1_g = w1.detach().requires_grad_(True)
    w2_g = w2.detach().requires_grad_(True)
    w3_g = w3.detach().requires_grad_(True)
    wl_g = wl.detach().requires_grad_(True)
    wn0_g = wn0.detach().requires_grad_(True)
    wn1_g = wn1.detach().requires_grad_(True)

    w0t_g = w0t.detach().requires_grad_(True)
    w1t_g = w1t.detach().requires_grad_(True)
    w2t_g = w2t.detach().requires_grad_(True)
    w3t_g = w3t.detach().requires_grad_(True)
    wlt_g = wlt.detach().requires_grad_(True)

    grad_inputs = (x0_g, y0_g, w0_g, w1_g, w2_g, w3_g, wn0_g, wn1_g)
    grad_inputs_t = (x0_g, y0_g, w0t_g, w1t_g, w2t_g, w3t_g, wn0_g, wn1_g)
    grad_outputs = (
        torch.randn((batch_size, seq_length, w2t.shape[1]), dtype=dtype, device="cuda"),
        torch.randn((batch_size, seq_length, w3t.shape[1]), dtype=dtype, device="cuda"),
    )
    # trainstation outputs in different order
    grad_outputs_human = (grad_outputs[1], grad_outputs[0])

    fn0 = None
    fn0_compile = None
    fn1 = None
    fn_human = None
    fn_liger = None
    fn_liger_compile = None
    fn_liger_transpose = None
    fn_liger_compile_transpose = None
    fn_liger2 = None
    fn_liger2_compile = None
    fn_finfer = None
    fn_finfer_compile = None
    fn_torch = None
    fn_torch_transpose = None

    if name == "layer-fwd":
        fn0 = lambda: ops.layer(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            cos_sin=cos_sin,
            cos=cos_liger,
            sin=sin_liger,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=1e-6,
            transpose=True,
            backend="rapier",
            use_compile=False,
        )
        fn0_compile = lambda: ops.layer(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            cos_sin=cos_sin,
            cos=cos_liger,
            sin=sin_liger,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=1e-6,
            transpose=True,
            backend="rapier",
            use_compile=True,
        )
        fn1 = lambda: ops.layer(
            x0=x0,
            y0=y0,
            w0=w0,
            w1=w1,
            w2=w2,
            w3=w3,
            wn0=wn0,
            wn1=wn1,
            cos_sin=cos_sin,
            cos=cos_liger,
            sin=sin_liger,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=1e-6,
            transpose=True,
            backend="torch",
            use_compile=True,
        )
        fn_human = lambda: trainstation_utils.fused_transformer_block_func(
            attn_out=y0,
            residual_in=x0,
            w_attn_o=w0,
            w_mlp_gate_up=w1,
            w_mlp_down=w2,
            w_qkv=w3,
            cos_sin=cos_sin,
            cos=cos_liger,
            sin=sin_liger,
            num_heads=num_heads,
            head_dim=head_dim,
            activation="swiglu",
            residual_in_fp32=False,
            norm1_weight=wn0,
            norm2_weight=wn1,
            tuned=True,
            use_quack_gemm=False,
        )
        fn_liger = lambda: ops2.layer(
            x0=x0,
            y0=y0,
            w0=w0t,
            w1=w1t,
            w2=w2t,
            w3=w3t,
            wn0=wn0,
            wn1=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            eps=1e-6,
            transpose=False,
            backend="liger",
            use_compile=False,
        )
        fn_liger_compile = lambda: ops2.layer(
            x0=x0,
            y0=y0,
            w0=w0t,
            w1=w1t,
            w2=w2t,
            w3=w3t,
            wn0=wn0,
            wn1=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            eps=1e-6,
            transpose=False,
            backend="liger",
            use_compile=True,
        )
        fn_finfer = lambda: ops2.layer(
            x0=x0,
            y0=y0,
            w0=w0t,
            w1=w1t,
            w2=w2t,
            w3=w3t,
            wn0=wn0,
            wn1=wn1,
            cos=cos_finfer,
            sin=sin_finfer,
            positions=positions,
            eps=1e-6,
            transpose=False,
            backend="flashinfer",
            use_compile=False,
        )
        fn_finfer_compile = lambda: ops2.layer(
            x0=x0,
            y0=y0,
            w0=w0t,
            w1=w1t,
            w2=w2t,
            w3=w3t,
            wn0=wn0,
            wn1=wn1,
            cos=cos_finfer,
            sin=sin_finfer,
            positions=positions,
            eps=1e-6,
            transpose=False,
            backend="flashinfer",
            use_compile=True,
        )
        fn_torch = lambda: ops2.layer(
            x0=x0,
            y0=y0,
            w0=w0t,
            w1=w1t,
            w2=w2t,
            w3=w3t,
            wn0=wn0,
            wn1=wn1,
            cos=cos,
            sin=sin,
            positions=positions,
            eps=1e-6,
            transpose=False,
            backend="torch",
            use_compile=True,
        )

    elif name == "layer-bwd":
        if bench_rapier:
            outs0 = ops.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=1e-6,
                transpose=True,
                backend="rapier",
                use_compile=False,
            )
            outs0_compile = ops.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=1e-6,
                transpose=True,
                backend="rapier",
                use_compile=True,
            )
            fn0 = lambda: torch.autograd.grad(
                outputs=outs0,
                inputs=grad_inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            fn0_compile = lambda: torch.autograd.grad(
                outputs=outs0_compile,
                inputs=grad_inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
        else:
            outs1 = ops.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=1e-6,
                transpose=True,
                backend="torch",
                use_compile=True,
            )
            outs_human = trainstation_utils.fused_transformer_block_func(
                attn_out=y0_g,
                residual_in=x0_g,
                w_attn_o=w0_g,
                w_mlp_gate_up=w1_g,
                w_mlp_down=w2_g,
                w_qkv=w3_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                activation="swiglu",
                residual_in_fp32=False,
                norm1_weight=wn0_g,
                norm2_weight=wn1_g,
                tuned=True,
                use_quack_gemm=False,
            )
            outs_liger = ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0t_g,
                w1=w1t_g,
                w2=w2t_g,
                w3=w3t_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=False,
                backend="liger",
                use_compile=False,
            )
            outs_liger_compile = ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0t_g,
                w1=w1t_g,
                w2=w2t_g,
                w3=w3t_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=False,
                backend="liger",
                use_compile=True,
            )
            outs_liger_transpose = ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=True,
                backend="liger",
                use_compile=False,
            )
            outs_liger_compile_transpose = ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=True,
                backend="liger",
                use_compile=True,
            )
            outs_torch = ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0t_g,
                w1=w1t_g,
                w2=w2t_g,
                w3=w3t_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos,
                sin=sin,
                positions=positions,
                eps=1e-6,
                transpose=False,
                backend="torch",
                use_compile=True,
            )
            outs_torch_transpose = ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos,
                sin=sin,
                positions=positions,
                eps=1e-6,
                transpose=True,
                backend="torch",
                use_compile=True,
            )
            fn1 = lambda: torch.autograd.grad(
                outputs=outs1,
                inputs=grad_inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            fn_human = lambda: torch.autograd.grad(
                outputs=outs_human,
                inputs=grad_inputs,
                grad_outputs=grad_outputs_human,
                retain_graph=True,
            )
            fn_liger = lambda: torch.autograd.grad(
                outputs=outs_liger,
                inputs=grad_inputs_t,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            fn_liger_compile = lambda: torch.autograd.grad(
                outputs=outs_liger_compile,
                inputs=grad_inputs_t,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            fn_liger_transpose = lambda: torch.autograd.grad(
                outputs=outs_liger_transpose,
                inputs=grad_inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            fn_liger_compile_transpose = lambda: torch.autograd.grad(
                outputs=outs_liger_compile_transpose,
                inputs=grad_inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            fn_torch = lambda: torch.autograd.grad(
                outputs=outs_torch,
                inputs=grad_inputs_t,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )
            fn_torch_transpose = lambda: torch.autograd.grad(
                outputs=outs_torch_transpose,
                inputs=grad_inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
            )

    elif name == "layer-fwd-bwd":
        fn0 = lambda: torch.autograd.grad(
            outputs=ops.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=1e-6,
                transpose=True,
                backend="rapier",
                use_compile=False,
            ),
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
        )
        fn0_compile = lambda: torch.autograd.grad(
            outputs=ops.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=1e-6,
                transpose=True,
                backend="rapier",
                use_compile=True,
            ),
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
        )
        fn1 = lambda: torch.autograd.grad(
            outputs=ops.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=1e-6,
                transpose=True,
                backend="torch",
                use_compile=True,
            ),
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
        )
        fn_human = lambda: torch.autograd.grad(
            outputs=trainstation_utils.fused_transformer_block_func(
                attn_out=y0_g,
                residual_in=x0_g,
                w_attn_o=w0_g,
                w_mlp_gate_up=w1_g,
                w_mlp_down=w2_g,
                w_qkv=w3_g,
                cos_sin=cos_sin,
                cos=cos_liger,
                sin=sin_liger,
                num_heads=num_heads,
                head_dim=head_dim,
                activation="swiglu",
                residual_in_fp32=False,
                norm1_weight=wn0_g,
                norm2_weight=wn1_g,
                tuned=True,
                use_quack_gemm=False,
            ),
            inputs=grad_inputs,
            grad_outputs=grad_outputs_human,
        )
        fn_liger = lambda: torch.autograd.grad(
            outputs=ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0t_g,
                w1=w1t_g,
                w2=w2t_g,
                w3=w3t_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=False,
                backend="liger",
                use_compile=False,
            ),
            inputs=grad_inputs_t,
            grad_outputs=grad_outputs,
        )
        fn_liger_compile = lambda: torch.autograd.grad(
            outputs=ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0t_g,
                w1=w1t_g,
                w2=w2t_g,
                w3=w3t_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=False,
                backend="liger",
                use_compile=True,
            ),
            inputs=grad_inputs_t,
            grad_outputs=grad_outputs,
        )
        fn_liger_transpose = lambda: torch.autograd.grad(
            outputs=ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=True,
                backend="liger",
                use_compile=False,
            ),
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
        )
        fn_liger_compile_transpose = lambda: torch.autograd.grad(
            outputs=ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos_liger,
                sin=sin_liger,
                positions=positions,
                eps=1e-6,
                transpose=True,
                backend="liger",
                use_compile=True,
            ),
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
        )
        fn_torch = lambda: torch.autograd.grad(
            outputs=ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0t_g,
                w1=w1t_g,
                w2=w2t_g,
                w3=w3t_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos,
                sin=sin,
                positions=positions,
                eps=1e-6,
                transpose=False,
                backend="torch",
                use_compile=True,
            ),
            inputs=grad_inputs_t,
            grad_outputs=grad_outputs,
        )
        fn_torch_transpose = lambda: torch.autograd.grad(
            outputs=ops2.layer(
                x0=x0_g,
                y0=y0_g,
                w0=w0_g,
                w1=w1_g,
                w2=w2_g,
                w3=w3_g,
                wn0=wn0_g,
                wn1=wn1_g,
                cos=cos,
                sin=sin,
                positions=positions,
                eps=1e-6,
                transpose=True,
                backend="torch",
                use_compile=True,
            ),
            inputs=grad_inputs,
            grad_outputs=grad_outputs,
        )

    elif name == "fwd-none":
        fn0 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos_sin=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            backend="rapier",
            use_compile=False,
        )
        fn0_compile = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos_sin=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            backend="rapier",
            use_compile=True,
        )
        fn1 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos_sin=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            backend="torch",
            use_compile=True,
        )
        fn_liger = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=None,
            sin=None,
            positions=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            transpose=False,
            backend="liger",
            use_compile=False,
        )
        fn_liger_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=None,
            sin=None,
            positions=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            transpose=False,
            backend="liger",
            use_compile=True,
        )
        fn_finfer = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=None,
            sin=None,
            positions=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            transpose=False,
            backend="flashinfer",
            use_compile=False,
        )
        fn_finfer_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=None,
            sin=None,
            positions=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            transpose=False,
            backend="flashinfer",
            use_compile=True,
        )
        fn_torch = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=None,
            sin=None,
            positions=None,
            targets=None,
            eps=1e-6,
            epilogue=None,
            transpose=False,
            backend="torch",
            use_compile=True,
        )

    elif name == "fwd-swiglu":
        fn0 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos_sin=None,
            targets=None,
            eps=1e-6,
            epilogue="swiglu",
            backend="rapier",
            use_compile=False,
        )
        fn0_compile = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos_sin=None,
            targets=None,
            eps=1e-6,
            epilogue="swiglu",
            backend="rapier",
            use_compile=True,
        )
        fn1 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos_sin=None,
            targets=None,
            eps=1e-6,
            epilogue="swiglu",
            backend="torch",
            use_compile=True,
        )
        fn_liger = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="swiglu",
            transpose=False,
            backend="liger",
            use_compile=False,
        )
        fn_liger_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="swiglu",
            transpose=False,
            backend="liger",
            use_compile=True,
        )
        fn_finfer = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=cos_finfer,
            sin=sin_finfer,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="swiglu",
            transpose=False,
            backend="flashinfer",
            use_compile=False,
        )
        fn_finfer_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=cos_finfer,
            sin=sin_finfer,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="swiglu",
            transpose=False,
            backend="flashinfer",
            use_compile=True,
        )
        fn_torch = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x0,
            y=y0,
            w_a=w0t,
            w_b=w1t,
            w_n=wn0,
            cos=cos,
            sin=sin,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="swiglu",
            transpose=False,
            backend="torch",
            use_compile=True,
        )
        fn_human = lambda: trainstation_utils.gemm_residual_rmsnorm_gemm_fwd(
            x=y0,
            residual_in=x0,
            w_a=w0,
            w_b=w1,
            epilogue="swiglu",
            cos_sin=cos_sin,
            norm_weight=wn0,
            residual_dtype=None,
            tuned=True,
        )

    elif name == "fwd-rope":
        fn0 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos_sin=cos_sin,
            targets=None,
            eps=1e-6,
            epilogue="rope",
            backend="rapier",
            use_compile=False,
        )
        fn0_compile = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos_sin=cos_sin,
            targets=None,
            eps=1e-6,
            epilogue="rope",
            backend="rapier",
            use_compile=True,
        )
        fn1 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos_sin=cos_sin,
            targets=None,
            eps=1e-6,
            epilogue="rope",
            backend="torch",
            use_compile=True,
        )
        fn_liger = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="rope",
            transpose=False,
            backend="liger",
            use_compile=False,
        )
        fn_liger_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="rope",
            transpose=False,
            backend="liger",
            use_compile=True,
        )
        fn_finfer = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos=cos_finfer,
            sin=sin_finfer,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="rope",
            transpose=False,
            backend="flashinfer",
            use_compile=False,
        )
        fn_finfer_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos=cos_finfer,
            sin=sin_finfer,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="rope",
            transpose=False,
            backend="flashinfer",
            use_compile=True,
        )
        fn_torch = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=w3t,
            w_n=wn1,
            cos=cos,
            sin=sin,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="rope",
            transpose=False,
            backend="torch",
            use_compile=True,
        )
        fn_human = lambda: trainstation_utils.gemm_residual_rmsnorm_gemm_fwd(
            x=y1,
            residual_in=x1,
            w_a=w2,
            w_b=w3,
            epilogue="rope",
            cos_sin=cos_sin,
            norm_weight=wn1,
            residual_dtype=None,
            tuned=True,
        )

    elif name == "fwd-cross-entropy":
        fn0 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=wlt,
            w_n=wn1,
            cos_sin=None,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            backend="rapier",
            use_compile=False,
        )
        fn0_compile = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=wlt,
            w_n=wn1,
            cos_sin=None,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            backend="rapier",
            use_compile=True,
        )
        fn1 = lambda: ops.gemm_residual_rmsnorm_gemm_fwd_tunable(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=wlt,
            w_n=wn1,
            cos_sin=None,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            backend="torch",
            use_compile=True,
        )
        fn_liger = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            # liger used transposed weight for FLCE
            w_b=wl,
            w_n=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            transpose=False,
            backend="liger",
            use_compile=False,
        )
        fn_liger_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            # liger used transposed weight for FLCE
            w_b=wl,
            w_n=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            transpose=False,
            backend="liger",
            use_compile=True,
        )
        fn_liger2 = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=wlt,
            w_n=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            transpose=False,
            backend="liger2",
            use_compile=False,
        )
        fn_liger2_compile = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=wlt,
            w_n=wn1,
            cos=cos_liger,
            sin=sin_liger,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            transpose=False,
            backend="liger2",
            use_compile=True,
        )
        fn_torch = lambda: ops2.gemm_residual_rmsnorm_gemm(
            x=x1,
            y=y1,
            w_a=w2t,
            w_b=wlt,
            w_n=wn1,
            cos=cos,
            sin=sin,
            positions=positions,
            targets=targets,
            eps=1e-6,
            epilogue="cross-entropy",
            transpose=False,
            backend="torch",
            use_compile=True,
        )
        fn_human = lambda: trainstation_utils.gemm_residual_rmsnorm_gemm_fwd(
            x=y1,
            residual_in=x1,
            w_a=w2,
            w_b=wl,
            epilogue="lse",
            cos_sin=cos_sin,
            norm_weight=wn1,
            residual_dtype=None,
            target=targets,
            calculate_loss=True,
            tuned=True,
        )

    else:
        raise NotImplementedError

    if bench_rapier:
        fn_dict = {
            "rapier": fn0,
            "rapier-compile": fn0_compile,
        }
    else:
        fn_dict = {
            "ref": fn1,
            "human": fn_human,
            "liger": fn_liger,
            "liger-compile": fn_liger_compile,
            "liger-transpose": fn_liger_transpose,
            "liger-compile-transpose": fn_liger_compile_transpose,
            "liger2": fn_liger2,
            "liger2-compile": fn_liger2_compile,
            "finfer": fn_finfer,
            "finfer-compile": fn_finfer_compile,
            "torch": fn_torch,
            "torch-transpose": fn_torch_transpose,
        }
    results = bench_utils.do_bench_dict(
        fn_dict=fn_dict,
        warmup=warmup,
        repeats=repeats,
    )

    if not bench_rapier:
        return results

    del fn_dict
    del fn0
    del fn0_compile
    if name == "layer-bwd":
        del outs0
        del outs0_compile

    gc.collect()
    torch.cuda.empty_cache()

    _IGNORED_VALUE_ERRORS = {
        "CTA tile shape N must be divisible by 16 and <= 208",
        "CTA tile shape M must be 64/128/192 if pingpong",
    }

    def _maybe_skip(fn: Callable) -> None:
        try:
            fn()
        except pytest.skip.Exception:
            pass
        except ValueError as e:
            if str(e) not in _IGNORED_VALUE_ERRORS:
                raise

    for bs0 in tests.BlockSizeOptions:
        for bs1 in tests.BlockSizeOptions:
            if name == "fwd-none":
                _maybe_skip(lambda: ops.gemm_residual_rmsnorm_gemm_fwd(
                    x=x0,
                    y=y0,
                    w_a=w0t,
                    w_b=w1t,
                    w_n=wn0,
                    block_size_norm=bs0,
                    block_size_loss=bs1,
                    cos_sin=None,
                    targets=None,
                    eps=1e-6,
                    epilogue=None,
                    backend="rapier-test",
                    use_quack=False,
                ))
            if name in ("fwd-swiglu", "layer-fwd", "layer-bwd", "layer-fwd-bwd"):
                _maybe_skip(lambda: ops.gemm_residual_rmsnorm_gemm_fwd(
                    x=x0,
                    y=y0,
                    w_a=w0t,
                    w_b=w1t,
                    w_n=wn0,
                    block_size_norm=bs0,
                    block_size_loss=bs1,
                    cos_sin=None,
                    targets=None,
                    eps=1e-6,
                    epilogue="swiglu",
                    backend="rapier-test",
                    use_quack=False,
                ))
            if name in ("fwd-rope", "layer-fwd", "layer-bwd", "layer-fwd-bwd"):
                _maybe_skip(lambda: ops.gemm_residual_rmsnorm_gemm_fwd(
                    x=x1,
                    y=y1,
                    w_a=w2t,
                    w_b=w3t,
                    w_n=wn1,
                    block_size_norm=bs0,
                    block_size_loss=bs1,
                    cos_sin=cos_sin,
                    targets=None,
                    eps=1e-6,
                    epilogue="rope",
                    backend="rapier-test",
                    use_quack=False,
                ))
            if name == "fwd-cross-entropy":
                # fwd cross-entropy asserts `vocab_size % block_size_loss == 0`.
                if config.vocab_size % bs1 != 0:
                    continue
                _maybe_skip(lambda: ops.gemm_residual_rmsnorm_gemm_fwd(
                    x=x1,
                    y=y1,
                    w_a=w2t,
                    w_b=wlt,
                    w_n=wn1,
                    block_size_norm=bs0,
                    block_size_loss=bs1,
                    cos_sin=None,
                    targets=targets,
                    eps=1e-6,
                    epilogue="cross-entropy",
                    backend="rapier-test",
                    use_quack=False,
                ))
            if name in ("layer-bwd", "layer-fwd-bwd"):
                # bwd asserts `(B*T) % block_size_norm == 0`; swiglu bwd
                # additionally asserts `mlp_proj_dim % block_size_curr == 0`.
                if (batch_size * seq_length) % bs0 != 0:
                    continue
                if block.mlp_proj_dim % bs1 == 0:
                    _maybe_skip(lambda: ops.gemm_residual_rmsnorm_gemm_bwd(
                        x=x2,
                        z=z1,
                        dx=x2,
                        dz=z2,
                        w_a=w2t,
                        w_b=w3t,
                        w_n=wn1,
                        rstd=rstd2,
                        zdz_prev=zdz2,
                        block_size_prev=None,
                        block_size_curr=bs1,
                        block_size_norm=bs0,
                        epilogue="swiglu",
                        backend="rapier-test",
                    ))
                _maybe_skip(lambda: ops.gemm_residual_rmsnorm_gemm_bwd(
                    x=x1,
                    z=y0,
                    dx=x1,
                    dz=z1,
                    w_a=w0t,
                    w_b=w1t,
                    w_n=wn0,
                    rstd=rstd1,
                    zdz_prev=zdz1,
                    block_size_prev=bs1,
                    block_size_curr=None,
                    block_size_norm=bs0,
                    epilogue=None,
                    backend="rapier-test",
                ))

    return results


ALL_CONFIGS = {
    8192: GPTConfig0,
    4096: GPTConfig1,
    2048: GPTConfig2,
}


def benchmark_block_shapes(
    num: int,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> list[dict[str, dict]]:

    all_results = []
    for i in range(num + 1):
        print(f"Iteration {i}/{num}")

        results_dict = {}
        for name in [
            "layer-fwd",
            "layer-bwd",
            "layer-fwd-bwd",
            "fwd-none",
            "fwd-swiglu",
            "fwd-rope",
            "fwd-cross-entropy",
        ]:
            for size in [8192, 4096, 2048]:
                for bench_rapier in [True, False]:
                    suffix = "rapier" if bench_rapier else "other"
                    results_dict[f"{name}-{size}-{suffix}"] = benchmark_layer(
                        name=name,
                        config=ALL_CONFIGS[size](),
                        batch_size=batch_size,
                        seq_length=seq_length,
                        dtype=dtype,
                        warmup=warmup,
                        repeats=repeats,
                        bench_rapier=bench_rapier,
                    )

        if i == 0:
            time.sleep(60)
        else:
            all_results.append(results_dict)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    DEFAULT_BATCH_SIZE = 2
    DEFAULT_SEQ_LENGTH = 8192
    DEFAULT_DTYPE = torch.bfloat16
    DEFAULT_WARMUP = 5
    DEFAULT_REPEATS = 30

    results = benchmark_block_shapes(
        num=args.num,
        batch_size=DEFAULT_BATCH_SIZE,
        seq_length=DEFAULT_SEQ_LENGTH,
        dtype=DEFAULT_DTYPE,
        warmup=DEFAULT_WARMUP,
        repeats=DEFAULT_REPEATS,
    )
    torch.save(results, args.output)
