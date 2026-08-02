import os
import sys
import json
import torch
import argparse
from triton.testing import do_bench

from torchtitan.models.llama3 import Transformer, TransformerModelArgs


_BATCH = 4
_LENGTH = 8192


@torch.no_grad()
def build(seed: int) -> Transformer:
    torch.manual_seed(seed)
    model_args = TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        # llama3 1B ships GQA: 32 query heads over 8 KV heads
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=_LENGTH,
        attn_type="fa3",
    )
    with torch.device("cuda"):
        model = Transformer(model_args)
        model.init_weights()
    # parameters only: model.to(bf16) would also cast the complex freqs_cis buffer
    # and silently throw away its imaginary part
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    return model
