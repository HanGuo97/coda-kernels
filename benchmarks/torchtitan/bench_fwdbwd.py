import os
import sys
import json
import torch
import argparse
from typing import Callable
from triton.testing import do_bench
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.llama3 import Transformer, TransformerModelArgs

from benchmarks.torchtitan import liger_utils


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


def bf16_cross_entropy_loss(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1),
        targets.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def make_forward(
    name: str,
    model: Transformer,
    positions: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name == "coda":
        def _forward(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return model(tokens=tokens, targets=targets, positions=positions)
        return _forward

    if name == "liger":
        def _forward(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return liger_utils.cross_entropy(pred=model(tokens), targets=targets)
        return _forward

    if name == "torch":
        def _forward(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return bf16_cross_entropy_loss(pred=model(tokens), targets=targets)
        return _forward

    raise NotImplementedError
