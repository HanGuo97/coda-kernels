import os
import torch
from einops import rearrange
from torchtitan.models.llama3.model.args import TransformerModelArgs
from flash_attn_interface import flash_attn_qkvpacked_func
from coda.kernels.blocks.llama3 import block, block_post, block_pre

_IGNORE_INDEX = -100
_FA3_DETERMINISTIC = os.environ.get("CODA_FA3_DETERMINISTIC", "0") == "1"


class CodaRuntime(object):

    def __init__(self, model_args: TransformerModelArgs) -> None:
        self.num_heads = model_args.n_heads
        self.num_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.head_dim = model_args.dim // model_args.n_heads
        self.norm_eps = model_args.norm_eps
        self.num_layers = model_args.n_layers
        self.attention_scale = self.head_dim**-0.5

    def forward(
        self,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        layers = [
            model.layers[str(layer_id)]
            for layer_id in range(self.num_layers)
        ]

        h = model.tok_embeddings(tokens)
        h, qkv = block_pre(
            x=h,
            w=layers[0].attention.wqkv.weight,
            wn=layers[0].attention_norm.weight,
            positions=positions,
            frequencies=frequencies,
            eps=self.norm_eps,
        )
        for layer_id in range(self.num_layers - 1):
            layer = layers[layer_id]
            layer_next = layers[layer_id + 1]
            o = self._attention(qkv=qkv)
            h, qkv = block(
                x0=h,
                y0=o,
                w0=layer.attention.wo.weight,
                w1=layer.feed_forward.wgu.weight,
                w2=layer.feed_forward.wd.weight,
                w3=layer_next.attention.wqkv.weight,
                wn0=layer.ffn_norm.weight,
                wn1=layer_next.attention_norm.weight,
                positions=positions,
                frequencies=frequencies,
                eps=self.norm_eps,
            )

        layer_last = layers[-1]
        o = self._attention(qkv=qkv)
        return block_post(
            x0=h,
            y0=o,
            w0=layer_last.attention.wo.weight,
            w1=layer_last.feed_forward.wgu.weight,
            w2=layer_last.feed_forward.wd.weight,
            w3=model.output.weight,
            wn0=layer_last.ffn_norm.weight,
            wn1=model.norm.weight,
            targets=targets,
            eps=self.norm_eps,
            ignore_index=_IGNORE_INDEX,
            reduction="sum",
        )

    def _attention(self, qkv: torch.Tensor) -> torch.Tensor:
        qkv_view = rearrange(
            qkv,
            "b t (h d) -> b t h d",
            h=self.num_heads + 2 * self.num_kv_heads,
            d=self.head_dim,
        )
        o = flash_attn_qkvpacked_func(
            qkv=qkv_view,
            softmax_scale=self.attention_scale,
            causal=True,
            deterministic=_FA3_DETERMINISTIC,
            num_heads_q=self.num_heads,
        )
        return rearrange(o, "b t h d -> b t (h d)")
