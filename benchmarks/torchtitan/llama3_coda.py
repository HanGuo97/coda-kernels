import torch
from einops import rearrange
from flash_attn_interface import flash_attn_qkvpacked_func
from coda.kernels.blocks.llama3 import block, block_post, block_pre


class CodaRuntime(object):

    def forward(self, model: torch.nn.Module, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        layers = [model.layers[str(layer_id)] for layer_id in range(model.n_layers)]
        h = model.tok_embeddings(tokens)
        h, qkv = block_pre(
            x=h,
            w=layers[0].attention.wqkv.weight,
            wn=layers[0].attention_norm.weight,
            positions=positions,
            frequencies=frequencies,
            eps=self.eps,
        )
        for layer_id in range(model.n_layers - 1):
            layer = layers[layer_id]
            layer_next = layers[layer_id + 1]
            o = self._attention(
                qkv=qkv,
                layer=layer,
            )
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
                eps=self.eps,
            )

        layer_last = layers[-1]
        o = self._attention(
            qkv=qkv,
            layer=layer_last,
        )
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
            eps=self.eps,
            ignore_index=_IGNORE_INDEX,
            reduction="mean",
        )

    def _attention(self, qkv: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
        qkv_view = rearrange(
            qkv,
            "b t (h d) -> b t h d",
            h=layer.attention.n_heads + 2 * layer.attention.n_kv_heads,
            d=layer.attention.head_dim,
        )
        o = flash_attn_qkvpacked_func(
            qkv=qkv_view,
            softmax_scale=self.scale,
            causal=True,
            deterministic=_FA3_DETERMINISTIC,
            num_heads_q=layer.attention.n_heads,
        )
        return rearrange(o, "b t h d -> b t (h d)")
