import torch
from einops import repeat, rearrange
from flash_attn import flash_attn_func

from models.gpt_ref import (
    GPTConfig,
    norm,
    round_up_to_next_multiple,
)
from models.ops import (
    layer,
    layer_pre,
    layer_post,
)


def preprocess_rope(
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor:
    cos = repeat(cos, "1 t 1 d -> (b t) (h d)", b=batch_size, h=num_heads)
    sin = repeat(sin, "1 t 1 d -> (b t) (h d)", b=batch_size, h=num_heads)
    cos = torch.stack([cos.clone(), cos.clone(), torch.ones_like(cos)], dim=-1)
    sin = torch.stack([sin.clone(), sin.clone(), torch.zeros_like(sin)], dim=-1)
    cos = rearrange(cos, "m n trio -> m (trio n)", trio=3)
    sin = rearrange(sin, "m n trio -> m (trio n)", trio=3)
    cos_sin = torch.stack([cos, sin], dim=-1)
    cos_sin = rearrange(cos_sin, "... n pair -> ... (n pair)", pair=2)
    return cos_sin


class _Block(torch.nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.num_heads    = config.n_head
        self.num_kv_heads = config.n_kv_head

        self.hidden_dim = config.n_embd
        self.head_dim   = self.hidden_dim // self.num_heads
        self.q_dim      = self.num_heads * self.head_dim
        self.kv_dim     = self.num_kv_heads * self.head_dim

        # https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/mlp.py
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        self.qkv_proj_dim = self.q_dim + self.kv_dim * 2
        self.mlp_proj_dim = round_up_to_next_multiple(int(self.hidden_dim * 4 * 2 / 3), 256)

        assert self.num_heads == self.num_kv_heads
        assert self.hidden_dim % self.num_heads == 0
        assert self.num_kv_heads <= self.num_heads and self.num_heads % self.num_kv_heads == 0


class BlockPre(_Block):

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.proj_qkv = torch.nn.Linear(self.hidden_dim, self.qkv_proj_dim, bias=False)

    def forward(self, idx: torch.Tensor, cos_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(idx)
        # x = norm(x)
        x, qkv = layer_pre(
            x=x,
            w=self.proj_qkv.weight,
            cos_sin=cos_sin,
            transpose=True,
        )
        return qkv, x


class BlockPost(_Block):

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.proj_out    = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.proj_gateup = torch.nn.Linear(self.hidden_dim, self.mlp_proj_dim * 2, bias=False)
        self.proj_down   = torch.nn.Linear(self.mlp_proj_dim, self.hidden_dim, bias=False)
        self.lm_head     = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, qkv: torch.Tensor, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Attention
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads, d=self.head_dim)

        o = flash_attn_func(q, k, v, causal=True)
        o = rearrange(o, "b t h d -> b t (h d)")
        loss = layer_post(
            x0=x,
            y0=o,
            w0=self.proj_out.weight,
            w1=self.proj_gateup.weight,
            w2=self.proj_down.weight,
            w3=self.lm_head.weight,
            targets=targets,
            transpose=True,
        )
        return loss


class Block(_Block):

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.proj_qkv    = torch.nn.Linear(self.hidden_dim, self.qkv_proj_dim, bias=False)
        self.proj_out    = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.proj_gateup = torch.nn.Linear(self.hidden_dim, self.mlp_proj_dim * 2, bias=False)
        self.proj_down   = torch.nn.Linear(self.mlp_proj_dim, self.hidden_dim, bias=False)

    def forward(self, qkv: torch.Tensor, x: torch.Tensor, cos_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads, d=self.head_dim)

        o = flash_attn_func(q, k, v, causal=True)
        o = rearrange(o, "b t h d -> b t (h d)")

        x, qkv = layer(
            x0=x,
            y0=o,
            w0=self.proj_out.weight,
            w1=self.proj_gateup.weight,
            w2=self.proj_down.weight,
            w3=self.proj_qkv.weight,
            cos_sin=cos_sin,
            transpose=True,
        )
        return qkv, x


class GPT(torch.nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.block0 = BlockPre(config, 0)
        self.blockL = BlockPost(config, config.n_layer)
        self.blocks = torch.nn.ModuleList([
            Block(config, layer_idx)
            for layer_idx in range(1, config.n_layer)
        ])

        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 10000,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # autodetect the device from model embeddings
        if device is None:
            device = self.block0.embedding.weight.device

        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # keep them in bfloat16/float16
        # cos = cos.to(dtype=DEFAULT_DTYPE)
        # sin = sin.to(dtype=DEFAULT_DTYPE)
        # add batch and head dims for later broadcasting
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.cos.size(1)
        assert idx.device == self.cos.device
        # assert self.cos.dtype == DEFAULT_DTYPE
        # truncate cache to current sequence length
        cos_sin = preprocess_rope(
            cos=self.cos[:, :T],
            sin=self.sin[:, :T],
            batch_size=B,
            num_heads=self.config.n_head,
        )

        # Forward
        qkv, x = self.block0(idx, cos_sin)
        for block in self.blocks:
            qkv, x = block(qkv, x, cos_sin)
        loss = self.blockL(qkv, x, targets)
        return loss
