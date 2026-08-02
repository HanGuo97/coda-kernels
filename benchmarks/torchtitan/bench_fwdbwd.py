from torchtitan.models.llama3 import TransformerModelArgs

_BATCH = 4
_LENGTH = 8192


def llama3_1b_args() -> TransformerModelArgs:
    return TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=_LENGTH,
        attn_type="fa3",
    )
