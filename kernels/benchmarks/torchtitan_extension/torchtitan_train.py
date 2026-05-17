import runpy

from torchtitan.models.llama3 import llama3_args, TransformerModelArgs

llama3_args["1B"] = TransformerModelArgs(
    dim=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.5,
    multiple_of=1024,
    rope_theta=500000,
)

runpy.run_module("torchtitan.train", run_name="__main__")
