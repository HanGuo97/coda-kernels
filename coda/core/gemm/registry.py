from quack.gemm_sm90 import GemmSm90
from quack.activation import gate_fn_map

from coda.core.epilogue.activation import Gated


GemmSwiGLU = (
    Gated(
        fn=gate_fn_map["swiglu"],
    )
    .bind(
        name="GemmSwiGLU",
        gemm_cls=GemmSm90,
    )
)
