from quack.gemm_sm90 import GemmSm90
from quack.activation import gate_fn_map

from coda.core.epilogue.base import compose
from coda.core.epilogue.activation import Gated
from coda.core.epilogue.lse import LSE, SelectLogits


GemmSwiGLU = (
    Gated(
        fn=gate_fn_map["swiglu"],
    )
    .bind(
        name="GemmSwiGLU",
        gemm_cls=GemmSm90,
    )
)

GemmLSE = (
    LSE()
    .bind(
        name="GemmLSE",
        gemm_cls=GemmSm90,
    )
)

GemmLSESelectLogits = (
    compose(
        [
            LSE(),
            SelectLogits(),
        ]
    )
    .bind(
        name="GemmLSESelectLogits",
        gemm_cls=GemmSm90,
    )
)
