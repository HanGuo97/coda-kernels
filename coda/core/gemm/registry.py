from quack.gemm_sm90 import GemmSm90
from quack.activation import gate_fn_map

from coda.core.epilogue.base import compose
from coda.core.epilogue.activation import Gated, RoPE
from coda.core.epilogue.lse import LSE, SelectLogits
from coda.core.epilogue.qknorm import SqSum, VStore


GemmSwiGLU = (
    Gated(
        fn=gate_fn_map["swiglu"],
    )
    .bind(
        name="GemmSwiGLU",
        gemm_cls=GemmSm90,
    )
)

GemmRoPE = (
    RoPE()
    .bind(
        name="GemmRoPE",
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

GemmQKVSqSum = (
    compose(
        [
            SqSum(),
            VStore(),
        ]
    )
    .bind(
        name="GemmQKVSqSum",
        gemm_cls=GemmSm90,
    )
)
