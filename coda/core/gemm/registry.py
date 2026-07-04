from quack.gemm_sm90 import GemmSm90
from quack.activation import gate_fn_map

from coda.core.epilogue.base import compose
from coda.core.epilogue.activation import Gated
from coda.core.epilogue.lse import LSE, SelectLogits, CEGrad


GemmSwiGLU = (
    Gated(
        fn=gate_fn_map["swiglu"],
    )
    .bind(
        name="GemmSwiGLU",
        gemm_cls=GemmSm90,
    )
)

GemmLinearCE = (
    compose(
        [
            SelectLogits(),
            LSE(),
        ]
    )
    .bind(
        name="GemmLinearCE",
        gemm_cls=GemmSm90,
    )
)

GemmCEGrad = (
    CEGrad()
    .bind(
        name="GemmCEGrad",
        gemm_cls=GemmSm90,
    )
)
