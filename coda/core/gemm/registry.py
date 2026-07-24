from quack.gemm_sm90 import GemmSm90
from quack.activation import gate_fn_map

from coda.core.epilogue.base import compose
from coda.core.epilogue.affine import Affine, Scale
from coda.core.epilogue.activation import Gated, RoPE
from coda.core.epilogue.lse import LSE, SelectLogits
from coda.core.epilogue.qknorm import SqSum


GemmScale = (
    Scale(
        auxiliary_store=False,
    )
    .bind(
        name="GemmScale",
        gemm_cls=GemmSm90,
    )
)

GemmSwiGLU = (
    Gated(
        fn=gate_fn_map["swiglu"],
    )
    .bind(
        name="GemmSwiGLU",
        gemm_cls=GemmSm90,
    )
)

GemmScaleSwiGLU = (
    compose(
        [
            Scale(
                auxiliary_store=False,
            ),
            Gated(
                fn=gate_fn_map["swiglu"],
            ),
        ]
    )
    .bind(
        name="GemmScaleSwiGLU",
        gemm_cls=GemmSm90,
    )
)

GemmRoPE = (
    RoPE(
        auxiliary_store=False,
    )
    .bind(
        name="GemmRoPE",
        gemm_cls=GemmSm90,
    )
)

GemmRoPEAux = (
    RoPE(
        auxiliary_store=True,
    )
    .bind(
        name="GemmRoPEAux",
        gemm_cls=GemmSm90,
    )
)

GemmScaleRoPEAux = (
    compose(
        [
            Scale(
                auxiliary_store=False,
            ),
            RoPE(
                auxiliary_store=True,
            ),
        ]
    )
    .bind(
        name="GemmScaleRoPEAux",
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

GemmScaleLSESelectLogits = (
    compose(
        [
            Scale(
                auxiliary_store=False,
            ),
            LSE(),
            SelectLogits(),
        ]
    )
    .bind(
        name="GemmScaleLSESelectLogits",
        gemm_cls=GemmSm90,
    )
)

GemmQKVSqSum = (
    SqSum()
    .bind(
        name="GemmQKVSqSum",
        gemm_cls=GemmSm90,
    )
)
