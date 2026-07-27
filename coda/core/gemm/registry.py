from quack.gemm_sm90 import GemmSm90
from quack.activation import gate_fn_map

from coda.core.epilogue.base import compose
from coda.core.epilogue.affine import Residual, Scale, ScalarScale
from coda.core.epilogue.activation import Gated, RoPE
from coda.core.epilogue.lse import LSE, SelectLogits
from coda.core.epilogue.partials import SqSum
from coda.core.epilogue.qknorm import HeadSqSum
from coda.core.epilogue.swiglu_bwd import SwiGLUBwdZdZ
from coda.core.epilogue.rmsnorm_bwd import ResidualRMSNormBwd


GemmScalarScale = (
    ScalarScale()
    .bind(
        name="GemmScalarScale",
        gemm_cls=GemmSm90,
    )
)

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

GemmSwiGLUBwdZdZ = (
    SwiGLUBwdZdZ()
    .bind(
        name="GemmSwiGLUBwdZdZ",
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

GemmScaleRoPE = (
    compose(
        [
            Scale(
                auxiliary_store=False,
            ),
            RoPE(
                auxiliary_store=False,
            ),
        ]
    )
    .bind(
        name="GemmScaleRoPE",
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

GemmScaleLSE = (
    compose(
        [
            Scale(
                auxiliary_store=False,
            ),
            LSE(),
        ]
    )
    .bind(
        name="GemmScaleLSE",
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
    HeadSqSum()
    .bind(
        name="GemmQKVSqSum",
        gemm_cls=GemmSm90,
    )
)

GemmResidualSqSumScaledAux = (
    compose(
        [
            Residual(),
            SqSum(),
            Scale(
                auxiliary_store=True,
                row_name="mRowVecScale",
                col_name="mColVecScale",
            ),
        ]
    )
    .bind(
        name="GemmResidualSqSumScaledAux",
        gemm_cls=GemmSm90,
    )
)

GemmScalarScaleResidualRMSNormBwd = (
    compose(
        [
            ScalarScale(),
            ResidualRMSNormBwd(),
        ]
    )
    .bind(
        name="GemmScalarScaleResidualRMSNormBwd",
        gemm_cls=GemmSm90,
    )
)

GemmScalarScaleSwiGLU = (
    compose(
        [
            ScalarScale(),
            Gated(
                fn=gate_fn_map["swiglu"],
            ),
        ]
    )
    .bind(
        name="GemmScalarScaleSwiGLU",
        gemm_cls=GemmSm90,
    )
)

GemmScalarScaleRoPE = (
    compose(
        [
            ScalarScale(),
            RoPE(
                auxiliary_store=False,
            ),
        ]
    )
    .bind(
        name="GemmScalarScaleRoPE",
        gemm_cls=GemmSm90,
    )
)
