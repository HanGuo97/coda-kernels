import cutlass.cute as cute
from quack.activation import dswiglu, swiglu
# Re-exported unchanged
from quack.epilogue.library import amax_epi, lse_epi, rstd_lse_epi
from quack.epilogue.math import F2, Pair, pack, unpack
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import (
    Scalar,
    ColVecLoad,
    ColVecReduce,
    RowVecLoad,
)


EpiValue = cute.Float32 | Pair | F2
EpiOut = dict[str, EpiValue | tuple[EpiValue, EpiValue]]


@gemm_epilogue(ops={"rstd": ColVecLoad("rstd")})
def rstd_epi(acc: EpiValue, rstd: EpiValue) -> EpiOut:
    return {"D": acc * rstd}


@gemm_epilogue(ops={"alpha": Scalar("alpha")})
def alpha_epi(acc: EpiValue, alpha: EpiValue) -> EpiOut:
    return {"D": acc * alpha}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def swiglu_preact_epi(acc: EpiValue) -> EpiOut:
    gate, up = unpack(acc)
    return {"D": acc, "postact": swiglu(gate, up)}


@gemm_epilogue(
    outputs=("postact",),
    ops={"rstd": ColVecLoad("rstd")},
    mode="acc_pair",
)
def rstd_swiglu_scaled_preact_epi(acc: EpiValue, rstd: EpiValue) -> EpiOut:
    scaled = acc * rstd
    gate, up = unpack(scaled)
    return {"D": scaled, "postact": swiglu(gate, up)}


@gemm_epilogue(
    outputs=("postact",),
    ops={"alpha": Scalar("alpha")},
    mode="acc_pair",
)
def alpha_swiglu_preact_epi(acc: EpiValue, alpha: EpiValue) -> EpiOut:
    scaled = acc * alpha
    gate, up = unpack(scaled)
    return {"D": scaled, "postact": swiglu(gate, up)}


@gemm_epilogue(
    outputs=("scaled_out",),
    ops={"weight": RowVecLoad("weight")},
    reduces={"sqsum": ColVecReduce("sqsum", scaled=True)},
)
def residual_sqsum_scaled_epi(acc: EpiValue, c: EpiValue, weight: EpiValue) -> EpiOut:
    y = acc + c
    o = y * weight
    return {"D": y, "scaled_out": o, "sqsum": (y, y)}


@gemm_epilogue(
    outputs=("scaled_out",),
    ops={"weight": RowVecLoad("weight"), "alpha": Scalar("alpha")},
    reduces={"sqsum": ColVecReduce("sqsum", scaled=True)},
)
def alpha_residual_sqsum_scaled_epi(acc: EpiValue, c: EpiValue, weight: EpiValue, alpha: EpiValue) -> EpiOut:
    y = acc * alpha + c
    o = y * weight
    return {"D": y, "scaled_out": o, "sqsum": (y, y)}


@gemm_epilogue(
    outputs=("postact",),
    reduces={"zdz": ColVecReduce("zdz")},
    mode="packed_cd_b16x2",
)
def dswiglu_preact_zdz_epi(acc: EpiValue, c: EpiValue) -> EpiOut:
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc)
    zdz = dx * x + dy * y
    return {"D": pack(dx, dy), "postact": out, "zdz": zdz}


@gemm_epilogue(
    outputs=("postact",),
    ops={"scale": Scalar("scale")},
    reduces={"zdz": ColVecReduce("zdz")},
    mode="packed_cd_b16x2",
)
def dswiglu_preact_zdz_scaled_epi(acc: EpiValue, c: EpiValue, scale: EpiValue) -> EpiOut:
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc)
    zdz = (dx * x + dy * y) * scale
    return {"D": pack(dx, dy), "postact": out, "zdz": zdz}
