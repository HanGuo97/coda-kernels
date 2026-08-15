import cutlass.cute as cute
from quack.activation import dswiglu, swiglu
from quack.epilogue.math import F2, Pair, pack, unpack
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import (
    Scalar,
    ColVecLoad,
)

EpiValue = cute.Float32 | Pair | F2
EpiOut = dict[str, EpiValue | tuple[EpiValue, EpiValue]]


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def swiglu_preact_epi(acc: EpiValue) -> EpiOut:
    gate, up = unpack(acc)
    return {"D": acc, "postact": swiglu(gate, up)}


@gemm_epilogue(ops={"rstd": ColVecLoad("rstd")})
def rstd_epi(acc: EpiValue, rstd: EpiValue) -> EpiOut:
    return {"D": acc * rstd}


@gemm_epilogue(ops={"alpha": Scalar("alpha")})
def alpha_epi(acc: EpiValue, alpha: EpiValue) -> EpiOut:
    return {"D": acc * alpha}
