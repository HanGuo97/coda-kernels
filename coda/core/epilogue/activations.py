import cutlass.cute as cute
from quack.activation import dswiglu, swiglu
from quack.epilogue.math import F2, Pair, pack, unpack
from quack.epilogue.frontend import gemm_epilogue

EpiValue = cute.Float32 | Pair | F2
EpiOut = dict[str, EpiValue | tuple[EpiValue, EpiValue]]


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def swiglu_preact_epi(acc: EpiValue) -> EpiOut:
    gate, up = unpack(acc)
    return {"D": acc, "postact": swiglu(gate, up)}
