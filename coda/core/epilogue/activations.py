import cutlass.cute as cute
from quack.activation import dswiglu, swiglu
from quack.epilogue.math import pack, unpack
from quack.epilogue.frontend import gemm_epilogue


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def swiglu_preact_epi(acc):
    gate, up = unpack(acc)
    return {"D": acc, "postact": swiglu(gate, up)}
