import torch
import cutlass
import cutlass.cute as cute

from quack.activation import dswiglu

from coda.core.ops.misc_utils import static_assert
from coda.core.gemm.gemm_interface import _kernel_op
from coda.core.elementwise.templates import _elementwise_op_tuned


@cute.jit
def _dswiglu_op(tX: cute.Tensor, tY: cute.Tensor, tZ: cute.Tensor) -> None:
    static_assert(tX.dtype == cute.Int32)
    static_assert(tZ.dtype == cute.Int32)
    static_assert(tY.dtype in (cute.Float16, cute.BFloat16))
    dtype = tY.dtype
    tX_pair = cute.recast_tensor(tX, dtype=dtype)
    tZ_pair = cute.recast_tensor(tZ, dtype=dtype)
    for i in cutlass.range_constexpr(cute.size(tY)):
        g = tX_pair[2 * i].to(dtype=cutlass.Float32)
        u = tX_pair[2 * i + 1].to(dtype=cutlass.Float32)
        dout = tY[i].to(dtype=cutlass.Float32)
        dg, du, _ = dswiglu(x=g, y=u, dout=dout)
        tZ_pair[2 * i] = dg.to(dtype=dtype)
        tZ_pair[2 * i + 1] = du.to(dtype=dtype)


@_kernel_op("coda::_dswiglu_backward", mutates_args=("Z",))
def _dswiglu_backward(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> None:
    return _elementwise_op_tuned(op=_dswiglu_op, X=X, Y=Y, Z=Z)


def dswiglu_backward(pre_act: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    assert pre_act.dtype in (torch.bfloat16, torch.float16)
    assert grad_out.dtype == pre_act.dtype
    assert pre_act.is_contiguous()
    assert grad_out.is_contiguous()
    grad_pre = torch.empty_like(pre_act)
    _dswiglu_backward(
        X=pre_act.view(dtype=torch.int32),
        Y=grad_out,
        Z=grad_pre.view(dtype=torch.int32),
    )
    return grad_pre
