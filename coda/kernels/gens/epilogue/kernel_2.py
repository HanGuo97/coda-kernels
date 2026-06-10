import torch
import cutlass
import cutlass.cute as cute
from typing import Callable

from quack.cute_dsl_utils import torch2cute_dtype_map
from coda.core.epilogue import (
    EVTList,
    EVTActivationWithDualOutputs,
    EpilogueVisitorTree,
)
from coda.kernels.gens.epilogue.kernel_1 import EVTRMSNormScale


# SwiGLU: O = SiLU(G) * U where (G, U) are interleaved pairs from the accumulator.
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
_swiglu_fn = lambda g, u: g * (1.0 / (1.0 + cute.math.exp(-g, fastmath=True))) * u


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    O: torch.Tensor,
    R: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with RMSNorm scaling and SwiGLU activation.

    Composes two EVT visitors:
        1. EVTRMSNormScale: D = acc * R (per-row broadcast)
        2. EVTActivationWithDualOutputs (contraction): O = SiLU(G) * U from interleaved pairs

    Args:
        shape_mnkl: Problem shape (M, N, K, L).
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        O: Output tensor for post-activation of shape (M, N//2).
        R: Pre-computed reciprocal std dev of shape (M,) in fp32.

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    post_act_dtype = torch2cute_dtype_map[O.dtype]

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
        EVTRMSNormScale(
            acc_dtype=acc_dtype,
            tile_shape_mnk=tile_shape_mnk,
        ),
        EVTActivationWithDualOutputs(
            fn=_swiglu_fn,
            ftype="contraction",
            acc_dtype=acc_dtype,
            post_act_dtype=post_act_dtype,
            tile_shape_mnk=tile_shape_mnk,
            buffer_align_bytes=buffer_align_bytes,
        ),
    ])

    epi_args = EVTList.EpilogueArguments([
        EVTRMSNormScale.EpilogueArguments(
            mColVec=R,
        ),
        EVTActivationWithDualOutputs.EpilogueArguments(
            mPostAct=O,
        ),
    ])

    epi_keys = (
        R.dtype,
        O.dtype,
        EVTRMSNormScale,
        EVTActivationWithDualOutputs,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
