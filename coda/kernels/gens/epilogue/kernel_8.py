import torch
import cutlass
import cutlass.cute as cute
from typing import Callable

from quack.cute_dsl_utils import torch2cute_dtype_map
from coda.core.epilogue import (
    EVTActivationWithDualOutputs,
    EpilogueVisitorTree,
)


# SwiGLU: O = SiLU(G) * U where (G, U) are interleaved pairs from the accumulator.
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
_swiglu_fn = lambda g, u: g * (1.0 / (1.0 + cute.math.exp(-g, fastmath=True))) * u


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    O: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with SwiGLU activation (no RMSNorm).

    Single EVT visitor:
        EVTActivationWithDualOutputs (contraction): O = SiLU(G) * U from interleaved pairs.

    Args:
        shape_mnkl: Problem shape (M, N, K, L).
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        O: Output tensor for post-activation of shape (M, N//2).

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    post_act_dtype = torch2cute_dtype_map[O.dtype]

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTActivationWithDualOutputs(
        fn=_swiglu_fn,
        ftype="contraction",
        acc_dtype=acc_dtype,
        post_act_dtype=post_act_dtype,
        tile_shape_mnk=tile_shape_mnk,
        buffer_align_bytes=buffer_align_bytes,
    )

    epi_args = EVTActivationWithDualOutputs.EpilogueArguments(
        mPostAct=O,
    )

    epi_keys = (
        O.dtype,
        EVTActivationWithDualOutputs,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
