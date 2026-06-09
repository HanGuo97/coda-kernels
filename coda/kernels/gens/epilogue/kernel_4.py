import torch
import cutlass
import cutlass.cute as cute
from typing import Callable

from quack.cute_dsl_utils import torch2cute_dtype_map
from rapier.epilogue import (
    EVTList,
    EpilogueVisitorTree,
)
from kernels.gens.epilogue.kernel_1 import EVTRMSNormScale
from kernels.gens.epilogue.kernel_3 import EVTRoPE


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    O: torch.Tensor,
    R: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with RMSNorm scaling and RoPE rotation.

    Composes two EVT visitors:
        1. EVTRMSNormScale: D = acc * R (per-row broadcast)
        2. EVTRoPE: O = RoPE(D, cos_sin) pairwise rotation

    Args:
        shape_mnkl: Problem shape (M, N, K, L).
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        O: Output tensor for RoPE-rotated result of shape (M, N).
        R: Pre-computed reciprocal std dev of shape (M,) in fp32.
        cos_sin: Interleaved cos/sin tensor of shape (M, N).

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    epi_dtype = torch2cute_dtype_map[cos_sin.dtype]

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
        EVTRMSNormScale(
            acc_dtype=acc_dtype,
            tile_shape_mnk=tile_shape_mnk,
        ),
        EVTRoPE(
            acc_dtype=acc_dtype,
            epi_dtype=epi_dtype,
            tile_shape_mnk=tile_shape_mnk,
            buffer_align_bytes=buffer_align_bytes,
        ),
    ])

    epi_args = EVTList.EpilogueArguments([
        EVTRMSNormScale.EpilogueArguments(
            mColVec=R,
        ),
        EVTRoPE.EpilogueArguments(
            mCosSin=cos_sin,
            mOutput=O,
        ),
    ])

    epi_keys = (
        R.dtype,
        cos_sin.dtype,
        O.dtype,
        EVTRMSNormScale,
        EVTRoPE,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
