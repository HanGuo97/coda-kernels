import torch
import cutlass
import cutlass.cute as cute
from typing import Callable

from coda.core.epilogue import (
    EVTList,
    EVTPartialCrossEntropy,
    EVTSelectLogits,
    EpilogueVisitorTree,
)
from coda.kernels.gens.epilogue.kernel_1 import EVTRMSNormScale


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    R: torch.Tensor,
    targets: torch.Tensor,
    logits_tgt: torch.Tensor,
    logits_lse: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with RMSNorm, target logit selection, and partial LSE.

    Composes three EVT visitors:
        1. EVTRMSNormScale: D = acc * R (per-row broadcast)
        2. EVTSelectLogits: logits_tgt[row] = D[row, targets[row]]
        3. EVTPartialCrossEntropy: per-tile fused LSE = max + log(sum exp(x - max))

    Args:
        shape_mnkl: Problem shape (M, N, K, L).
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        R: Pre-computed reciprocal std dev of shape (M,) in fp32.
        targets: Target indices of shape (M,).
        logits_tgt: Output tensor for target logits of shape (M,).
        logits_lse: Output tensor for partial LSE of shape (M, num_blocks) in fp32.

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
        EVTRMSNormScale(
            acc_dtype=acc_dtype,
            tile_shape_mnk=tile_shape_mnk,
        ),
        EVTSelectLogits(
            dtype=cute.Int32,
            tile_shape_mnk=tile_shape_mnk,
        ),
        EVTPartialCrossEntropy(
            dtype=acc_dtype,
            tile_shape_mnk=tile_shape_mnk,
        ),
    ])

    epi_args = EVTList.EpilogueArguments([
        EVTRMSNormScale.EpilogueArguments(
            mColVec=R,
        ),
        EVTSelectLogits.EpilogueArguments(
            mTarget=targets,
            mLogits=logits_tgt,
        ),
        EVTPartialCrossEntropy.EpilogueArguments(
            mLSEVec=logits_lse,
        ),
    ])

    epi_keys = (
        R.dtype,
        targets.dtype,
        logits_tgt.dtype,
        logits_lse.dtype,
        EVTRMSNormScale,
        EVTSelectLogits,
        EVTPartialCrossEntropy,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
