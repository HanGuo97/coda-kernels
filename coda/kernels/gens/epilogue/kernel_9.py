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


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    targets: torch.Tensor,
    logits_tgt: torch.Tensor,
    logits_lse: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with target logit selection and partial LSE
    (no RMSNorm).

    Composes two EVT visitors:
        1. EVTSelectLogits: logits_tgt[row] = D[row, targets[row]]
        2. EVTPartialCrossEntropy: per-tile fused LSE = max + log(sum exp(x - max))

    Args:
        shape_mnkl: Problem shape (M, N, K, L).
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        targets: Target indices of shape (M,).
        logits_tgt: Output tensor for target logits of shape (M,).
        logits_lse: Output tensor for partial LSE of shape (M, num_blocks) in fp32.

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
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
        EVTSelectLogits.EpilogueArguments(
            mTarget=targets,
            mLogits=logits_tgt,
        ),
        EVTPartialCrossEntropy.EpilogueArguments(
            mLSEVec=logits_lse,
        ),
    ])

    epi_keys = (
        targets.dtype,
        logits_tgt.dtype,
        logits_lse.dtype,
        EVTSelectLogits,
        EVTPartialCrossEntropy,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
