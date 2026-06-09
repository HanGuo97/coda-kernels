# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import warnings
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from typing import Callable
from quack.cute_dsl_utils import torch2cute_dtype_map
from rapier.ops.reduction_utils import get_registered_reduction_op
from rapier.epilogue import (
    EVTNoOp,
    EVTList,
    EVTRowOrColBias,
    EVTRowOrColBlockReductionLoad,
    EVTColBlockReductionStore,
    EVTColBlockReductionStore2X,
    EVTRowBlockReductionStore,
    EVTPartialCrossEntropy,
    EVTSelectLogits,
    EVTActivationWithDualOutputs,
    EVTResidual,
    EVTMatrixLoad2X,
    EpilogueVisitorTree,
)
from rapier.gemm.gemm_interface import (
    DEFAULT_PINGPONG,
    DEFAULT_TILE_M,
    DEFAULT_TILE_N,
    DEFAULT_CLUSTER_M,
    DEFAULT_CLUSTER_N,
    preprocess_vector,
    preprocess_tensor,
    gemm_epilogue,
    gemm_epilogue_tuned,
)


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    bias_row: torch.Tensor | None,
    bias_col: torch.Tensor | None,
    bias_rbk: torch.Tensor | None,
    bias_cbk: torch.Tensor | None,
    warp_row: torch.Tensor | None,
    warp_col: torch.Tensor | None,
    warp_lse: torch.Tensor | None,
    post_act: torch.Tensor | None,
    residual: torch.Tensor | None,
    reduction: str | None,
    activation: str | None,
    target_index: torch.Tensor | None,
    target_logit: torch.Tensor | None,
) -> tuple[Callable[[object], EpilogueVisitorTree],
           EpilogueVisitorTree.EpilogueArguments,
           dict,
           tuple]:

    M, N, K, L = shape_mnkl
    assert bias_row is None or bias_row.shape == (L, N)
    assert bias_col is None or bias_col.shape == (L, M)
    assert bias_rbk is None or bias_rbk.shape == (L, N, (M + tile_shape_mn[0] - 1) // tile_shape_mn[0])
    assert bias_cbk is None or bias_cbk.shape == (L, M, (N + tile_shape_mn[1] - 1) // tile_shape_mn[1])
    assert warp_row is None or warp_row.shape in ((L, N, (M + tile_shape_mn[0] - 1) // tile_shape_mn[0]), (L, N, (2 * M + tile_shape_mn[0] - 1) // tile_shape_mn[0]))
    assert warp_col is None or warp_col.shape in ((L, M, (N + tile_shape_mn[1] - 1) // tile_shape_mn[1]), (L, M, (2 * N + tile_shape_mn[1] - 1) // tile_shape_mn[1]))
    assert warp_lse is None or warp_lse.shape == (L, M, (N + tile_shape_mn[1] - 1) // tile_shape_mn[1])
    assert post_act is None or post_act.shape in ((M, N, L), (M, N // 2, L))
    assert residual is None or residual.shape in ((M, N, L), (M, N * 2, L))
    assert target_index is None or target_index.shape == (L, M)
    assert target_logit is None or target_logit.shape == (L, M)

    # bias_row_tensor = to_cute_vector(bias_row) if bias_row is not None else None
    # bias_col_tensor = to_cute_vector(bias_col) if bias_col is not None else None
    # bias_rbk_tensor = to_cute_tensor(bias_rbk, assumed_align=32) if bias_rbk is not None else None
    # bias_cbk_tensor = to_cute_tensor(bias_cbk, assumed_align=32) if bias_cbk is not None else None
    # warp_row_tensor = to_cute_tensor(warp_row, assumed_align=4) if warp_row is not None else None
    # warp_col_tensor = to_cute_tensor(warp_col, assumed_align=4) if warp_col is not None else None
    # warp_lse_tensor = to_cute_tensor(warp_lse, assumed_align=4) if warp_lse is not None else None
    # post_act_tensor = to_cute_tensor(post_act) if post_act is not None else None
    # residual_tensor = to_cute_tensor(residual) if residual is not None else None
    # target_index_tensor = to_cute_vector(target_index) if target_index is not None else None
    # target_logit_tensor = to_cute_vector(target_logit) if target_logit is not None else None

    epi_keys = (
        bias_row.dtype if bias_row is not None else None,
        bias_col.dtype if bias_col is not None else None,
        bias_rbk.dtype if bias_rbk is not None else None,
        bias_cbk.dtype if bias_cbk is not None else None,
        warp_row.dtype if warp_row is not None else None,
        warp_col.dtype if warp_col is not None else None,
        warp_lse.dtype if warp_lse is not None else None,
        post_act.dtype if post_act is not None else None,
        residual.dtype if residual is not None else None,
        target_index.dtype if target_index is not None else None,
        target_logit.dtype if target_logit is not None else None,
        reduction,
        activation,
    )

    if all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        warp_col is None,
        warp_lse is None,
        post_act is None,
        residual is None,
        reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTNoOp()
        )
        epi_args = EVTNoOp.EpilogueArguments()
        epi_keys = (*epi_keys, EVTNoOp)

    elif all([
        # bias_row is None,
        # bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        warp_col is None,
        warp_lse is None,
        post_act is None,
        residual is None,
        reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTRowOrColBias(
                acc_dtype=acc_dtype,
                tile_shape_mnk=tile_shape_mnk,
            )
        )
        epi_args = EVTRowOrColBias.EpilogueArguments(
            mRowVec=bias_row,
            mColVec=bias_col,
        )
        epi_keys = (*epi_keys, EVTRowOrColBias)

    elif all([
        # bias_row is None,
        # bias_col is None,
        # bias_rbk is None,
        # bias_cbk is None,
        warp_row is None,
        warp_col is None,
        warp_lse is None,
        post_act is None,
        residual is None,
        # reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        assert reduction is not None
        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTRowOrColBlockReductionLoad(
                acc_dtype=acc_dtype,
                reduction_op=reduction,
                tile_shape_mnk=tile_shape_mnk,
            )
        )
        epi_args = EVTRowOrColBlockReductionLoad.EpilogueArguments(
            # load
            mRowBlkLd=bias_rbk,
            mColBlkLd=bias_cbk,
            # store
            mRowVecSt=bias_row,
            mColVecSt=bias_col,
        )
        epi_keys = (*epi_keys, EVTRowOrColBlockReductionLoad)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        # warp_col is None,
        warp_lse is None,
        post_act is None,
        residual is None,
        # reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        assert warp_col is not None
        assert reduction is not None

        if warp_col.shape == (L, M, (N + tile_shape_mn[1] - 1) // tile_shape_mn[1]):
            EVTColBlockReductionStoreCls = EVTColBlockReductionStore
        elif warp_col.shape == (L, M, (2 * N + tile_shape_mn[1] - 1) // tile_shape_mn[1]):
            EVTColBlockReductionStoreCls = EVTColBlockReductionStore2X
        else:
            raise NotImplementedError

        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTColBlockReductionStoreCls(
                reduction_op=get_registered_reduction_op(
                    name=reduction,
                    element_type=acc_dtype,
                ),
                tile_shape_mnk=tile_shape_mnk,
            )
        )
        epi_args = EVTColBlockReductionStoreCls.EpilogueArguments(
            mColVec=warp_col,
        )
        epi_keys = (*epi_keys, EVTColBlockReductionStoreCls)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        # warp_row is None,
        warp_col is None,
        warp_lse is None,
        post_act is None,
        residual is None,
        # reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        assert warp_row is not None
        assert reduction is not None

        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTRowBlockReductionStore(
                reduction_op=get_registered_reduction_op(
                    name=reduction,
                    element_type=acc_dtype,
                ),
                tile_shape_mnk=tile_shape_mnk,
            )
        )
        epi_args = EVTRowBlockReductionStore.EpilogueArguments(
            mRowVec=warp_row,
        )
        epi_keys = (*epi_keys, EVTRowBlockReductionStore)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        # warp_row is None,
        # warp_col is None,
        warp_lse is None,
        post_act is None,
        residual is None,
        # reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        assert warp_row is not None
        assert warp_col is not None
        assert reduction is not None
        if isinstance(reduction, tuple):
            row_reduction, col_reduction = reduction
        else:
            row_reduction = reduction
            col_reduction = reduction
        if warp_col.shape == (L, M, (N + tile_shape_mn[1] - 1) // tile_shape_mn[1]):
            EVTColBlockReductionStoreCls = EVTColBlockReductionStore
        elif warp_col.shape == (L, M, (2 * N + tile_shape_mn[1] - 1) // tile_shape_mn[1]):
            EVTColBlockReductionStoreCls = EVTColBlockReductionStore2X
        else:
            raise NotImplementedError

        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
                EVTRowBlockReductionStore(
                    reduction_op=get_registered_reduction_op(
                        name=row_reduction,
                        element_type=acc_dtype,
                    ),
                    tile_shape_mnk=tile_shape_mnk,
                ),
                EVTColBlockReductionStoreCls(
                    reduction_op=get_registered_reduction_op(
                        name=col_reduction,
                        element_type=acc_dtype,
                    ),
                    tile_shape_mnk=tile_shape_mnk,
                ),
            ])
        )
        epi_args = EVTList.EpilogueArguments([
            EVTRowBlockReductionStore.EpilogueArguments(
                mRowVec=warp_row,
            ),
            EVTColBlockReductionStoreCls.EpilogueArguments(
                mColVec=warp_col,
            ),
        ])
        epi_keys = (*epi_keys, EVTRowBlockReductionStore, EVTColBlockReductionStoreCls)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        warp_col is None,
        # warp_lse is None,
        post_act is None,
        residual is None,
        reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTPartialCrossEntropy(
                dtype=acc_dtype,
                tile_shape_mnk=tile_shape_mnk,
            )
        )
        epi_args = EVTPartialCrossEntropy.EpilogueArguments(
            mLSEVec=warp_lse,
        )
        epi_keys = (*epi_keys, EVTPartialCrossEntropy)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        warp_col is None,
        warp_lse is None,
        post_act is None,
        residual is None,
        reduction is None,
        activation is None,
        # target_index is None,
        # target_logit is None,
    ]):
        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTSelectLogits(
                dtype=cute.Int32,
                tile_shape_mnk=tile_shape_mnk,
            )
        )
        epi_args = EVTSelectLogits.EpilogueArguments(
            mTarget=target_index,
            mLogits=target_logit,
        )
        epi_keys = (*epi_keys, EVTSelectLogits)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        warp_col is None,
        warp_lse is None,
        # post_act is None,
        residual is None,
        reduction is None,
        # activation is None,
        target_index is None,
        target_logit is None,
    ]):
        activations_dict = {
            "elementwise": lambda x: cute.math.tanh(x),
            "pairwise"   : lambda x0, x1: (0.7 * x1, 0.3 * x0),
            "contraction": lambda x0, x1: 0.3 * x0 + 0.7 * x1,
            "expansion"  : lambda x: (0.3 * x, 0.7 * x),
        }
        if activation == "contraction":
            assert post_act is None or post_act.shape == (M, N // 2, L)
        else:
            assert post_act is None or post_act.shape == (M, N, L)

        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTActivationWithDualOutputs(
                fn=activations_dict[activation],
                ftype=activation,
                acc_dtype=acc_dtype,
                post_act_dtype=cute.Float16,
                tile_shape_mnk=tile_shape_mnk,
                buffer_align_bytes=buffer_align_bytes,
            )
        )
        epi_args = EVTActivationWithDualOutputs.EpilogueArguments(
            mPostAct=post_act,
        )
        epi_keys = (*epi_keys, EVTActivationWithDualOutputs)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        warp_col is None,
        warp_lse is None,
        post_act is None,
        # residual is None,
        reduction is None,
        activation is None,
        target_index is None,
        target_logit is None,
    ]):
        if residual.shape == (M, N, L):
            EVTMatrixLoadCls = EVTResidual
            evt_kwargs = {}
        elif residual.shape == (M, N * 2, L):
            EVTMatrixLoadCls = EVTMatrixLoad2X
            evt_kwargs = {"fn": lambda a, b0, b1: a + 0.3 * b0 - 0.7 * b1}
        else:
            raise NotImplementedError

        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTMatrixLoadCls(
                acc_dtype=acc_dtype,
                epi_dtype=torch2cute_dtype_map[residual.dtype],
                tile_shape_mnk=tile_shape_mnk,
                buffer_align_bytes=buffer_align_bytes,
                **evt_kwargs,
            )
        )
        epi_args = EVTMatrixLoadCls.EpilogueArguments(
            mMatrix=residual,
        )
        epi_keys = (*epi_keys, EVTMatrixLoadCls)

    elif all([
        bias_row is None,
        bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        warp_row is None,
        warp_col is None,
        # warp_lse is None,
        post_act is None,
        residual is None,
        reduction is None,
        activation is None,
        # target_index is None,
        # target_logit is None,
    ]):
        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
                EVTPartialCrossEntropy(
                    dtype=acc_dtype,
                    tile_shape_mnk=tile_shape_mnk,
                ),
                EVTSelectLogits(
                    dtype=cute.Int32,
                    tile_shape_mnk=tile_shape_mnk,
                ),
            ])
        )
        epi_args = EVTList.EpilogueArguments([
            EVTPartialCrossEntropy.EpilogueArguments(
                mLSEVec=warp_lse,
            ),
            EVTSelectLogits.EpilogueArguments(
                mTarget=target_index,
                mLogits=target_logit,
            ),
        ])
        epi_keys = (*epi_keys, EVTPartialCrossEntropy, EVTSelectLogits)

    elif all([
        # bias_row is None,
        # bias_col is None,
        bias_rbk is None,
        bias_cbk is None,
        # warp_row is None,
        # warp_col is None,
        warp_lse is None,
        # post_act is None,
        # residual is None,
        # reduction is None,
        # activation is None,
        target_index is None,
        target_logit is None,
    ]):

        activations_dict = {
            "elementwise": lambda x: cute.math.tanh(x),
            "pairwise"   : lambda x0, x1: (0.7 * x1, 0.3 * x0),
            "contraction": lambda x0, x1: 0.3 * x0 + 0.7 * x1,
            "expansion"  : lambda x: (0.3 * x, 0.7 * x),
        }
        if activation == "contraction":
            assert post_act is None or post_act.shape == (M, N // 2, L)
        else:
            assert post_act is None or post_act.shape == (M, N, L)
        assert warp_row is None
        assert warp_col.shape == (L, M, (N + tile_shape_mn[1] - 1) // tile_shape_mn[1])
        assert residual.shape == (M, N, L)

        epi_cls = (
            lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
                EVTRowOrColBias(
                    acc_dtype=acc_dtype,
                    tile_shape_mnk=tile_shape_mnk,
                ),
                EVTColBlockReductionStore(
                    reduction_op=get_registered_reduction_op(
                        name=reduction,
                        element_type=acc_dtype,
                    ),
                    tile_shape_mnk=tile_shape_mnk,
                ),
                EVTActivationWithDualOutputs(
                    fn=activations_dict[activation],
                    ftype=activation,
                    acc_dtype=acc_dtype,
                    post_act_dtype=cute.Float16,
                    tile_shape_mnk=tile_shape_mnk,
                    buffer_align_bytes=buffer_align_bytes,
                ),
                EVTResidual(
                    acc_dtype=acc_dtype,
                    epi_dtype=torch2cute_dtype_map[residual.dtype],
                    tile_shape_mnk=tile_shape_mnk,
                    buffer_align_bytes=buffer_align_bytes,
                ),
            ])
        )
        epi_args = EVTList.EpilogueArguments([
            EVTRowOrColBias.EpilogueArguments(
                mRowVec=bias_row,
                mColVec=bias_col,
            ),
            EVTColBlockReductionStore.EpilogueArguments(
                mColVec=warp_col,
            ),
            EVTActivationWithDualOutputs.EpilogueArguments(
                mPostAct=post_act,
            ),
            EVTResidual.EpilogueArguments(
                mMatrix=residual,
            ),
        ])
        epi_keys = (*epi_keys, EVTRowOrColBias, EVTColBlockReductionStore, EVTActivationWithDualOutputs, EVTResidual)

    else:
        raise NotImplementedError

    epi_outs = {}
    if warp_row is not None:
        # warp_row = torch.permute(warp_row, dims=(1, 0))
        warp_row = torch.squeeze(warp_row, dim=0)
        epi_outs["warp_row"] = warp_row
    if warp_col is not None:
        # warp_col = torch.permute(warp_col, dims=(1, 0))
        warp_col = torch.squeeze(warp_col, dim=0)
        epi_outs["warp_col"] = warp_col
    if warp_lse is not None:
        # warp_lse = torch.permute(warp_lse, dims=(1, 0))
        warp_lse = torch.squeeze(warp_lse, dim=0)
        epi_outs["warp_lse"] = warp_lse
    if post_act is not None:
        post_act = torch.permute(post_act, dims=(2, 0, 1))
        post_act = torch.squeeze(post_act, dim=0)
        epi_outs["post_act"] = post_act
    if target_logit is not None:
        # target_logit = torch.permute(target_logit, dims=(1, 0))
        target_logit = torch.squeeze(target_logit, dim=0)
        epi_outs["target_logit"] = target_logit

    return epi_cls, epi_args, epi_outs, epi_keys


def gemm_op(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
    bias_row: torch.Tensor | None = None,
    bias_col: torch.Tensor | None = None,
    bias_rbk: torch.Tensor | None = None,
    bias_cbk: torch.Tensor | None = None,
    warp_row: torch.Tensor | None = None,
    warp_col: torch.Tensor | None = None,
    warp_lse: torch.Tensor | None = None,
    post_act: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    reduction: str | None = None,
    activation: str | None = None,
    target_index: torch.Tensor | None = None,
    target_logit: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    # Enable autotuning for optimal performance.
    # Keep False during development for faster iteration.
    use_tuned: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    M, K = A.shape
    _, N = B.shape

    if out_dtype is None:
        out_dtype = A.dtype
    if out is None:
        out = torch.empty(
            (M, N),
            dtype=out_dtype,
            device=A.device,
        )
        add_to_output = False
    else:
        assert out.dtype is out_dtype
        add_to_output = True

    A = preprocess_tensor(A, permute=True, transpose=False)
    B = preprocess_tensor(B, permute=True, transpose=True)
    out = preprocess_tensor(out, permute=True, transpose=False)

    # we keep `l` for biases in the first dimension
    bias_row = preprocess_vector(bias_row, permute=False) if bias_row is not None else None
    bias_col = preprocess_vector(bias_col, permute=False) if bias_col is not None else None
    bias_rbk = preprocess_tensor(bias_rbk, permute=False) if bias_rbk is not None else None
    bias_cbk = preprocess_tensor(bias_cbk, permute=False) if bias_cbk is not None else None
    warp_row = preprocess_tensor(warp_row, permute=False) if warp_row is not None else None
    warp_col = preprocess_tensor(warp_col, permute=False) if warp_col is not None else None
    warp_lse = preprocess_tensor(warp_lse, permute=False) if warp_lse is not None else None
    post_act = preprocess_tensor(post_act, permute=True) if post_act is not None else None
    residual = preprocess_tensor(residual, permute=True) if residual is not None else None
    target_index = preprocess_vector(target_index, permute=False) if target_index is not None else None
    target_logit = preprocess_vector(target_logit, permute=False) if target_logit is not None else None

    epi_cls, epi_args, epi_outs, epi_keys = prepare_epilogue(
        shape_mnkl=(M, N, K, 1),
        tile_shape_mn=(DEFAULT_TILE_M, DEFAULT_TILE_N),
        bias_row=bias_row,
        bias_col=bias_col,
        bias_rbk=bias_rbk,
        bias_cbk=bias_cbk,
        warp_row=warp_row,
        warp_col=warp_col,
        warp_lse=warp_lse,
        post_act=post_act,
        residual=residual,
        reduction=reduction,
        activation=activation,
        target_index=target_index,
        target_logit=target_logit,
    )

    if not use_tuned:
        gemm_epilogue(
            A=A,
            B=B,
            C=out,
            epi_cls=epi_cls,
            epi_args=epi_args,
            epi_keys=epi_keys,
            pingpong=DEFAULT_PINGPONG,
            tile_shape_mn=(DEFAULT_TILE_M, DEFAULT_TILE_N),
            cluster_shape_mn=(DEFAULT_CLUSTER_M, DEFAULT_CLUSTER_N),
            add_to_output=add_to_output,
        )
    else:
        gemm_epilogue_tuned(
            A=A,
            B=B,
            C=out,
            epi_cls=epi_cls,
            epi_args=epi_args,
            epi_keys=epi_keys,
            add_to_output=add_to_output,
        )

    out = torch.permute(out, dims=(2, 0, 1))
    out = torch.squeeze(out, dim=0)
    return out, epi_outs
