import operator
import torch
import cutlass
import cutlass.cute as cute
import torch.utils._pytree as pytree
from dataclasses import dataclass
from typing import Callable, NamedTuple

from quack.cute_dsl_utils import torch2cute_dtype_map
from hilt.dtype_utils import get_dtype
from rapier.ops import misc_utils
from rapier.ops import dtype_utils
from rapier.ops import struct_utils
from rapier.ops import layout_utils
from rapier.ops import memory_utils
from rapier.ops import creation_utils
from rapier.ops import epilogue_utils
from rapier.ops.reduction_utils import BlockReductionOp
from rapier.epilogue import (
    EVTList,
    EVTResidual,
    EVTColBlockReductionStore,
    EpilogueVisitorTree,
)
from rapier.epilogue.base import (
    EpilogueSharedStorage,
)

HOPPER_WARP_REDUCTION_WIDTH = 4


def _create_mean_sq_reduction_op(element_type, inv_block_size):
    """Create a reduction op that accumulates mean of squares: acc + val^2 * inv_block_size.

    The combine_fn squares each new element and scales by 1/block_size before adding
    to the accumulator. The warp-level reduction uses standard addition since partial
    sums are already accumulated and scaled.
    """
    init_value = element_type(0.)
    inv_bs = element_type(inv_block_size)

    _sq_combine = lambda x, y: x + y * y * inv_bs
    _add_wrp = lambda tree_x, tree_y: pytree.tree_map(operator.add, tree_x, tree_y)

    return BlockReductionOp(
        combine_fn=lambda tree_x, tree_y: pytree.tree_map(_sq_combine, tree_x, tree_y),
        reduce_ssa=None,
        reduce_wrp=lambda xs: pytree.tree_map(
            lambda x: cute.arch.warp_reduction(
                x,
                op=_add_wrp,
                threads_in_group=HOPPER_WARP_REDUCTION_WIDTH,
            ),
            xs,
        ),
        init_value=init_value,
    )


class EVTRowVecMulPostAct(EpilogueVisitorTree):
    """
    Loads a per-N row vector W (cp.async to smem, then s2r), multiplies the
    accumulator by W into a separate register tile, and stores that scaled
    tile to a side output mPostAct via TMA. tRS_rD itself is left unchanged
    so the main D output (the unscaled GEMM result) is unaffected.

    This mirrors the rowvec=norm_weight side-output path of trainstation's
    `gemm_partial_rms_fwd`, kept local to this kernel rather than as a
    general-purpose rapier EVT.

    Inputs:
        - GEMM output (in registers): [M x N], unchanged by this op
        - mRowVec: [L, N]  — RMSNorm weight, broadcast along M

    Outputs:
        - mPostAct: [M x N] = D * W  (side output, written via TMA)
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor | None
        mRowVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        mPostAct: cute.Tensor | None
        mRowVec: cute.Tensor | None
        epi_tma_atom: cute.CopyAtom
        epi_gmem_layout: cutlass.utils.LayoutEnum
        epi_smem_layout_staged: cute.Layout

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sPostAct: cute.Tensor | None
        sRowVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDsPostAct: cute.Tensor
        tDgPostAct: cute.Tensor
        tRS_sPostAct: cute.Tensor
        epi_tma_atom: cute.CopyAtom
        tiled_copy_postact_r2s: cute.TiledCopy
        tDsRowVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tDsPostAct: cute.Tensor
        tDgPostAct: cute.Tensor
        tRS_rPostAct: cute.Tensor | None
        tRS_sPostAct: cute.Tensor
        epi_tma_atom: cute.CopyAtom
        tiled_copy_postact_r2s: cute.TiledCopy
        tDrRowVec_epi: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        acc_dtype: type[cute.Numeric],
        post_act_dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        buffer_align_bytes: int,
    ) -> None:
        super().__init__()
        self.arch = 90
        self.acc_dtype = acc_dtype
        self.post_act_dtype = post_act_dtype
        self.container_dtype = post_act_dtype
        self.tile_shape_mnk = tile_shape_mnk
        self.buffer_align_bytes = buffer_align_bytes

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:

        if cutlass.const_expr(epi_args.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_args.mPostAct)
            misc_utils.static_assert(get_dtype(mPostAct) is self.container_dtype)
            (
                epi_gmem_layout,
                epi_smem_layout_staged,
                epi_tma_atom,
                epi_tma_tensor,
            ) = epilogue_utils.prepare_tma(
                tma_op="s2g",
                epi_tile=epi_tile,
                epi_stage=epi_stage,
                epi_tensor=mPostAct,
            )

        if cutlass.const_expr(epi_args.mRowVec is not None):
            misc_utils.static_assert(epi_args.mPostAct is not None)
            mRowVec = misc_utils.static_assert_is_Tensor(epi_args.mRowVec)
            mRowVec = layout_utils.assumed_align_stride(
                mRowVec,
                assumed_align=4,
            )
        else:
            mRowVec = None

        return self.EpilogueParams(
            mPostAct=epi_tma_tensor,
            mRowVec=mRowVec,
            epi_tma_atom=epi_tma_atom,
            epi_gmem_layout=epi_gmem_layout,
            epi_smem_layout_staged=epi_smem_layout_staged,
        )

    @cute.jit
    def prefetch_tma_descriptors(
        self,
        epi_params: EpilogueParams,
    ) -> None:
        cute.nvgpu.cpasync.prefetch_descriptor(epi_params.epi_tma_atom)

    @cute.jit
    def consumer_begin(
        self,
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        tidx: cute.Int32,
        tiled_mma: cute.TiledMma,
        tRS_rD_layout: cute.Layout,
        epi_tile: cute.Tile,
        epi_num_threads: int,
        epi_num_matrices: int,
        epi_barrier: cutlass.pipeline.NamedBarrier,
        epi_params: EpilogueParams,
        epi_tensors_smem: EpilogueTensorsSMem,
    ) -> EpilogueTensors:

        tile_M = self.tile_shape_mnk[0]
        tile_N = self.tile_shape_mnk[1]
        m_idx, n_idx, _, batch_idx = tile_coord_mnkl
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        # Side output (PostAct) TMA setup
        mPostAct = misc_utils.static_assert_is_Tensor(epi_params.mPostAct)
        sPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sPostAct)
        tiled_copy_postact_r2s, _, tRS_sPostAct = epilogue_utils.prepare_copy_r2s_sm90(
            tiled_copy_r2s=tiled_copy_r2s,
            tidx=tidx,
            dst=sPostAct,
            epi_layout=epi_params.epi_gmem_layout,
            epi_dtype=self.container_dtype,
            acc_dtype=self.acc_dtype,
        )
        gPostAct = mPostAct[None, None, batch_idx]
        gPostAct = cute.local_tile(gPostAct, (tile_M, tile_N), (m_idx, n_idx))
        gPostAct = cute.zipped_divide(gPostAct, epi_tile)

        tDsPostAct, tDgPostAct = cute.nvgpu.cpasync.tma_partition(
            atom=epi_params.epi_tma_atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
            gmem_tensor=cute.group_modes(gPostAct, 0, cute.rank(gPostAct) - 1),
        )

        # RowVec cp.async load (per-N broadcast across M)
        if cutlass.const_expr(epi_params.mRowVec is not None):
            mRowVec = misc_utils.static_assert_is_Tensor(epi_params.mRowVec)
            sRowVec = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sRowVec)
            mRowVec = mRowVec[batch_idx, None]
            gRowVec = cute.local_tile(mRowVec, (tile_N,), (n_idx,))
            cRowVec = cute.make_identity_tensor(tile_N)
            limit_n = min(mRowVec.shape[0] - n_idx * tile_N, tile_N)
            memory_utils.g2s_copy_1d(
                src=gRowVec,
                dst=sRowVec,
                crd=cRowVec,
                shape=(limit_n,),
                num_threads=epi_num_threads,
                thread_index=tidx,
            )
            sRowVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(0, 1),
            )
            sRowVec_view = cute.make_tensor(
                iterator=sRowVec.iterator,
                layout=sRowVec_view_layout,
            )
            tDsRowVec = thr_copy_r2s.partition_S(
                cute.flat_divide(sRowVec_view, epi_tile)
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epi_barrier.arrive_and_wait()
        else:
            tDsRowVec = None

        return self.EpilogueTensors(
            tDsPostAct=tDsPostAct,
            tDgPostAct=tDgPostAct,
            tRS_sPostAct=tRS_sPostAct,
            epi_tma_atom=epi_params.epi_tma_atom,
            tiled_copy_postact_r2s=tiled_copy_postact_r2s,
            tDsRowVec=tDsRowVec,
        )

    @cute.jit
    def consumer_end(
        self,
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        tidx: cute.Int32,
        shape_mnk: cute.Shape,
        epi_tile: cute.Tile,
        epi_num_threads: int,
        epi_barrier: cutlass.pipeline.NamedBarrier,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_tensors_smem: EpilogueTensorsSMem,
    ) -> None:
        pass

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:

        if cutlass.const_expr(epi_tensors.tDsRowVec is not None):
            tDsRowVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDsRowVec)
            tDsRowVec_cur = cute.group_modes(tDsRowVec, 3, cute.rank(tDsRowVec))
            tDsRowVec_cur = tDsRowVec_cur[None, None, None, epi_coord]
            tDrRowVec_cvt = memory_utils.s2r_copy_1d(tDsRowVec_cur, dtype=self.acc_dtype)
        else:
            tDrRowVec_cvt = None

        return (
            self.EpilogueTensorsLoop(
                tDsPostAct=epi_tensors.tDsPostAct,
                tDgPostAct=epi_tensors.tDgPostAct,
                tRS_rPostAct=None,
                tRS_sPostAct=epi_tensors.tRS_sPostAct,
                epi_tma_atom=epi_tensors.epi_tma_atom,
                tiled_copy_postact_r2s=epi_tensors.tiled_copy_postact_r2s,
                tDrRowVec_epi=tDrRowVec_cvt,
            ),
            self.EpiloguePipelines(),
        )

    @cute.jit
    def consumer_visit(
        self,
        tRS_rD: cute.Tensor,
        shape_mnk: cute.Shape,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> EpilogueTensorsLoop:

        tRS_rPostAct = creation_utils.allocate_tensor_like(
            tensor=tRS_rD,
            memspace="rmem",
            smem_allocator=None,
            dtype=self.acc_dtype,
        )
        if cutlass.const_expr(self.arch < 100):
            if cutlass.const_expr(epi_tensors_loop.tDrRowVec_epi is not None):
                tDrRowVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrRowVec_epi)
                for i in cutlass.range_constexpr(cute.size(tRS_rPostAct)):
                    tRS_rPostAct[i] = tRS_rD[i] * tDrRowVec_epi[i]
            else:
                for i in cutlass.range_constexpr(cute.size(tRS_rPostAct)):
                    tRS_rPostAct[i] = tRS_rD[i]
        else:
            raise NotImplementedError

        tRS_rPostAct = dtype_utils.convert(
            tRS_rPostAct,
            dtype=self.post_act_dtype,
        )

        return self.EpilogueTensorsLoop(
            tDsPostAct=epi_tensors_loop.tDsPostAct,
            tDgPostAct=epi_tensors_loop.tDgPostAct,
            tRS_rPostAct=tRS_rPostAct,
            tRS_sPostAct=epi_tensors_loop.tRS_sPostAct,
            epi_tma_atom=epi_tensors_loop.epi_tma_atom,
            tiled_copy_postact_r2s=epi_tensors_loop.tiled_copy_postact_r2s,
            tDrRowVec_epi=epi_tensors_loop.tDrRowVec_epi,
        )

    @cute.jit
    def consumer_smem_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        tiled_copy = epi_tensors_loop.tiled_copy_postact_r2s
        tRS_rPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_rPostAct)
        tRS_sPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_sPostAct)
        src = tiled_copy.retile(tRS_rPostAct)
        dst = tRS_sPostAct[None, None, None, epi_buffer]
        cute.copy(atom=tiled_copy, src=src, dst=dst)

    @cute.jit
    def consumer_tma_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        atom = epi_tensors_loop.epi_tma_atom
        tDsPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDsPostAct)
        tDgPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDgPostAct)
        src = tDsPostAct[None, epi_buffer]
        dst = tDgPostAct[None, epi_coord]
        cute.copy(atom=atom, src=src, dst=dst)

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:

        if cutlass.const_expr(epi_params.mPostAct is not None):
            post_act_smem_size = cute.cosize(epi_params.epi_smem_layout_staged)
        else:
            post_act_smem_size = 0

        if cutlass.const_expr(epi_params.mRowVec is not None):
            mRowVec = misc_utils.static_assert_is_Tensor(epi_params.mRowVec)
            row_vec_dtype = get_dtype(mRowVec)
            row_vec_smem_size = epilogue_utils.get_smem_size_vector(
                mTensor=mRowVec,
                epi_tile=self.tile_shape_mnk[1],
                epi_num_threads=epi_num_threads,
            )
        else:
            row_vec_dtype = cute.Float32
            row_vec_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            sPostAct: cute.struct.Align[cute.struct.MemRange[self.container_dtype, post_act_smem_size], self.buffer_align_bytes]
            sRowVec: cute.struct.Align[cute.struct.MemRange[row_vec_dtype, row_vec_smem_size], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mPostAct is not None):
            sPostAct = storage.sPostAct.get_tensor(
                epi_params.epi_smem_layout_staged.outer,
                swizzle=epi_params.epi_smem_layout_staged.inner,
            )
        else:
            sPostAct = None

        if cutlass.const_expr(epi_params.mRowVec is not None):
            sRowVec_layout = cute.make_layout(self.tile_shape_mnk[1])
            sRowVec = storage.sRowVec.get_tensor(sRowVec_layout)
        else:
            sRowVec = None

        return self.EpilogueTensorsSMem(
            sPostAct=sPostAct,
            sRowVec=sRowVec,
        )

    @cute.jit
    def get_smem_bytes_per_stage(
        self,
        epi_tile: cute.Tile,
        epi_num_threads: int,
        epi_args: EpilogueArguments,
    ) -> tuple[int, int, int]:
        epi_smem_bytes_fixed = 0
        epi_smem_bytes_per_stage_cst = 0
        epi_smem_bytes_per_stage_pld = 0

        if cutlass.const_expr(epi_args.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_args.mPostAct)
            misc_utils.static_assert(get_dtype(mPostAct) is self.container_dtype)
            epi_smem_bytes_per_stage_cst = epi_smem_bytes_per_stage_cst + (
                epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                    mTensor=mPostAct,
                    epi_tile=epi_tile,
                )
            )

        if cutlass.const_expr(epi_args.mRowVec is not None):
            mRowVec = misc_utils.static_assert_is_Tensor(epi_args.mRowVec)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + (
                epilogue_utils.get_epi_smem_bytes_per_stage_fixed_vector(
                    mTensor=mRowVec,
                    epi_tile=self.tile_shape_mnk[1],
                    epi_num_threads=epi_num_threads,
                )
            )

        return (
            epi_smem_bytes_fixed,
            epi_smem_bytes_per_stage_cst,
            epi_smem_bytes_per_stage_pld,
        )


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    C: torch.Tensor,
    S: torch.Tensor,
    W: torch.Tensor,
    O: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with residual, partial mean-of-squares, and
    fused per-N RMSNorm-weight scaling — mirrors trainstation's `gemm_partial_rms_fwd`.

    Composes three EVT visitors:
        1. EVTResidual: D = acc + C
        2. EVTColBlockReductionStore: S[m, nb] = mean(D[m, nb*bs:(nb+1)*bs]^2)
        3. EVTRowVecMulPostAct (local): O[m, n] = D[m, n] * W[n], side output via TMA

    The partial sum-of-squares is computed on the *unscaled* D, so a downstream
    rstd reduction sees the GEMM output before W is applied. tRS_rD is preserved
    so the main D output is also unscaled.

    Args:
        shape_mnkl: Problem shape (M, N, K, L) where L is batch dimension.
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        C: Residual matrix of shape (M, N).
        S: Output for partial mean-of-squares of shape (M, num_blocks) in fp32.
        W: RMSNorm weight of shape (N,), broadcast across M.
        O: Output of shape (M, N) for D * W.

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    epi_dtype = torch2cute_dtype_map[C.dtype]
    post_act_dtype = torch2cute_dtype_map[O.dtype]

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList([
        EVTResidual(
            acc_dtype=acc_dtype,
            epi_dtype=epi_dtype,
            tile_shape_mnk=tile_shape_mnk,
            buffer_align_bytes=buffer_align_bytes,
        ),
        EVTColBlockReductionStore(
            reduction_op=_create_mean_sq_reduction_op(
                element_type=acc_dtype,
                inv_block_size=1.0 / tile_shape_mnk[1],
            ),
            tile_shape_mnk=tile_shape_mnk,
        ),
        EVTRowVecMulPostAct(
            acc_dtype=acc_dtype,
            post_act_dtype=post_act_dtype,
            tile_shape_mnk=tile_shape_mnk,
            buffer_align_bytes=buffer_align_bytes,
        ),
    ])

    epi_args = EVTList.EpilogueArguments([
        EVTResidual.EpilogueArguments(
            mMatrix=C,
        ),
        EVTColBlockReductionStore.EpilogueArguments(
            mColVec=S,
        ),
        EVTRowVecMulPostAct.EpilogueArguments(
            mPostAct=O,
            mRowVec=W,
        ),
    ])

    epi_keys = (
        C.dtype,
        S.dtype,
        W.dtype,
        O.dtype,
        EVTResidual,
        EVTColBlockReductionStore,
        EVTRowVecMulPostAct,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
