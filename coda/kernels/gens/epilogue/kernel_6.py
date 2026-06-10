import torch
import cutlass
import cutlass.cute as cute
from dataclasses import dataclass
from typing import Callable, NamedTuple
from quack.cute_dsl_utils import torch2cute_dtype_map

from hilt.dtype_utils import get_dtype
from coda.core.ops import misc_utils
from coda.core.ops import dtype_utils
from coda.core.ops import struct_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops import epilogue_utils
from coda.core.ops import pipeline_utils
from coda.core.epilogue import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)


# Hopper wgmma fragment lays lanes out as 8 (M) x 4 (N) within a warp.
# Lanes 0..3 share the same M position; the M reduction butterflies across
# offsets 16, 8, 4 — i.e. down to a width-4 group at the end.
_HOPPER_WARP_REDUCTION_WIDTH = 4


class EVTResidualRMSNormBwd(EpilogueVisitorTree):
    """
    Custom epilogue for backward pass of GEMM-RMSNorm pattern.

    Loads R and ZdZ column vectors via cp.async, loads C matrix via TMA pipeline,
    computes:
        C_out = C * R (stored via TMA store: reg → smem → gmem)
        result = (D - C_out * (ZdZ / K)) * R (stored as main output via TMA add)

    The O residual addition is handled by add_to_output mode in the GEMM kernel,
    which uses TMA reduction-add to write: O_out += result.
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mColVecR: cute.Tensor | None
        mColVecZdZ: cute.Tensor | None
        mMatrix: cute.Tensor | None
        mPostAct: cute.Tensor | None
        # dW row-block reduction store: shape (L, N, M // tile_M), fp32
        mRowVec: cute.Tensor | None
        # RMSNorm weight W: shape (L, N), broadcast along M
        mRowBias: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        mColVecR: cute.Tensor | None
        mColVecZdZ: cute.Tensor | None
        # C matrix TMA load
        mMatrix: cute.Tensor | None
        epi_tma_atom_load: cute.CopyAtom
        epi_gmem_layout_load: cutlass.utils.LayoutEnum
        epi_smem_layout_staged_load: cute.Layout
        # C_out TMA store
        mPostAct: cute.Tensor | None
        epi_tma_atom_store: cute.CopyAtom
        epi_gmem_layout_store: cutlass.utils.LayoutEnum
        epi_smem_layout_staged_store: cute.Layout
        # dW row-block reduction store
        mRowVec: cute.Tensor | None
        # RMSNorm weight W
        mRowBias: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sColVecR: cute.Tensor | None
        sColVecZdZ: cute.Tensor | None
        sMatrix: cute.Tensor | None
        epi_load_pipeline_array_ptr: cute.Pointer
        sPostAct: cute.Tensor | None
        # Inter-warp combine slab for the dW row-block reduction
        sRowVec: cute.Tensor | None
        # RMSNorm weight W loaded once per CTA-tile
        sRowBias: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDsColVecR: cute.Tensor | None
        tDsColVecZdZ: cute.Tensor | None
        # C matrix TMA load tensors
        tDsMatrix: cute.Tensor | None
        tDgMatrix: cute.Tensor | None
        tSR_sMatrix: cute.Tensor | None
        tRS_rMatrix: cute.Tensor | None
        tSR_rMatrix: cute.Tensor | None
        tiled_copy_s2r: cute.TiledCopy
        # C_out TMA store tensors
        tDsPostAct: cute.Tensor | None
        tDgPostAct: cute.Tensor | None
        tRS_sPostAct: cute.Tensor | None
        epi_tma_atom_store: cute.CopyAtom
        tiled_copy_postact_r2s: cute.TiledCopy
        # dW row-block reducer (per-thread, broadcast over M)
        tDrRowVec: cute.Tensor | None
        # W broadcast smem view, partitioned per-thread
        tDsRowBias: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tDrColVecR_epi: cute.Tensor | None
        tDrColVecZdZ_epi: cute.Tensor | None
        tRS_rMatrix: cute.Tensor | None
        # C_out TMA store loop tensors
        tDsPostAct: cute.Tensor | None
        tDgPostAct: cute.Tensor | None
        tRS_rPostAct: cute.Tensor | None
        tRS_sPostAct: cute.Tensor | None
        epi_tma_atom_store: cute.CopyAtom
        tiled_copy_postact_r2s: cute.TiledCopy
        # dW row-block reducer per-epi-tile slice
        tDrRowVec_epi: cute.Tensor | None
        # W per-epi-tile s→r register slice
        tDrRowBias_epi: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        epi_load_pipeline: cutlass.pipeline.PipelineTmaAsync
        epi_load_consumer_state: cutlass.pipeline.PipelineState
        epi_load_producer_state: cutlass.pipeline.PipelineState

    def __init__(
        self,
        acc_dtype: type[cute.Numeric],
        epi_dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        buffer_align_bytes: int,
    ) -> None:
        super().__init__()
        self.acc_dtype = acc_dtype
        self.epi_dtype = epi_dtype
        self.container_dtype = epi_dtype
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
        if cutlass.const_expr(epi_args.mColVecR is not None):
            mColVecR = misc_utils.static_assert_is_Tensor(epi_args.mColVecR)
            mColVecR = layout_utils.assumed_align_stride(mColVecR, assumed_align=4)
        else:
            mColVecR = None

        if cutlass.const_expr(epi_args.mColVecZdZ is not None):
            mColVecZdZ = misc_utils.static_assert_is_Tensor(epi_args.mColVecZdZ)
            mColVecZdZ = layout_utils.assumed_align_stride(mColVecZdZ, assumed_align=4)
        else:
            mColVecZdZ = None

        if cutlass.const_expr(epi_args.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_args.mMatrix)
            misc_utils.static_assert(get_dtype(mMatrix) is self.epi_dtype)
            (
                epi_gmem_layout_load,
                epi_smem_layout_staged_load,
                epi_tma_atom_load,
                epi_tma_tensor_load,
            ) = epilogue_utils.prepare_tma(
                tma_op="g2s",
                epi_tile=epi_tile,
                epi_stage=epi_load_stage,
                epi_tensor=mMatrix,
            )

        if cutlass.const_expr(epi_args.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_args.mPostAct)
            misc_utils.static_assert(get_dtype(mPostAct) is self.epi_dtype)
            (
                epi_gmem_layout_store,
                epi_smem_layout_staged_store,
                epi_tma_atom_store,
                epi_tma_tensor_store,
            ) = epilogue_utils.prepare_tma(
                tma_op="s2g",
                epi_tile=epi_tile,
                epi_stage=epi_stage,
                epi_tensor=mPostAct,
            )
        else:
            mPostAct = None

        if cutlass.const_expr(epi_args.mRowVec is not None):
            mRowVec = misc_utils.static_assert_is_Tensor(epi_args.mRowVec)
            mRowVec = layout_utils.assumed_align_stride(mRowVec, assumed_align=4)
        else:
            mRowVec = None

        if cutlass.const_expr(epi_args.mRowBias is not None):
            mRowBias = misc_utils.static_assert_is_Tensor(epi_args.mRowBias)
            mRowBias = layout_utils.assumed_align_stride(mRowBias, assumed_align=4)
        else:
            mRowBias = None

        return self.EpilogueParams(
            mColVecR=mColVecR,
            mColVecZdZ=mColVecZdZ,
            mMatrix=epi_tma_tensor_load,
            epi_tma_atom_load=epi_tma_atom_load,
            epi_gmem_layout_load=epi_gmem_layout_load,
            epi_smem_layout_staged_load=epi_smem_layout_staged_load,
            mPostAct=epi_tma_tensor_store,
            epi_tma_atom_store=epi_tma_atom_store,
            epi_gmem_layout_store=epi_gmem_layout_store,
            epi_smem_layout_staged_store=epi_smem_layout_staged_store,
            mRowVec=mRowVec,
            mRowBias=mRowBias,
        )

    @cute.jit
    def prefetch_tma_descriptors(self, epi_params: EpilogueParams) -> None:
        cute.nvgpu.cpasync.prefetch_descriptor(epi_params.epi_tma_atom_load)
        if cutlass.const_expr(epi_params.mPostAct is not None):
            cute.nvgpu.cpasync.prefetch_descriptor(epi_params.epi_tma_atom_store)

    @cute.jit
    def prepare_pipelines(
        self,
        epi_load_stage: int,
        epi_num_warps: int,
        epi_params: EpilogueParams,
        epi_tensors_smem: EpilogueTensorsSMem,
    ) -> EpiloguePipelines:
        if cutlass.const_expr(epi_params.mMatrix is not None):
            epi_smem_layout = cute.slice_(epi_params.epi_smem_layout_staged_load, (None, None, 0))
            epi_load_pipeline, epi_load_consumer_state, epi_load_producer_state = epilogue_utils.prepare_epi_load_pipeline(
                epi_load_stage=epi_load_stage,
                epi_dtype=self.container_dtype,
                epi_num_warps=epi_num_warps,
                epi_smem_layout=epi_smem_layout,
                epi_load_pipeline_mbar_ptr=epi_tensors_smem.epi_load_pipeline_array_ptr,
            )
        return self.EpiloguePipelines(
            epi_load_pipeline=epi_load_pipeline,
            epi_load_consumer_state=epi_load_consumer_state,
            epi_load_producer_state=epi_load_producer_state,
        )

    @cute.jit
    def advance_pipelines(
        self,
        tile_count: int,
        epi_params: EpilogueParams,
        epi_pipelines: EpiloguePipelines,
    ) -> EpiloguePipelines:
        epi_load_pipeline = epi_pipelines.epi_load_pipeline
        epi_load_consumer_state = epi_pipelines.epi_load_consumer_state
        epi_load_producer_state = epi_pipelines.epi_load_producer_state
        if cutlass.const_expr(epi_params.mMatrix is not None):
            epi_load_consumer_state = pipeline_utils.advance_n(
                state=epi_load_consumer_state,
                num_iterations=tile_count,
            )
            epi_load_producer_state = pipeline_utils.advance_n(
                state=epi_load_producer_state,
                num_iterations=tile_count,
            )
        return self.EpiloguePipelines(
            epi_load_pipeline=epi_load_pipeline,
            epi_load_consumer_state=epi_load_consumer_state,
            epi_load_producer_state=epi_load_producer_state,
        )

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

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        # === Load R column vector via cp.async ===
        if cutlass.const_expr(epi_params.mColVecR is not None):
            mColVecR = misc_utils.static_assert_is_Tensor(epi_params.mColVecR)
            sColVecR = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sColVecR)
            mColVecR = mColVecR[batch_idx, None]
            gColVecR = cute.local_tile(mColVecR, (tile_M,), (m_idx,))
            cColVecR = cute.make_identity_tensor(tile_M)
            limit_m = min(mColVecR.shape[0] - m_idx * tile_M, tile_M)
            memory_utils.g2s_copy_1d(
                src=gColVecR, dst=sColVecR, crd=cColVecR,
                shape=(limit_m,), num_threads=epi_num_threads, thread_index=tidx,
            )
            sColVecR_view = cute.make_tensor(
                iterator=sColVecR.iterator,
                layout=cute.make_layout(shape=(tile_M, tile_N), stride=(1, 0)),
            )
            tDsColVecR = partition_for_epilogue(sColVecR_view)
        else:
            tDsColVecR = None

        # === Load ZdZ column vector via cp.async ===
        if cutlass.const_expr(epi_params.mColVecZdZ is not None):
            mColVecZdZ = misc_utils.static_assert_is_Tensor(epi_params.mColVecZdZ)
            sColVecZdZ = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sColVecZdZ)
            mColVecZdZ = mColVecZdZ[batch_idx, None]
            gColVecZdZ = cute.local_tile(mColVecZdZ, (tile_M,), (m_idx,))
            cColVecZdZ = cute.make_identity_tensor(tile_M)
            limit_m_zdz = min(mColVecZdZ.shape[0] - m_idx * tile_M, tile_M)
            memory_utils.g2s_copy_1d(
                src=gColVecZdZ, dst=sColVecZdZ, crd=cColVecZdZ,
                shape=(limit_m_zdz,), num_threads=epi_num_threads, thread_index=tidx,
            )
            sColVecZdZ_view = cute.make_tensor(
                iterator=sColVecZdZ.iterator,
                layout=cute.make_layout(shape=(tile_M, tile_N), stride=(1, 0)),
            )
            tDsColVecZdZ = partition_for_epilogue(sColVecZdZ_view)
        else:
            tDsColVecZdZ = None

        # === Load W (RMSNorm weight) row vector via cp.async ===
        # W is shape (N,) broadcast over M. Load tile_N elements once per
        # CTA-tile and present a stride-(0, 1) broadcast view over (tile_M, tile_N).
        if cutlass.const_expr(epi_params.mRowBias is not None):
            mRowBias = misc_utils.static_assert_is_Tensor(epi_params.mRowBias)
            sRowBias = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sRowBias)
            mRowBias = mRowBias[batch_idx, None]
            gRowBias = cute.local_tile(mRowBias, (tile_N,), (n_idx,))
            cRowBias = cute.make_identity_tensor(tile_N)
            limit_n_w = min(mRowBias.shape[0] - n_idx * tile_N, tile_N)
            memory_utils.g2s_copy_1d(
                src=gRowBias, dst=sRowBias, crd=cRowBias,
                shape=(limit_n_w,), num_threads=epi_num_threads, thread_index=tidx,
            )
            sRowBias_view = cute.make_tensor(
                iterator=sRowBias.iterator,
                layout=cute.make_layout(shape=(tile_M, tile_N), stride=(0, 1)),
            )
            tDsRowBias = partition_for_epilogue(sRowBias_view)
        else:
            tDsRowBias = None

        if cutlass.const_expr(
            (tDsColVecR is not None) or (tDsColVecZdZ is not None) or (tDsRowBias is not None)
        ):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epi_barrier.arrive_and_wait()

        # === Set up TMA load for C (g2s) ===
        if cutlass.const_expr(epi_params.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_params.mMatrix)
            sMatrix = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sMatrix)
            gMatrix = mMatrix[None, None, batch_idx]
            gMatrix = cute.local_tile(gMatrix, (tile_M, tile_N), (m_idx, n_idx))
            gMatrix = cute.zipped_divide(gMatrix, epi_tile)
            tDsMatrix, tDgMatrix = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.epi_tma_atom_load, cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sMatrix, 0, cute.rank(sMatrix) - 1),
                gmem_tensor=cute.group_modes(gMatrix, 0, cute.rank(gMatrix) - 1),
            )
            tiled_copy_s2r, _, tSR_sMatrix, tRS_rMatrix, tSR_rMatrix = epilogue_utils.prepare_copy_s2r_sm90(
                tiled_mma=tiled_mma, tidx=tidx, src=sMatrix,
                dst_layout=tRS_rD_layout, epi_dtype=self.epi_dtype,
                container_dtype=self.container_dtype,
                epi_gmem_layout=epi_params.epi_gmem_layout_load,
                epi_num_matrices=epi_num_matrices,
            )

        # === Set up TMA store for C_out (r2s then s2g) ===
        if cutlass.const_expr(epi_params.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_params.mPostAct)
            sPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sPostAct)
            # RMem -> SMem copy
            tiled_copy_postact_r2s, _, tRS_sPostAct = epilogue_utils.prepare_copy_r2s_sm90(
                tiled_copy_r2s=tiled_copy_r2s,
                tidx=tidx,
                dst=sPostAct,
                epi_layout=epi_params.epi_gmem_layout_store,
                epi_dtype=self.container_dtype,
                acc_dtype=self.acc_dtype,
            )
            # SMem -> GMem copy
            gPostAct = mPostAct[None, None, batch_idx]
            gPostAct = cute.local_tile(gPostAct, (tile_M, tile_N), (m_idx, n_idx))
            gPostAct = cute.zipped_divide(gPostAct, epi_tile)
            tDsPostAct, tDgPostAct = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.epi_tma_atom_store, cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
                gmem_tensor=cute.group_modes(gPostAct, 0, cute.rank(gPostAct) - 1),
            )
        else:
            tDsPostAct = None
            tDgPostAct = None
            tRS_sPostAct = None
            epi_tma_atom_store = None
            tiled_copy_postact_r2s = None

        # === Allocate per-thread dW row-block reducer (broadcast over M) ===
        if cutlass.const_expr(epi_params.mRowVec is not None):
            rRowVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(0, 1),
            )
            rRowVec_view = creation_utils.allocate_tensor_from_layout(
                layout=rRowVec_view_layout,
                dtype=cute.Float32,
                memspace="rmem",
                smem_allocator=None,
            )
            tDrRowVec = partition_for_epilogue(rRowVec_view)
            cute.filter_zeros(tDrRowVec).fill(0.0)
        else:
            tDrRowVec = None

        return self.EpilogueTensors(
            tDsColVecR=tDsColVecR, tDsColVecZdZ=tDsColVecZdZ,
            tDsMatrix=tDsMatrix, tDgMatrix=tDgMatrix,
            tSR_sMatrix=tSR_sMatrix, tRS_rMatrix=tRS_rMatrix,
            tSR_rMatrix=tSR_rMatrix, tiled_copy_s2r=tiled_copy_s2r,
            tDsPostAct=tDsPostAct, tDgPostAct=tDgPostAct,
            tRS_sPostAct=tRS_sPostAct,
            epi_tma_atom_store=epi_params.epi_tma_atom_store,
            tiled_copy_postact_r2s=tiled_copy_postact_r2s,
            tDrRowVec=tDrRowVec,
            tDsRowBias=tDsRowBias,
        )

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:

        epi_load_pipeline = epi_pipelines.epi_load_pipeline
        epi_load_consumer_state = epi_pipelines.epi_load_consumer_state
        epi_load_producer_state = epi_pipelines.epi_load_producer_state

        if cutlass.const_expr(epi_tensors.tDsColVecR is not None):
            tDsColVecR = misc_utils.static_assert_is_Tensor(epi_tensors.tDsColVecR)
            tDsColVecR_cur = cute.group_modes(tDsColVecR, 3, cute.rank(tDsColVecR))
            tDsColVecR_cur = tDsColVecR_cur[None, None, None, epi_coord]
            tDrColVecR_cvt = memory_utils.s2r_copy_1d(tDsColVecR_cur, dtype=self.acc_dtype)
        else:
            tDrColVecR_cvt = None

        if cutlass.const_expr(epi_tensors.tDsColVecZdZ is not None):
            tDsColVecZdZ = misc_utils.static_assert_is_Tensor(epi_tensors.tDsColVecZdZ)
            tDsColVecZdZ_cur = cute.group_modes(tDsColVecZdZ, 3, cute.rank(tDsColVecZdZ))
            tDsColVecZdZ_cur = tDsColVecZdZ_cur[None, None, None, epi_coord]
            tDrColVecZdZ_cvt = memory_utils.s2r_copy_1d(tDsColVecZdZ_cur, dtype=self.acc_dtype)
        else:
            tDrColVecZdZ_cvt = None

        if cutlass.const_expr(epi_tensors.tDsMatrix is not None):
            tSR_sMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tSR_sMatrix)
            tSR_rMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tSR_rMatrix)
            tRS_rMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tRS_rMatrix)
            tiled_copy = epi_tensors.tiled_copy_s2r
            src = tSR_sMatrix[None, None, None, epi_load_consumer_state.index]
            dst = tSR_rMatrix
            epi_load_pipeline.consumer_wait(epi_load_consumer_state)
            cute.copy(atom=tiled_copy, src=src, dst=dst)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                epi_load_pipeline.consumer_release(epi_load_consumer_state)
            epi_load_consumer_state.advance()

        if cutlass.const_expr(epi_tensors.tDrRowVec is not None):
            tDrRowVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrRowVec)
            tDrRowVec_cur = cute.group_modes(tDrRowVec, 3, cute.rank(tDrRowVec))
            tDrRowVec_cur = tDrRowVec_cur[None, None, None, epi_coord]
        else:
            tDrRowVec_cur = None

        if cutlass.const_expr(epi_tensors.tDsRowBias is not None):
            tDsRowBias = misc_utils.static_assert_is_Tensor(epi_tensors.tDsRowBias)
            tDsRowBias_cur = cute.group_modes(tDsRowBias, 3, cute.rank(tDsRowBias))
            tDsRowBias_cur = tDsRowBias_cur[None, None, None, epi_coord]
            tDrRowBias_cvt = memory_utils.s2r_copy_1d(tDsRowBias_cur, dtype=self.acc_dtype)
        else:
            tDrRowBias_cvt = None

        return (
            self.EpilogueTensorsLoop(
                tDrColVecR_epi=tDrColVecR_cvt,
                tDrColVecZdZ_epi=tDrColVecZdZ_cvt,
                tRS_rMatrix=tRS_rMatrix,
                tDsPostAct=epi_tensors.tDsPostAct,
                tDgPostAct=epi_tensors.tDgPostAct,
                tRS_rPostAct=None,
                tRS_sPostAct=epi_tensors.tRS_sPostAct,
                epi_tma_atom_store=epi_tensors.epi_tma_atom_store,
                tiled_copy_postact_r2s=epi_tensors.tiled_copy_postact_r2s,
                tDrRowVec_epi=tDrRowVec_cur,
                tDrRowBias_epi=tDrRowBias_cvt,
            ),
            self.EpiloguePipelines(
                epi_load_pipeline=epi_load_pipeline,
                epi_load_consumer_state=epi_load_consumer_state,
                epi_load_producer_state=epi_load_producer_state,
            ),
        )

    @cute.jit
    def consumer_visit(
        self,
        tRS_rD: cute.Tensor,
        shape_mnk: cute.Shape,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> EpilogueTensorsLoop:

        if cutlass.const_expr(epi_tensors_loop.tRS_rMatrix is not None):
            tRS_rC = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_rMatrix)
            tDrColVecR = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrColVecR_epi)
            tDrColVecZdZ = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrColVecZdZ_epi)
            tDrRowBias_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrRowBias_epi)
            tDrRowVec_epi = epi_tensors_loop.tDrRowVec_epi

            rC = tRS_rC.load()
            rC = dtype_utils.convert(rC, dtype=get_dtype(tRS_rD))

            # Allocate register tensor for C_out = C_norm * W (= h_norm)
            tRS_rPostAct = creation_utils.allocate_tensor_like(
                tensor=tRS_rD,
                memspace="rmem",
                smem_allocator=None,
                dtype=self.acc_dtype,
            )

            # NOTE: range_constexpr (not range(..., unroll_full=True)) is
            # load-bearing here. With dynamic-`i`, ptxas can't fully resolve
            # liveness through the indexed reads/writes below and conservatively
            # spills the broadcast dW accumulator to local memory — measured at
            # MNK=4096 as 120 STL + 22 LDL/thread, ~30% slower. range_constexpr
            # makes `i` a Python compile-time int so each [i] resolves to a
            # specific register slot at trace time and ptxas keeps everything
            # in registers (0 STL, 0 LDL).
            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                c_val = rC[i]
                r_val = tDrColVecR[i]
                zdz_val = tDrColVecZdZ[i]
                w_val = tDrRowBias_epi[i]
                d_val = tRS_rD[i]
                c_norm = c_val * r_val
                # residual gradient: (D * W - C_norm * ZdZ) * R
                tRS_rD[i] = (d_val * w_val - c_norm * zdz_val) * r_val
                # post-RMSNorm output (h * R), with W applied: C_out = C_norm * W
                tRS_rPostAct[i] = c_norm * w_val
                # per-thread row-reduce accumulator: dW += D * C_norm  (no W here)
                if cutlass.const_expr(tDrRowVec_epi is not None):
                    tDrRowVec_epi[i] = tDrRowVec_epi[i] + d_val * c_norm

            tRS_rPostAct = dtype_utils.convert(
                tRS_rPostAct,
                dtype=self.epi_dtype,
            )

        return self.EpilogueTensorsLoop(
            tDrColVecR_epi=epi_tensors_loop.tDrColVecR_epi,
            tDrColVecZdZ_epi=epi_tensors_loop.tDrColVecZdZ_epi,
            tRS_rMatrix=epi_tensors_loop.tRS_rMatrix,
            tDsPostAct=epi_tensors_loop.tDsPostAct,
            tDgPostAct=epi_tensors_loop.tDgPostAct,
            tRS_rPostAct=tRS_rPostAct,
            tRS_sPostAct=epi_tensors_loop.tRS_sPostAct,
            epi_tma_atom_store=epi_tensors_loop.epi_tma_atom_store,
            tiled_copy_postact_r2s=epi_tensors_loop.tiled_copy_postact_r2s,
            tDrRowVec_epi=epi_tensors_loop.tDrRowVec_epi,
            tDrRowBias_epi=epi_tensors_loop.tDrRowBias_epi,
        )

    @cute.jit
    def consumer_smem_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        if cutlass.const_expr(epi_params.mPostAct is not None):
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
        if cutlass.const_expr(epi_params.mPostAct is not None):
            atom = epi_tensors_loop.epi_tma_atom_store
            tDsPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDsPostAct)
            tDgPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDgPostAct)
            src = tDsPostAct[None, epi_buffer]
            dst = tDgPostAct[None, epi_coord]
            cute.copy(atom=atom, src=src, dst=dst)

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
        # Finalize the dW row-block reduction: warp-butterfly along M, optional
        # inter-warp combine via smem, then a single gmem store per N column.
        if cutlass.const_expr(epi_params.mRowVec is None):
            return

        tile_M = self.tile_shape_mnk[0]
        tile_N = self.tile_shape_mnk[1]
        m_idx, n_idx, _, batch_idx = tile_coord_mnkl
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        mRowVec = misc_utils.static_assert_is_Tensor(epi_params.mRowVec)
        sRowVec = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sRowVec)
        tDrRowVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrRowVec)
        row_vec_limit_n = min(shape_mnk[1] - n_idx * tile_N, tile_N)
        row_vec_limit_m = mRowVec.shape[2]

        # Filter the broadcast (stride-0 along M) modes — we only need to
        # butterfly the unique register slots.
        tDrRowVec_filtered = cute.filter_zeros(tDrRowVec)
        for i in cutlass.range_constexpr(cute.size(tDrRowVec_filtered)):
            offset = cute.arch.WARP_SIZE // 2
            while offset >= _HOPPER_WARP_REDUCTION_WIDTH:
                tDrRowVec_filtered[i] = (
                    tDrRowVec_filtered[i]
                    + cute.arch.shuffle_sync_bfly(
                        tDrRowVec_filtered[i],
                        offset=offset,
                    )
                )
                offset = offset // 2

        mRowVec = mRowVec[batch_idx, None, m_idx]
        gRowVec = cute.local_tile(mRowVec, (tile_N,), (n_idx,))
        cRowVec = cute.make_identity_tensor((tile_M, tile_N))
        tDcRowVec = partition_for_epilogue(cRowVec)
        tDrRowVec_n = layout_utils.select_nonzero_stride_modes(tDrRowVec, tDrRowVec.layout)
        tDcRowVec_n = layout_utils.select_nonzero_stride_modes(tDcRowVec, tDrRowVec.layout)

        misc_utils.static_assert(cute.size(tiled_copy_r2s) == epi_num_threads)
        warps_in_M = epi_num_threads // cute.arch.WARP_SIZE
        warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
        warp_m_idx = warp_idx

        # lanes 0..3 of EVERY warp deposit their per-warp partial into smem.
        is_lane_m_leader = cute.arch.lane_idx() < _HOPPER_WARP_REDUCTION_WIDTH
        # lanes 0..3 of warp 0 collect from smem and write to gmem (mirrors
        # the layout-derived check used by rapier's column reduction).
        should_write_gmem = tDcRowVec_n[0][0] == 0

        # Inter-warp reduction through smem (skipped for warps_in_M == 1)
        if cutlass.const_expr(warps_in_M > 1):
            if warp_m_idx > 0 and is_lane_m_leader:
                for n in cutlass.range(cute.size(tDcRowVec_n, mode=[0])):
                    col_idx = tDcRowVec_n[n][1]
                    sRowVec[col_idx, warp_m_idx - 1] = tDrRowVec_n[n]
            epi_barrier.arrive_and_wait()
            if should_write_gmem:
                for n in cutlass.range(cute.size(tDcRowVec_n, mode=[0])):
                    col_idx = tDcRowVec_n[n][1]
                    for warp_m in cutlass.range_constexpr(1, warps_in_M):
                        tDrRowVec_n[n] = (
                            tDrRowVec_n[n] + sRowVec[col_idx, warp_m - 1]
                        )

        if m_idx < row_vec_limit_m and should_write_gmem:
            for n in cutlass.range(cute.size(tDcRowVec_n, mode=[0])):
                col_idx = tDcRowVec_n[n][1]
                if col_idx < row_vec_limit_n:
                    gRowVec[col_idx] = tDrRowVec_n[n].to(dtype=get_dtype(gRowVec))

    @cute.jit
    def producer_begin(
        self, is_tma_warp, epi_load_stage, epi_tile_num, epi_tile_layout,
        epi_params, epi_tensors, epi_pipelines,
    ) -> EpiloguePipelines:
        epi_load_pipeline = epi_pipelines.epi_load_pipeline
        epi_load_consumer_state = epi_pipelines.epi_load_consumer_state
        epi_load_producer_state = epi_pipelines.epi_load_producer_state
        if cutlass.const_expr(epi_params.mMatrix is not None):
            tDgMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDgMatrix)
            tDsMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDsMatrix)
            epi_prefetch = cutlass.min(epi_tile_num, epi_load_stage)
            for epi_idx in cutlass.range(epi_prefetch, unroll=1):
                epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                if is_tma_warp:
                    atom = epi_params.epi_tma_atom_load
                    src = tDgMatrix[None, epi_coord]
                    dst = tDsMatrix[None, epi_load_producer_state.index]
                    tma_bar_ptr = epi_load_pipeline.producer_get_barrier(epi_load_producer_state)
                    epi_load_pipeline.producer_acquire(epi_load_producer_state)
                    cute.copy(atom=atom, src=src, dst=dst, tma_bar_ptr=tma_bar_ptr)
                    epi_load_pipeline.producer_commit(epi_load_producer_state)
                epi_load_producer_state.advance()
        return self.EpiloguePipelines(
            epi_load_pipeline=epi_load_pipeline,
            epi_load_consumer_state=epi_load_consumer_state,
            epi_load_producer_state=epi_load_producer_state,
        )

    @cute.jit
    def producer_tma_load(
        self, is_tma_warp, epi_idx, epi_load_stage, epi_tile_num, epi_tile_layout,
        epi_params, epi_tensors, epi_pipelines,
    ) -> EpiloguePipelines:
        epi_load_pipeline = epi_pipelines.epi_load_pipeline
        epi_load_consumer_state = epi_pipelines.epi_load_consumer_state
        epi_load_producer_state = epi_pipelines.epi_load_producer_state
        if cutlass.const_expr(epi_params.mMatrix is not None and epi_idx + epi_load_stage < epi_tile_num):
            tDgMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDgMatrix)
            tDsMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDsMatrix)
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx + epi_load_stage)
            if is_tma_warp:
                atom = epi_params.epi_tma_atom_load
                src = tDgMatrix[None, epi_coord]
                dst = tDsMatrix[None, epi_load_producer_state.index]
                tma_bar_ptr = epi_load_pipeline.producer_get_barrier(epi_load_producer_state)
                epi_load_pipeline.producer_acquire(epi_load_producer_state)
                cute.copy(atom=atom, src=src, dst=dst, tma_bar_ptr=tma_bar_ptr)
                epi_load_pipeline.producer_commit(epi_load_producer_state)
            epi_load_producer_state.advance()
        return self.EpiloguePipelines(
            epi_load_pipeline=epi_load_pipeline,
            epi_load_consumer_state=epi_load_consumer_state,
            epi_load_producer_state=epi_load_producer_state,
        )

    @cute.jit
    def get_smem_struct(self, epi_load_stage, epi_num_threads, epi_params):
        if cutlass.const_expr(epi_params.mColVecR is not None):
            mColVecR = misc_utils.static_assert_is_Tensor(epi_params.mColVecR)
            col_vec_r_dtype = get_dtype(mColVecR)
            col_vec_r_smem_size = epilogue_utils.get_smem_size_vector(
                mTensor=mColVecR, epi_tile=self.tile_shape_mnk[0], epi_num_threads=epi_num_threads,
            )
        else:
            col_vec_r_dtype = cute.Float32
            col_vec_r_smem_size = 0

        if cutlass.const_expr(epi_params.mColVecZdZ is not None):
            mColVecZdZ = misc_utils.static_assert_is_Tensor(epi_params.mColVecZdZ)
            col_vec_zdz_dtype = get_dtype(mColVecZdZ)
            col_vec_zdz_smem_size = epilogue_utils.get_smem_size_vector(
                mTensor=mColVecZdZ, epi_tile=self.tile_shape_mnk[0], epi_num_threads=epi_num_threads,
            )
        else:
            col_vec_zdz_dtype = cute.Float32
            col_vec_zdz_smem_size = 0

        if cutlass.const_expr(epi_params.mMatrix is not None):
            matrix_smem_size = cute.cosize(epi_params.epi_smem_layout_staged_load)
        else:
            matrix_smem_size = 0

        if cutlass.const_expr(epi_params.mPostAct is not None):
            postact_smem_size = cute.cosize(epi_params.epi_smem_layout_staged_store)
        else:
            postact_smem_size = 0

        # dW inter-warp combine: tile_N * (warps_in_M - 1) fp32 entries.
        if cutlass.const_expr(epi_params.mRowVec is not None):
            warps_in_M = epi_num_threads // cute.arch.WARP_SIZE
            row_vec_dtype = cute.Float32
            row_vec_smem_size = self.tile_shape_mnk[1] * (warps_in_M - 1)
        else:
            row_vec_dtype = cute.Float32
            row_vec_smem_size = 0

        # W (RMSNorm weight) row vector: tile_N elements, broadcast over M.
        if cutlass.const_expr(epi_params.mRowBias is not None):
            mRowBias = misc_utils.static_assert_is_Tensor(epi_params.mRowBias)
            row_bias_dtype = get_dtype(mRowBias)
            row_bias_smem_size = epilogue_utils.get_smem_size_vector(
                mTensor=mRowBias, epi_tile=self.tile_shape_mnk[1], epi_num_threads=epi_num_threads,
            )
        else:
            row_bias_dtype = cute.Float32
            row_bias_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            sColVecR: cute.struct.Align[cute.struct.MemRange[col_vec_r_dtype, col_vec_r_smem_size], 16]
            sColVecZdZ: cute.struct.Align[cute.struct.MemRange[col_vec_zdz_dtype, col_vec_zdz_smem_size], 16]
            epi_load_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, epi_load_stage * 2]
            sMatrix: cute.struct.Align[cute.struct.MemRange[self.container_dtype, matrix_smem_size], self.buffer_align_bytes]
            sPostAct: cute.struct.Align[cute.struct.MemRange[self.container_dtype, postact_smem_size], self.buffer_align_bytes]
            sRowVec: cute.struct.Align[cute.struct.MemRange[row_vec_dtype, row_vec_smem_size], 16]
            sRowBias: cute.struct.Align[cute.struct.MemRange[row_bias_dtype, row_bias_smem_size], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(self, storage, epi_num_threads, epi_params):
        if cutlass.const_expr(epi_params.mColVecR is not None):
            sColVecR = storage.sColVecR.get_tensor(cute.make_layout(self.tile_shape_mnk[0]))
        else:
            sColVecR = None
        if cutlass.const_expr(epi_params.mColVecZdZ is not None):
            sColVecZdZ = storage.sColVecZdZ.get_tensor(cute.make_layout(self.tile_shape_mnk[0]))
        else:
            sColVecZdZ = None
        if cutlass.const_expr(epi_params.mMatrix is not None):
            epi_load_pipeline_array_ptr = storage.epi_load_pipeline_array_ptr.data_ptr()
            sMatrix = storage.sMatrix.get_tensor(
                epi_params.epi_smem_layout_staged_load.outer,
                swizzle=epi_params.epi_smem_layout_staged_load.inner,
            )
        else:
            sMatrix = None
            epi_load_pipeline_array_ptr = None
        if cutlass.const_expr(epi_params.mPostAct is not None):
            sPostAct = storage.sPostAct.get_tensor(
                epi_params.epi_smem_layout_staged_store.outer,
                swizzle=epi_params.epi_smem_layout_staged_store.inner,
            )
        else:
            sPostAct = None
        if cutlass.const_expr(epi_params.mRowVec is not None):
            warps_in_M = epi_num_threads // cute.arch.WARP_SIZE
            sRowVec_layout = layout_utils.make_ordered_layout(
                shape=(self.tile_shape_mnk[1], warps_in_M - 1),
                order="row",
            )
            sRowVec = storage.sRowVec.get_tensor(sRowVec_layout)
        else:
            sRowVec = None
        if cutlass.const_expr(epi_params.mRowBias is not None):
            sRowBias = storage.sRowBias.get_tensor(cute.make_layout(self.tile_shape_mnk[1]))
        else:
            sRowBias = None
        return self.EpilogueTensorsSMem(
            sColVecR=sColVecR, sColVecZdZ=sColVecZdZ,
            sMatrix=sMatrix, epi_load_pipeline_array_ptr=epi_load_pipeline_array_ptr,
            sPostAct=sPostAct, sRowVec=sRowVec, sRowBias=sRowBias,
        )

    @cute.jit
    def get_smem_bytes_per_stage(self, epi_tile, epi_num_threads, epi_args):
        epi_smem_bytes_fixed = 0
        epi_smem_bytes_per_stage_cst = 0
        epi_smem_bytes_per_stage_pld = 0

        if cutlass.const_expr(epi_args.mColVecR is not None):
            mColVecR = misc_utils.static_assert_is_Tensor(epi_args.mColVecR)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + epilogue_utils.get_epi_smem_bytes_per_stage_fixed_vector(
                mTensor=mColVecR, epi_tile=self.tile_shape_mnk[0], epi_num_threads=epi_num_threads,
            )
        if cutlass.const_expr(epi_args.mColVecZdZ is not None):
            mColVecZdZ = misc_utils.static_assert_is_Tensor(epi_args.mColVecZdZ)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + epilogue_utils.get_epi_smem_bytes_per_stage_fixed_vector(
                mTensor=mColVecZdZ, epi_tile=self.tile_shape_mnk[0], epi_num_threads=epi_num_threads,
            )
        if cutlass.const_expr(epi_args.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_args.mMatrix)
            misc_utils.static_assert(get_dtype(mMatrix) is self.epi_dtype)
            epi_smem_bytes_per_stage_pld = epi_smem_bytes_per_stage_pld + epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                mTensor=mMatrix, epi_tile=epi_tile,
            )
        if cutlass.const_expr(epi_args.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_args.mPostAct)
            misc_utils.static_assert(get_dtype(mPostAct) is self.epi_dtype)
            epi_smem_bytes_per_stage_cst = epi_smem_bytes_per_stage_cst + epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                mTensor=mPostAct, epi_tile=epi_tile,
            )
        if cutlass.const_expr(epi_args.mRowVec is not None):
            warps_in_M = epi_num_threads // cute.arch.WARP_SIZE
            row_vec_dtype = cute.Float32
            row_vec_smem_size = self.tile_shape_mnk[1] * (warps_in_M - 1)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + (
                row_vec_smem_size * row_vec_dtype.width // 8
            )
        if cutlass.const_expr(epi_args.mRowBias is not None):
            mRowBias = misc_utils.static_assert_is_Tensor(epi_args.mRowBias)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + epilogue_utils.get_epi_smem_bytes_per_stage_fixed_vector(
                mTensor=mRowBias, epi_tile=self.tile_shape_mnk[1], epi_num_threads=epi_num_threads,
            )
        return (epi_smem_bytes_fixed, epi_smem_bytes_per_stage_cst, epi_smem_bytes_per_stage_pld)


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    C: torch.Tensor,
    W: torch.Tensor,
    R: torch.Tensor,
    ZdZ: torch.Tensor,
    C_out: torch.Tensor,
    dW: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for backward pass of GEMM-RMSNorm pattern.

    Single EVT visitor that:
        - Loads R, ZdZ (cp.async) and C (TMA load)
        - Computes C_out = C * R, stores via TMA store (reg → smem → gmem)
        - Computes (D - C_out * ZdZ) * R as main output
        - O residual addition handled by add_to_output TMA reduction-add
        - Row-block reduction store dW = sum_M (D * C_out) per M-tile
          (shape (L, N, M // tile_M), fp32). Per-thread accumulation runs
          inline with consumer_visit; consumer_end finalizes via warp
          butterfly + inter-warp smem combine + a single gmem store.
    """
    M, N, K, L = shape_mnkl

    epi_dtype = torch2cute_dtype_map[C.dtype]

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTResidualRMSNormBwd(
        acc_dtype=acc_dtype,
        epi_dtype=epi_dtype,
        tile_shape_mnk=tile_shape_mnk,
        buffer_align_bytes=buffer_align_bytes,
    )

    epi_args = EVTResidualRMSNormBwd.EpilogueArguments(
        mColVecR=R,
        mColVecZdZ=ZdZ,
        mMatrix=C,
        mPostAct=C_out,
        mRowVec=dW,
        mRowBias=W,
    )

    epi_keys = (
        R.dtype, ZdZ.dtype, C.dtype, W.dtype, C_out.dtype, dW.dtype,
        EVTResidualRMSNormBwd,
    )

    epi_outs = {}
    return epi_cls, epi_args, epi_outs, epi_keys
