import torch
import cutlass
import cutlass.cute as cute
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
from rapier.ops import pipeline_utils
from rapier.ops import reduction_utils
from rapier.epilogue import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)

HOPPER_WARP_REDUCTION_WIDTH = 4


class EVTSwiGLUBwdReduced(EpilogueVisitorTree):
    """
    Custom epilogue for backward pass of GEMM with SwiGLU activation,
    with per-tile block reduction of ZdZ.

    Loads Z matrix (M, 2N) via TMA pipeline using 2X container pattern
    (two Float16 values packed per Float32), computes SwiGLU backward:
        sigmoid(G), silu(G) = G * sigmoid(G)
        O = silu(G) * U (stored as main output via standard epilogue TMA)
        dG = D * U * (sigmoid(G) + silu(G) * (1 - sigmoid(G)))
        dU = D * silu(G)
        dZ = interleaved(dG, dU) (stored via TMA store, 2X container)
        ZdZ = reduce(Z * dZ, "m (nb bs 2) -> m nb", "sum", bs=block_size) / block_size
              (accumulated per-tile via stride-0 registers, scaled by 1/block_size,
               then warp-reduced and stored in consumer_end)
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mMatrix: cute.Tensor | None
        mDZ: cute.Tensor | None
        mZDZ: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        mMatrix: cute.Tensor | None
        epi_tma_atom_load: cute.CopyAtom
        epi_gmem_layout_load: cutlass.utils.LayoutEnum
        epi_smem_layout_staged_load: cute.Layout
        # dZ TMA store
        mDZ: cute.Tensor | None
        epi_tma_atom_store_dz: cute.CopyAtom
        epi_gmem_layout_store_dz: cutlass.utils.LayoutEnum
        epi_smem_layout_staged_store_dz: cute.Layout
        # ZdZ reduction output
        mZDZ: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sMatrix: cute.Tensor | None
        epi_load_pipeline_array_ptr: cute.Pointer
        sDZ: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        # Z matrix TMA load
        tDsMatrix: cute.Tensor | None
        tDgMatrix: cute.Tensor | None
        tSR_sMatrix: cute.Tensor | None
        tRS_rMatrix: cute.Tensor | None
        tSR_rMatrix: cute.Tensor | None
        tiled_copy_s2r: cute.TiledCopy
        # dZ TMA store
        tDsDZ: cute.Tensor | None
        tDgDZ: cute.Tensor | None
        tRS_sDZ: cute.Tensor | None
        epi_tma_atom_store_dz: cute.CopyAtom
        tiled_copy_dz_r2s: cute.TiledCopy
        # ZdZ reduction accumulator
        tDrColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tRS_rMatrix: cute.Tensor | None
        # dZ TMA store loop
        tDsDZ: cute.Tensor | None
        tDgDZ: cute.Tensor | None
        tRS_rDZ: cute.Tensor | None
        tRS_sDZ: cute.Tensor | None
        epi_tma_atom_store_dz: cute.CopyAtom
        tiled_copy_dz_r2s: cute.TiledCopy
        # ZdZ reduction accumulator (per epi-tile slice)
        tDrColVec_epi: cute.Tensor | None

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
        misc_utils.static_assert(epi_dtype.width == 16)
        self.container_dtype = cute.Float32
        self.tile_shape_mnk = tile_shape_mnk
        self.buffer_align_bytes = buffer_align_bytes
        self.inv_block_size = 1.0 / tile_shape_mnk[1]
        self.reduction_op = reduction_utils.get_registered_reduction_op(
            name="add",
            element_type=acc_dtype,
        )

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:
        if cutlass.const_expr(epi_args.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_args.mMatrix)
            misc_utils.static_assert(get_dtype(mMatrix) is self.epi_dtype)
            mMatrix = cute.recast_tensor(mMatrix, dtype=self.container_dtype)
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

        if cutlass.const_expr(epi_args.mDZ is not None):
            mDZ = misc_utils.static_assert_is_Tensor(epi_args.mDZ)
            misc_utils.static_assert(get_dtype(mDZ) is self.epi_dtype)
            mDZ_recast = cute.recast_tensor(mDZ, dtype=self.container_dtype)
            (
                epi_gmem_layout_store_dz,
                epi_smem_layout_staged_store_dz,
                epi_tma_atom_store_dz,
                epi_tma_tensor_store_dz,
            ) = epilogue_utils.prepare_tma(
                tma_op="s2g",
                epi_tile=epi_tile,
                epi_stage=epi_stage,
                epi_tensor=mDZ_recast,
            )
        else:
            mDZ = None

        if cutlass.const_expr(epi_args.mZDZ is not None):
            mZDZ = misc_utils.static_assert_is_Tensor(epi_args.mZDZ)
            mZDZ = layout_utils.assumed_align_stride(mZDZ, assumed_align=4)
        else:
            mZDZ = None

        return self.EpilogueParams(
            mMatrix=epi_tma_tensor_load,
            epi_tma_atom_load=epi_tma_atom_load,
            epi_gmem_layout_load=epi_gmem_layout_load,
            epi_smem_layout_staged_load=epi_smem_layout_staged_load,
            mDZ=epi_tma_tensor_store_dz,
            epi_tma_atom_store_dz=epi_tma_atom_store_dz,
            epi_gmem_layout_store_dz=epi_gmem_layout_store_dz,
            epi_smem_layout_staged_store_dz=epi_smem_layout_staged_store_dz,
            mZDZ=mZDZ,
        )

    @cute.jit
    def prefetch_tma_descriptors(self, epi_params: EpilogueParams) -> None:
        cute.nvgpu.cpasync.prefetch_descriptor(epi_params.epi_tma_atom_load)
        if cutlass.const_expr(epi_params.mDZ is not None):
            cute.nvgpu.cpasync.prefetch_descriptor(epi_params.epi_tma_atom_store_dz)

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

        # === Set up TMA load for Z matrix (2X container) ===
        if cutlass.const_expr(epi_params.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_params.mMatrix)
            sMatrix = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sMatrix)
            gMatrix = mMatrix[None, None, batch_idx]
            gMatrix = cute.local_tile(gMatrix, (tile_M, tile_N), (m_idx, n_idx))
            gMatrix = cute.zipped_divide(gMatrix, epi_tile)
            tDsMatrix, tDgMatrix = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.epi_tma_atom_load,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sMatrix, 0, cute.rank(sMatrix) - 1),
                gmem_tensor=cute.group_modes(gMatrix, 0, cute.rank(gMatrix) - 1),
            )
            tiled_copy_s2r, _, tSR_sMatrix, tRS_rMatrix, tSR_rMatrix = epilogue_utils.prepare_copy_s2r_sm90(
                tiled_mma=tiled_mma,
                tidx=tidx,
                src=sMatrix,
                dst_layout=tRS_rD_layout,
                epi_dtype=self.epi_dtype,
                container_dtype=self.container_dtype,
                epi_gmem_layout=epi_params.epi_gmem_layout_load,
                epi_num_matrices=epi_num_matrices,
            )

        # === Set up TMA store for dZ (2X container, r2s then s2g) ===
        if cutlass.const_expr(epi_params.mDZ is not None):
            mDZ = misc_utils.static_assert_is_Tensor(epi_params.mDZ)
            sDZ = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sDZ)
            tiled_copy_dz_r2s, _, tRS_sDZ = epilogue_utils.prepare_copy_r2s_sm90(
                tiled_copy_r2s=tiled_copy_r2s,
                tidx=tidx,
                dst=sDZ,
                epi_layout=epi_params.epi_gmem_layout_store_dz,
                epi_dtype=self.container_dtype,
                acc_dtype=self.acc_dtype,
            )
            gDZ = mDZ[None, None, batch_idx]
            gDZ = cute.local_tile(gDZ, (tile_M, tile_N), (m_idx, n_idx))
            gDZ = cute.zipped_divide(gDZ, epi_tile)
            tDsDZ, tDgDZ = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.epi_tma_atom_store_dz,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sDZ, 0, cute.rank(sDZ) - 1),
                gmem_tensor=cute.group_modes(gDZ, 0, cute.rank(gDZ) - 1),
            )
        else:
            tDsDZ = None
            tDgDZ = None
            tRS_sDZ = None
            tiled_copy_dz_r2s = None

        # === Set up ZdZ reduction accumulator (stride-0 in N) ===
        if cutlass.const_expr(epi_params.mZDZ is not None):
            rColVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(1, 0),
            )
            rColVec_view = creation_utils.allocate_tensor_from_layout(
                layout=rColVec_view_layout,
                dtype=cute.Float32,
                memspace="rmem",
                smem_allocator=None,
            )
            tDrColVec = partition_for_epilogue(rColVec_view)
            cute.filter_zeros(tDrColVec).fill(0.0)
        else:
            tDrColVec = None

        return self.EpilogueTensors(
            tDsMatrix=tDsMatrix, tDgMatrix=tDgMatrix,
            tSR_sMatrix=tSR_sMatrix, tRS_rMatrix=tRS_rMatrix,
            tSR_rMatrix=tSR_rMatrix, tiled_copy_s2r=tiled_copy_s2r,
            tDsDZ=tDsDZ, tDgDZ=tDgDZ, tRS_sDZ=tRS_sDZ,
            epi_tma_atom_store_dz=epi_params.epi_tma_atom_store_dz,
            tiled_copy_dz_r2s=tiled_copy_dz_r2s,
            tDrColVec=tDrColVec,
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

        tile_M = self.tile_shape_mnk[0]
        tile_N = self.tile_shape_mnk[1]
        m_idx, n_idx, _, batch_idx = tile_coord_mnkl
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mZDZ is not None):
            mZDZ = misc_utils.static_assert_is_Tensor(epi_params.mZDZ)
            tDrColVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrColVec)
            col_vec_limit_m = min(shape_mnk[0] - m_idx * tile_M, tile_M)
            col_vec_limit_n = mZDZ.shape[2]

            # Warp reduction (Hopper: 4 threads per warp hold one row's values)
            tDrColVec_filtered = cute.filter_zeros(tDrColVec)
            for i in cutlass.range_constexpr(cute.size(tDrColVec_filtered)):
                tDrColVec_filtered[i] = self.reduction_op.warp_reduction_singleton(
                    tDrColVec_filtered[i],
                    width=HOPPER_WARP_REDUCTION_WIDTH,
                )

            # Write reduced values to global memory
            mZDZ_slice = mZDZ[batch_idx, None, n_idx]
            gColVec = cute.local_tile(mZDZ_slice, (tile_M,), (m_idx,))
            cColVec = cute.make_identity_tensor((tile_M, tile_N))

            tDcColVec = partition_for_epilogue(cColVec)
            tDrColVec_m = layout_utils.select_nonzero_stride_modes(tDrColVec, tDrColVec.layout)
            tDcColVec_m = layout_utils.select_nonzero_stride_modes(tDcColVec, tDrColVec.layout)
            if n_idx < col_vec_limit_n and tDcColVec_m[0][1] == 0:
                for m in cutlass.range(cute.size(tDcColVec_m, mode=[0])):
                    row_idx = tDcColVec_m[m][0]
                    if row_idx < col_vec_limit_m:
                        gColVec[row_idx] = tDrColVec_m[m].to(dtype=get_dtype(gColVec))

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

        # Wait for TMA load and copy Z from smem to registers
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

        # Slice reduction accumulator for current epi tile
        if cutlass.const_expr(epi_tensors.tDrColVec is not None):
            tDrColVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrColVec)
            tDrColVec_cur = cute.group_modes(tDrColVec, 3, cute.rank(tDrColVec))
            tDrColVec_cur = tDrColVec_cur[None, None, None, epi_coord]
        else:
            tDrColVec_cur = None

        return (
            self.EpilogueTensorsLoop(
                tRS_rMatrix=tRS_rMatrix,
                tDsDZ=epi_tensors.tDsDZ, tDgDZ=epi_tensors.tDgDZ,
                tRS_rDZ=None, tRS_sDZ=epi_tensors.tRS_sDZ,
                epi_tma_atom_store_dz=epi_tensors.epi_tma_atom_store_dz,
                tiled_copy_dz_r2s=epi_tensors.tiled_copy_dz_r2s,
                tDrColVec_epi=tDrColVec_cur,
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
            tRS_rZ = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_rMatrix)
            misc_utils.static_assert(get_dtype(tRS_rZ) is self.container_dtype)
            tRS_rZ = cute.recast_tensor(tRS_rZ, dtype=self.epi_dtype)
            tRS_rZ = dtype_utils.convert(tRS_rZ, dtype=get_dtype(tRS_rD))

            has_zdz_red = cutlass.const_expr(epi_tensors_loop.tDrColVec_epi is not None)
            if cutlass.const_expr(has_zdz_red):
                tDrColVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrColVec_epi)

            # Allocate 2X register tensor for dZ
            tRS_rDZ = creation_utils.allocate_tensor_from_recast_layout(
                layout=tRS_rD.layout,
                new_type_bits=self.epi_dtype.width,
                old_type_bits=get_dtype(tRS_rD).width,
                memspace="rmem",
                smem_allocator=None,
                dtype=self.acc_dtype,
            )

            inv_bs = self.acc_dtype(self.inv_block_size)
            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                d_val = tRS_rD[i]
                g_val = tRS_rZ[2 * i]
                u_val = tRS_rZ[2 * i + 1]

                sigmoid_g = 1.0 / (1.0 + cute.math.exp(-g_val, fastmath=True))
                silu_g = g_val * sigmoid_g
                o_val = silu_g * u_val
                dU = d_val * silu_g
                dG = d_val * u_val * (sigmoid_g + silu_g * (1.0 - sigmoid_g))

                tRS_rD[i] = o_val
                tRS_rDZ[2 * i] = dG
                tRS_rDZ[2 * i + 1] = dU

                # Accumulate ZdZ reduction: mean(g*dG + u*dU) per row per tile
                # Scaled by 1/block_size to match reference's mean semantics.
                if cutlass.const_expr(has_zdz_red):
                    tDrColVec_epi[i] = tDrColVec_epi[i] + ((g_val * dG) + (u_val * dU)) * inv_bs

            # Convert to fp16 and recast pairs into fp32 container
            tRS_rDZ = dtype_utils.convert(tRS_rDZ, dtype=self.epi_dtype)
            tRS_rDZ = cute.recast_tensor(tRS_rDZ, dtype=self.container_dtype)

        return self.EpilogueTensorsLoop(
            tRS_rMatrix=epi_tensors_loop.tRS_rMatrix,
            tDsDZ=epi_tensors_loop.tDsDZ, tDgDZ=epi_tensors_loop.tDgDZ,
            tRS_rDZ=tRS_rDZ, tRS_sDZ=epi_tensors_loop.tRS_sDZ,
            epi_tma_atom_store_dz=epi_tensors_loop.epi_tma_atom_store_dz,
            tiled_copy_dz_r2s=epi_tensors_loop.tiled_copy_dz_r2s,
            tDrColVec_epi=epi_tensors_loop.tDrColVec_epi,
        )

    @cute.jit
    def consumer_smem_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        if cutlass.const_expr(epi_params.mDZ is not None):
            tiled_copy = epi_tensors_loop.tiled_copy_dz_r2s
            tRS_rDZ = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_rDZ)
            tRS_sDZ = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_sDZ)
            src = tiled_copy.retile(tRS_rDZ)
            dst = tRS_sDZ[None, None, None, epi_buffer]
            cute.copy(atom=tiled_copy, src=src, dst=dst)

    @cute.jit
    def consumer_tma_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        if cutlass.const_expr(epi_params.mDZ is not None):
            atom = epi_tensors_loop.epi_tma_atom_store_dz
            tDsDZ = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDsDZ)
            tDgDZ = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDgDZ)
            src = tDsDZ[None, epi_buffer]
            dst = tDgDZ[None, epi_coord]
            cute.copy(atom=atom, src=src, dst=dst)

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
        if cutlass.const_expr(epi_params.mMatrix is not None):
            matrix_smem_size = cute.cosize(epi_params.epi_smem_layout_staged_load)
        else:
            matrix_smem_size = 0

        if cutlass.const_expr(epi_params.mDZ is not None):
            dz_smem_size = cute.cosize(epi_params.epi_smem_layout_staged_store_dz)
        else:
            dz_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            epi_load_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, epi_load_stage * 2]
            sMatrix: cute.struct.Align[cute.struct.MemRange[self.container_dtype, matrix_smem_size], self.buffer_align_bytes]
            sDZ: cute.struct.Align[cute.struct.MemRange[self.container_dtype, dz_smem_size], self.buffer_align_bytes]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(self, storage, epi_num_threads, epi_params):
        if cutlass.const_expr(epi_params.mMatrix is not None):
            epi_load_pipeline_array_ptr = storage.epi_load_pipeline_array_ptr.data_ptr()
            sMatrix = storage.sMatrix.get_tensor(
                epi_params.epi_smem_layout_staged_load.outer,
                swizzle=epi_params.epi_smem_layout_staged_load.inner,
            )
        else:
            sMatrix = None
            epi_load_pipeline_array_ptr = None
        if cutlass.const_expr(epi_params.mDZ is not None):
            sDZ = storage.sDZ.get_tensor(
                epi_params.epi_smem_layout_staged_store_dz.outer,
                swizzle=epi_params.epi_smem_layout_staged_store_dz.inner,
            )
        else:
            sDZ = None
        return self.EpilogueTensorsSMem(
            sMatrix=sMatrix, epi_load_pipeline_array_ptr=epi_load_pipeline_array_ptr,
            sDZ=sDZ,
        )

    @cute.jit
    def get_smem_bytes_per_stage(self, epi_tile, epi_num_threads, epi_args):
        epi_smem_bytes_fixed = 0
        epi_smem_bytes_per_stage_cst = 0
        epi_smem_bytes_per_stage_pld = 0

        if cutlass.const_expr(epi_args.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_args.mMatrix)
            misc_utils.static_assert(get_dtype(mMatrix) is self.epi_dtype)
            mMatrix = cute.recast_tensor(mMatrix, dtype=self.container_dtype)
            epi_smem_bytes_per_stage_pld = epi_smem_bytes_per_stage_pld + epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                mTensor=mMatrix, epi_tile=epi_tile,
            )

        if cutlass.const_expr(epi_args.mDZ is not None):
            mDZ = misc_utils.static_assert_is_Tensor(epi_args.mDZ)
            misc_utils.static_assert(get_dtype(mDZ) is self.epi_dtype)
            mDZ = cute.recast_tensor(mDZ, dtype=self.container_dtype)
            epi_smem_bytes_per_stage_cst = epi_smem_bytes_per_stage_cst + epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                mTensor=mDZ, epi_tile=epi_tile,
            )

        return (epi_smem_bytes_fixed, epi_smem_bytes_per_stage_cst, epi_smem_bytes_per_stage_pld)


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    Z: torch.Tensor,
    dZ: torch.Tensor,
    ZdZ: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for backward pass of GEMM with SwiGLU activation (reduced).

    Single EVT visitor that:
        - Loads Z (M, 2N) via TMA pipeline using 2X container
        - Computes SwiGLU backward: O, dZ, ZdZ
        - Stores O as main GEMM output via standard epilogue TMA
        - Stores dZ via TMA store using 2X container (reg → smem → gmem)
        - Accumulates ZdZ = reduce(Z * dZ) per-tile via stride-0 registers,
          warp-reduces, and stores to (M, N // block_size) in consumer_end
    """
    M, N, K, L = shape_mnkl

    epi_dtype = torch2cute_dtype_map[Z.dtype]

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTSwiGLUBwdReduced(
        acc_dtype=acc_dtype,
        epi_dtype=epi_dtype,
        tile_shape_mnk=tile_shape_mnk,
        buffer_align_bytes=buffer_align_bytes,
    )

    epi_args = EVTSwiGLUBwdReduced.EpilogueArguments(
        mMatrix=Z,
        mDZ=dZ,
        mZDZ=ZdZ,
    )

    epi_keys = (
        Z.dtype, dZ.dtype, ZdZ.dtype,
        EVTSwiGLUBwdReduced,
    )

    epi_outs = {}
    return epi_cls, epi_args, epi_outs, epi_keys
