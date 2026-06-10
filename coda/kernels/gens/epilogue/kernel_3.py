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
from coda.core.ops import pipeline_utils
from coda.core.ops import creation_utils
from coda.core.ops import epilogue_utils
from coda.core.epilogue.base import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)


class EVTRoPE(EpilogueVisitorTree):
    """
    Loads cos_sin matrix via TMA, applies RoPE rotation pairwise to the GEMM
    accumulator, and stores the rotated result to a separate output buffer via TMA.

    The accumulator (tRS_rD) is NOT modified — the standard epilogue stores
    D = A @ B unchanged, while this visitor stores the rotated O separately.

    RoPE rotation on pairs (x, y) with interleaved (cos, sin):
        o_even = x * cos + y * sin
        o_odd  = -x * sin + y * cos

    Inputs:
        - GEMM output: [M x N]
        - cos_sin matrix (mCosSin): [M, N, L] with interleaved [cos, sin] pairs

    Output:
        - O = RoPE(acc, cos_sin): [M x N] stored to mOutput
        - D = acc: [M x N] stored by default epilogue (unchanged)
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mCosSin: cute.Tensor | None
        mOutput: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        # For loading cos_sin via TMA
        mCosSin: cute.Tensor | None
        load_tma_atom: cute.CopyAtom
        load_gmem_layout: cutlass.utils.LayoutEnum
        load_smem_layout_staged: cute.Layout
        # For storing output via TMA
        mOutput: cute.Tensor | None
        store_tma_atom: cute.CopyAtom
        store_gmem_layout: cutlass.utils.LayoutEnum
        store_smem_layout_staged: cute.Layout

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sCosSin: cute.Tensor | None
        sOutput: cute.Tensor | None
        epi_load_pipeline_array_ptr: cute.Pointer

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        # Load path (cos_sin)
        tDsCosSin: cute.Tensor | None
        tDgCosSin: cute.Tensor | None
        tSR_sCosSin: cute.Tensor | None
        tRS_rCosSin: cute.Tensor | None
        tSR_rCosSin: cute.Tensor | None
        tiled_copy_s2r: cute.TiledCopy
        # Store path (output)
        tDsOutput: cute.Tensor
        tDgOutput: cute.Tensor
        tRS_sOutput: cute.Tensor
        store_tma_atom: cute.CopyAtom
        tiled_copy_output_r2s: cute.TiledCopy

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tRS_rCosSin: cute.Tensor | None
        # Store path carried through loop
        tDsOutput: cute.Tensor
        tDgOutput: cute.Tensor
        tRS_rOutput: cute.Tensor | None
        tRS_sOutput: cute.Tensor
        store_tma_atom: cute.CopyAtom
        tiled_copy_output_r2s: cute.TiledCopy

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

        if cutlass.const_expr(epi_args.mCosSin is not None):
            mCosSin = misc_utils.static_assert_is_Tensor(epi_args.mCosSin)
            misc_utils.static_assert(get_dtype(mCosSin) is self.epi_dtype)
            (
                load_gmem_layout,
                load_smem_layout_staged,
                load_tma_atom,
                load_tma_tensor,
            ) = epilogue_utils.prepare_tma(
                tma_op="g2s",
                epi_tile=epi_tile,
                epi_stage=epi_load_stage,
                epi_tensor=mCosSin,
            )

        if cutlass.const_expr(epi_args.mOutput is not None):
            mOutput = misc_utils.static_assert_is_Tensor(epi_args.mOutput)
            misc_utils.static_assert(get_dtype(mOutput) is self.epi_dtype)
            (
                store_gmem_layout,
                store_smem_layout_staged,
                store_tma_atom,
                store_tma_tensor,
            ) = epilogue_utils.prepare_tma(
                tma_op="s2g",
                epi_tile=epi_tile,
                epi_stage=epi_stage,
                epi_tensor=mOutput,
            )

        return self.EpilogueParams(
            mCosSin=load_tma_tensor,
            load_tma_atom=load_tma_atom,
            load_gmem_layout=load_gmem_layout,
            load_smem_layout_staged=load_smem_layout_staged,
            mOutput=store_tma_tensor,
            store_tma_atom=store_tma_atom,
            store_gmem_layout=store_gmem_layout,
            store_smem_layout_staged=store_smem_layout_staged,
        )

    @cute.jit
    def prefetch_tma_descriptors(
        self,
        epi_params: EpilogueParams,
    ) -> None:
        cute.nvgpu.cpasync.prefetch_descriptor(epi_params.load_tma_atom)
        cute.nvgpu.cpasync.prefetch_descriptor(epi_params.store_tma_atom)

    @cute.jit
    def prepare_pipelines(
        self,
        epi_load_stage: int,
        epi_num_warps: int,
        epi_params: EpilogueParams,
        epi_tensors_smem: EpilogueTensorsSMem,
    ) -> EpiloguePipelines:
        if cutlass.const_expr(epi_params.mCosSin is not None):
            epi_smem_layout = cute.slice_(epi_params.load_smem_layout_staged, (None, None, 0))
            epi_load_pipeline, epi_load_consumer_state, epi_load_producer_state = epilogue_utils.prepare_epi_load_pipeline(
                epi_load_stage=epi_load_stage,
                epi_dtype=self.epi_dtype,
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
        if cutlass.const_expr(epi_params.mCosSin is not None):
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

        # --- Load path: cos_sin ---
        if cutlass.const_expr(epi_params.mCosSin is not None):
            mCosSin = misc_utils.static_assert_is_Tensor(epi_params.mCosSin)
            sCosSin = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sCosSin)
            gCosSin = mCosSin[None, None, batch_idx]
            gCosSin = cute.local_tile(gCosSin, (tile_M, tile_N), (m_idx, n_idx))
            gCosSin = cute.zipped_divide(gCosSin, epi_tile)

            tDsCosSin, tDgCosSin = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.load_tma_atom,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sCosSin, 0, cute.rank(sCosSin) - 1),
                gmem_tensor=cute.group_modes(gCosSin, 0, cute.rank(gCosSin) - 1),
            )

            tiled_copy_s2r, _, tSR_sCosSin, tRS_rCosSin, tSR_rCosSin = epilogue_utils.prepare_copy_s2r_sm90(
                tiled_mma=tiled_mma,
                tidx=tidx,
                src=sCosSin,
                dst_layout=tRS_rD_layout,
                epi_dtype=self.epi_dtype,
                container_dtype=self.epi_dtype,
                epi_gmem_layout=epi_params.load_gmem_layout,
                epi_num_matrices=epi_num_matrices,
            )

        # --- Store path: output ---
        if cutlass.const_expr(epi_params.mOutput is not None):
            mOutput = misc_utils.static_assert_is_Tensor(epi_params.mOutput)
            sOutput = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sOutput)
            # RMem -> SMem Copy
            tiled_copy_output_r2s, _, tRS_sOutput = epilogue_utils.prepare_copy_r2s_sm90(
                tiled_copy_r2s=tiled_copy_r2s,
                tidx=tidx,
                dst=sOutput,
                epi_layout=epi_params.store_gmem_layout,
                epi_dtype=self.epi_dtype,
                acc_dtype=self.acc_dtype,
            )
            # SMem -> GMem Copy
            gOutput = mOutput[None, None, batch_idx]
            gOutput = cute.local_tile(gOutput, (tile_M, tile_N), (m_idx, n_idx))
            gOutput = cute.zipped_divide(gOutput, epi_tile)

            tDsOutput, tDgOutput = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.store_tma_atom,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sOutput, 0, cute.rank(sOutput) - 1),
                gmem_tensor=cute.group_modes(gOutput, 0, cute.rank(gOutput) - 1),
            )

        return self.EpilogueTensors(
            tDsCosSin=tDsCosSin,
            tDgCosSin=tDgCosSin,
            tSR_sCosSin=tSR_sCosSin,
            tRS_rCosSin=tRS_rCosSin,
            tSR_rCosSin=tSR_rCosSin,
            tiled_copy_s2r=tiled_copy_s2r,
            tDsOutput=tDsOutput,
            tDgOutput=tDgOutput,
            tRS_sOutput=tRS_sOutput,
            store_tma_atom=epi_params.store_tma_atom,
            tiled_copy_output_r2s=tiled_copy_output_r2s,
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
        if cutlass.const_expr(epi_tensors.tDsCosSin is not None):
            tSR_sCosSin = misc_utils.static_assert_is_Tensor(epi_tensors.tSR_sCosSin)
            tSR_rCosSin = misc_utils.static_assert_is_Tensor(epi_tensors.tSR_rCosSin)
            tRS_rCosSin = misc_utils.static_assert_is_Tensor(epi_tensors.tRS_rCosSin)

            tiled_copy = epi_tensors.tiled_copy_s2r
            src = tSR_sCosSin[None, None, None, epi_load_consumer_state.index]
            dst = tSR_rCosSin

            epi_load_pipeline.consumer_wait(epi_load_consumer_state)
            cute.copy(atom=tiled_copy, src=src, dst=dst)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                epi_load_pipeline.consumer_release(epi_load_consumer_state)
            epi_load_consumer_state.advance()

        return (
            self.EpilogueTensorsLoop(
                tRS_rCosSin=tRS_rCosSin,
                tDsOutput=epi_tensors.tDsOutput,
                tDgOutput=epi_tensors.tDgOutput,
                tRS_rOutput=None,
                tRS_sOutput=epi_tensors.tRS_sOutput,
                store_tma_atom=epi_tensors.store_tma_atom,
                tiled_copy_output_r2s=epi_tensors.tiled_copy_output_r2s,
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
        if cutlass.const_expr(epi_tensors_loop.tRS_rCosSin is not None):
            tRS_rCS = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_rCosSin)
            # Convert cos_sin from epi_dtype to acc_dtype (fp32)
            tRS_rCS = dtype_utils.convert(tRS_rCS, dtype=get_dtype(tRS_rD))

            # Allocate output register tensor (same shape as accumulator)
            tRS_rOutput = creation_utils.allocate_tensor_like(
                tensor=tRS_rD,
                memspace="rmem",
                smem_allocator=None,
                dtype=self.acc_dtype,
            )

            # Apply RoPE pairwise: pairs at (2i, 2i+1)
            # cos_sin is interleaved: [cos0, sin0, cos1, sin1, ...]
            # o[2i]   =  d[2i]*cs[2i] + d[2i+1]*cs[2i+1]   (x*cos + y*sin)
            # o[2i+1] = -d[2i]*cs[2i+1] + d[2i+1]*cs[2i]   (-x*sin + y*cos)
            for i in cutlass.range_constexpr(cute.size(tRS_rD) // 2):
                x = tRS_rD[2 * i]
                y = tRS_rD[2 * i + 1]
                cos_val = tRS_rCS[2 * i]
                sin_val = tRS_rCS[2 * i + 1]
                tRS_rOutput[2 * i] = x * cos_val + y * sin_val
                tRS_rOutput[2 * i + 1] = -x * sin_val + y * cos_val

            # Convert to output dtype for storage
            tRS_rOutput = dtype_utils.convert(
                tRS_rOutput,
                dtype=self.epi_dtype,
            )
        else:
            tRS_rOutput = None

        return self.EpilogueTensorsLoop(
            tRS_rCosSin=epi_tensors_loop.tRS_rCosSin,
            tDsOutput=epi_tensors_loop.tDsOutput,
            tDgOutput=epi_tensors_loop.tDgOutput,
            tRS_rOutput=tRS_rOutput,
            tRS_sOutput=epi_tensors_loop.tRS_sOutput,
            store_tma_atom=epi_tensors_loop.store_tma_atom,
            tiled_copy_output_r2s=epi_tensors_loop.tiled_copy_output_r2s,
        )

    @cute.jit
    def consumer_smem_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        tiled_copy = epi_tensors_loop.tiled_copy_output_r2s
        tRS_rOutput = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_rOutput)
        tRS_sOutput = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_sOutput)
        src = tiled_copy.retile(tRS_rOutput)
        dst = tRS_sOutput[None, None, None, epi_buffer]
        cute.copy(atom=tiled_copy, src=src, dst=dst)

    @cute.jit
    def consumer_tma_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        atom = epi_tensors_loop.store_tma_atom
        tDsOutput = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDsOutput)
        tDgOutput = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDgOutput)
        src = tDsOutput[None, epi_buffer]
        dst = tDgOutput[None, epi_coord]
        cute.copy(atom=atom, src=src, dst=dst)

    @cute.jit
    def producer_begin(
        self,
        is_tma_warp: cute.Boolean,
        epi_load_stage: int,
        epi_tile_num: cute.Int32,
        epi_tile_layout: cute.Layout,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> EpiloguePipelines:

        epi_load_pipeline = epi_pipelines.epi_load_pipeline
        epi_load_consumer_state = epi_pipelines.epi_load_consumer_state
        epi_load_producer_state = epi_pipelines.epi_load_producer_state
        if cutlass.const_expr(epi_params.mCosSin is not None):
            tDgCosSin = misc_utils.static_assert_is_Tensor(epi_tensors.tDgCosSin)
            tDsCosSin = misc_utils.static_assert_is_Tensor(epi_tensors.tDsCosSin)

            epi_prefetch = cutlass.min(epi_tile_num, epi_load_stage)
            for epi_idx in cutlass.range(epi_prefetch, unroll=1):
                epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                if is_tma_warp:
                    atom = epi_params.load_tma_atom
                    src = tDgCosSin[None, epi_coord]
                    dst = tDsCosSin[None, epi_load_producer_state.index]
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
        self,
        is_tma_warp: cute.Boolean,
        epi_idx: int,
        epi_load_stage: int,
        epi_tile_num: int,
        epi_tile_layout: cute.Layout,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> EpiloguePipelines:

        epi_load_pipeline = epi_pipelines.epi_load_pipeline
        epi_load_consumer_state = epi_pipelines.epi_load_consumer_state
        epi_load_producer_state = epi_pipelines.epi_load_producer_state
        if cutlass.const_expr(epi_params.mCosSin is not None and epi_idx + epi_load_stage < epi_tile_num):
            tDgCosSin = misc_utils.static_assert_is_Tensor(epi_tensors.tDgCosSin)
            tDsCosSin = misc_utils.static_assert_is_Tensor(epi_tensors.tDsCosSin)
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx + epi_load_stage)

            if is_tma_warp:
                atom = epi_params.load_tma_atom
                src = tDgCosSin[None, epi_coord]
                dst = tDsCosSin[None, epi_load_producer_state.index]
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
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:

        if cutlass.const_expr(epi_params.mCosSin is not None):
            load_smem_size = cute.cosize(epi_params.load_smem_layout_staged)
        else:
            load_smem_size = 0

        if cutlass.const_expr(epi_params.mOutput is not None):
            store_smem_size = cute.cosize(epi_params.store_smem_layout_staged)
        else:
            store_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            epi_load_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, epi_load_stage * 2]
            sCosSin: cute.struct.Align[cute.struct.MemRange[self.epi_dtype, load_smem_size], self.buffer_align_bytes]
            sOutput: cute.struct.Align[cute.struct.MemRange[self.epi_dtype, store_smem_size], self.buffer_align_bytes]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mCosSin is not None):
            epi_load_pipeline_array_ptr = storage.epi_load_pipeline_array_ptr.data_ptr()
            sCosSin = storage.sCosSin.get_tensor(
                epi_params.load_smem_layout_staged.outer,
                swizzle=epi_params.load_smem_layout_staged.inner,
            )
        else:
            sCosSin = None
            epi_load_pipeline_array_ptr = None

        if cutlass.const_expr(epi_params.mOutput is not None):
            sOutput = storage.sOutput.get_tensor(
                epi_params.store_smem_layout_staged.outer,
                swizzle=epi_params.store_smem_layout_staged.inner,
            )
        else:
            sOutput = None

        return self.EpilogueTensorsSMem(
            sCosSin=sCosSin,
            sOutput=sOutput,
            epi_load_pipeline_array_ptr=epi_load_pipeline_array_ptr,
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

        # cos_sin loading: producer-staged
        if cutlass.const_expr(epi_args.mCosSin is not None):
            mCosSin = misc_utils.static_assert_is_Tensor(epi_args.mCosSin)
            misc_utils.static_assert(get_dtype(mCosSin) is self.epi_dtype)
            epi_smem_bytes_per_stage_pld = epi_smem_bytes_per_stage_pld + (
                epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                    mTensor=mCosSin,
                    epi_tile=epi_tile,
                )
            )

        # output storing: consumer-staged
        if cutlass.const_expr(epi_args.mOutput is not None):
            mOutput = misc_utils.static_assert_is_Tensor(epi_args.mOutput)
            misc_utils.static_assert(get_dtype(mOutput) is self.epi_dtype)
            epi_smem_bytes_per_stage_cst = epi_smem_bytes_per_stage_cst + (
                epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                    mTensor=mOutput,
                    epi_tile=epi_tile,
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
    O: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with RoPE positional encoding.

    Applies RoPE rotation to the GEMM accumulator and stores the rotated
    result to O, while the unmodified accumulator is stored to D by the
    standard epilogue.

    Args:
        shape_mnkl: Problem shape (M, N, K, L).
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        O: Output tensor for RoPE-rotated result of shape (M, N).
        cos_sin: Interleaved cos/sin tensor of shape (M, N).

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    epi_dtype = torch2cute_dtype_map[cos_sin.dtype]

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTRoPE(
        acc_dtype=acc_dtype,
        epi_dtype=epi_dtype,
        tile_shape_mnk=tile_shape_mnk,
        buffer_align_bytes=buffer_align_bytes,
    )

    epi_args = EVTRoPE.EpilogueArguments(
        mCosSin=cos_sin,
        mOutput=O,
    )

    epi_keys = (
        cos_sin.dtype,
        O.dtype,
        EVTRoPE,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
