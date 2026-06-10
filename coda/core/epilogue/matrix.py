import cutlass
import cutlass.cute as cute
from typing import Callable, NamedTuple
from dataclasses import dataclass

from hilt.dtype_utils import get_dtype
from coda.core.ops import misc_utils
from coda.core.ops import dtype_utils
from coda.core.ops import struct_utils
from coda.core.ops import pipeline_utils
from coda.core.ops import epilogue_utils
from coda.core.epilogue.base import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)


class EVTMatrixLoad(EpilogueVisitorTree):
    """
    Base class for loading and fusing matrix tensors with GEMM output element-wise.

    This is a base class providing multi-stage TMA loading infrastructure. For direct use, prefer:
        - EVTResidual: Simple element-wise addition (sets container_dtype automatically)
        - EVTMatrixLoad2X: Custom function with 2x matrix elements per output

    Inputs:
        - GEMM output: [M x N]
        - Matrix (mMatrix): [M, N, L]
          * L is batch dimension (batch-last)
          * Element-wise fusion with GEMM output

    Output:
        - D = fn((A @ B), Matrix): [M x N]
        - Default: element-wise addition

    Implementation:
        Multi-stage TMA pipeline with producer-consumer pattern. Producer warps
        asynchronously load matrix tiles to shared memory while consumer warps
        fuse with accumulator in registers. Overlaps loading with computation.

    Use cases:
        - Base class for custom matrix loading epilogue operations
        - Extend to implement specialized loading patterns with custom fusion functions
        - See EVTResidual and EVTMatrixLoad2X for concrete implementations

    Example:
        >>> # Not typically used directly - see EVTResidual or EVTMatrixLoad2X
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTMatrixLoad(
        ...         acc_dtype=acc_dtype,
        ...         epi_dtype=cute.Float16,
        ...         container_dtype=cute.Float16,
        ...         tile_shape_mnk=tile_shape_mnk,
        ...         buffer_align_bytes=buffer_align_bytes,
        ...     )
        >>> epi_args = EVTMatrixLoad.EpilogueArguments(
        ...     mMatrix=matrix_tensor,  # [M, N, L]
        ... )
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mMatrix: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mMatrix: cute.Tensor | None
        epi_tma_atom: cute.CopyAtom
        epi_gmem_layout: cutlass.utils.LayoutEnum
        epi_smem_layout_staged: cute.Layout

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sMatrix: cute.Tensor | None
        epi_load_pipeline_array_ptr: cute.Pointer

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDsMatrix: cute.Tensor | None
        tDgMatrix: cute.Tensor | None
        tSR_sMatrix: cute.Tensor | None
        tRS_rMatrix: cute.Tensor | None
        tSR_rMatrix: cute.Tensor | None
        tiled_copy_s2r: cute.TiledCopy

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tRS_rMatrix: cute.Tensor | None

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
        container_dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        buffer_align_bytes: int,
    ) -> None:
        super().__init__()
        self.acc_dtype = acc_dtype
        self.epi_dtype = epi_dtype
        self.container_dtype = container_dtype
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

        if cutlass.const_expr(epi_args.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_args.mMatrix)
            misc_utils.static_assert(get_dtype(mMatrix) is self.epi_dtype)
            if cutlass.const_expr(self.container_dtype is not self.epi_dtype):
                mMatrix = cute.recast_tensor(
                    mMatrix,
                    dtype=self.container_dtype,
                )

            (
                epi_gmem_layout,
                epi_smem_layout_staged,
                epi_tma_atom,
                epi_tma_tensor,
            ) = epilogue_utils.prepare_tma(
                tma_op="g2s",
                epi_tile=epi_tile,
                # we use `epi_load_stage` here
                # since this is for loading
                epi_stage=epi_load_stage,
                epi_tensor=mMatrix,
            )

        return self.EpilogueParams(
            mMatrix=epi_tma_tensor,
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
    def prepare_pipelines(
        self,
        epi_load_stage: int,
        epi_num_warps: int,
        epi_params: EpilogueParams,
        epi_tensors_smem: EpilogueTensorsSMem,
    ) -> EpiloguePipelines:
        if cutlass.const_expr(epi_params.mMatrix is not None):
            epi_smem_layout = cute.slice_(epi_params.epi_smem_layout_staged, (None, None, 0))
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

        # https://github.com/NVIDIA/cutlass/issues/2931
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

        if cutlass.const_expr(epi_params.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_params.mMatrix)
            sMatrix = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sMatrix)
            # (bM, bN)
            gMatrix = mMatrix[None, None, batch_idx]
            gMatrix = cute.local_tile(gMatrix, (tile_M, tile_N), (m_idx, n_idx))
            gMatrix = cute.zipped_divide(gMatrix, epi_tile)

            # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
            tDsMatrix, tDgMatrix = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.epi_tma_atom,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sMatrix, 0, cute.rank(sMatrix) - 1),
                gmem_tensor=cute.group_modes(gMatrix, 0, cute.rank(gMatrix) - 1),
            )
            filter_zeros = cutlass.const_expr(False)
            if cutlass.const_expr(filter_zeros):
                tDsMatrix = cute.filter_zeros(tDsMatrix)
                tDgMatrix = cute.filter_zeros(tDgMatrix)

            tiled_copy_s2r, _, tSR_sMatrix, tRS_rMatrix, tSR_rMatrix = epilogue_utils.prepare_copy_s2r_sm90(
                tiled_mma=tiled_mma,
                tidx=tidx,
                src=sMatrix,
                dst_layout=tRS_rD_layout,
                epi_dtype=self.epi_dtype,
                container_dtype=self.container_dtype,
                epi_gmem_layout=epi_params.epi_gmem_layout,
                epi_num_matrices=epi_num_matrices,
            )

        return self.EpilogueTensors(
            tDsMatrix=tDsMatrix,
            tDgMatrix=tDgMatrix,
            tSR_sMatrix=tSR_sMatrix,
            tRS_rMatrix=tRS_rMatrix,
            tSR_rMatrix=tSR_rMatrix,
            tiled_copy_s2r=tiled_copy_s2r,
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
        if cutlass.const_expr(epi_tensors.tDsMatrix is not None):
            tSR_sMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tSR_sMatrix)
            tSR_rMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tSR_rMatrix)
            tRS_rMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tRS_rMatrix)

            tiled_copy = epi_tensors.tiled_copy_s2r
            src = tSR_sMatrix[None, None, None, epi_load_consumer_state.index]
            dst = tSR_rMatrix

            epi_load_pipeline.consumer_wait(epi_load_consumer_state)
            cute.copy(atom=tiled_copy, src=src, dst=dst)
            # Fence to make sure shared memory read is visible to TMA load
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                epi_load_pipeline.consumer_release(epi_load_consumer_state)
            epi_load_consumer_state.advance()

        return (
            self.EpilogueTensorsLoop(
                tRS_rMatrix=tRS_rMatrix,
            ),
            # https://github.com/NVIDIA/cutlass/issues/2931
            self.EpiloguePipelines(
                epi_load_pipeline=epi_load_pipeline,
                epi_load_consumer_state=epi_load_consumer_state,
                epi_load_producer_state=epi_load_producer_state,
            ),
        )

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
        if cutlass.const_expr(epi_params.mMatrix is not None):
            tDgMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDgMatrix)
            tDsMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDsMatrix)

            epi_prefetch = cutlass.min(epi_tile_num, epi_load_stage)
            for epi_idx in cutlass.range(epi_prefetch, unroll=1):
                epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
                if is_tma_warp:
                    atom = epi_params.epi_tma_atom
                    src = tDgMatrix[None, epi_coord]
                    dst = tDsMatrix[None, epi_load_producer_state.index]
                    tma_bar_ptr = epi_load_pipeline.producer_get_barrier(epi_load_producer_state)
                    epi_load_pipeline.producer_acquire(epi_load_producer_state)
                    cute.copy(atom=atom, src=src, dst=dst, tma_bar_ptr=tma_bar_ptr)
                    epi_load_pipeline.producer_commit(epi_load_producer_state)
                epi_load_producer_state.advance()

        # https://github.com/NVIDIA/cutlass/issues/2931
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
        if cutlass.const_expr(epi_params.mMatrix is not None and epi_idx + epi_load_stage < epi_tile_num):
            tDgMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDgMatrix)
            tDsMatrix = misc_utils.static_assert_is_Tensor(epi_tensors.tDsMatrix)
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx + epi_load_stage)

            if is_tma_warp:
                atom = epi_params.epi_tma_atom
                src = tDgMatrix[None, epi_coord]
                dst = tDsMatrix[None, epi_load_producer_state.index]
                tma_bar_ptr = epi_load_pipeline.producer_get_barrier(epi_load_producer_state)

                epi_load_pipeline.producer_acquire(epi_load_producer_state)
                cute.copy(atom=atom, src=src, dst=dst, tma_bar_ptr=tma_bar_ptr)
                epi_load_pipeline.producer_commit(epi_load_producer_state)
            epi_load_producer_state.advance()

        # https://github.com/NVIDIA/cutlass/issues/2931
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

        if cutlass.const_expr(epi_params.mMatrix is not None):
            epi_smem_size = cute.cosize(epi_params.epi_smem_layout_staged)
        else:
            epi_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            epi_load_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, epi_load_stage * 2]
            sMatrix: cute.struct.Align[cute.struct.MemRange[self.container_dtype, epi_smem_size], self.buffer_align_bytes]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mMatrix is not None):
            epi_load_pipeline_array_ptr = storage.epi_load_pipeline_array_ptr.data_ptr()
            sMatrix = storage.sMatrix.get_tensor(
                epi_params.epi_smem_layout_staged.outer,
                swizzle=epi_params.epi_smem_layout_staged.inner,
            )
        else:
            sMatrix = None
            epi_load_pipeline_array_ptr = None

        return self.EpilogueTensorsSMem(
            sMatrix=sMatrix,
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

        # we load matrix tensors, hence this is under
        # epilogue producers, and since this is done in
        # stages, the smem storage is per stage
        if cutlass.const_expr(epi_args.mMatrix is not None):
            mMatrix = misc_utils.static_assert_is_Tensor(epi_args.mMatrix)
            misc_utils.static_assert(get_dtype(mMatrix) is self.epi_dtype)
            if cutlass.const_expr(self.container_dtype is not self.epi_dtype):
                mMatrix = cute.recast_tensor(
                    mMatrix,
                    dtype=self.container_dtype,
                )

            # when TMA is used, the tile size should be `cute.size(epi_tile)`
            epi_smem_bytes_per_stage_pld = epi_smem_bytes_per_stage_pld + (
                epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                    mTensor=mMatrix,
                    epi_tile=epi_tile,
                )
            )

        return (
            epi_smem_bytes_fixed,
            epi_smem_bytes_per_stage_cst,
            epi_smem_bytes_per_stage_pld,
        )


class EVTResidual(EVTMatrixLoad):
    """
    Adds residual matrix tensor to GEMM output element-wise.

    Specialized version of EVTMatrixLoad for simple element-wise addition,
    automatically setting container_dtype to match epi_dtype.

    Inputs:
        - GEMM output: [M x N]
        - Matrix (mMatrix): [M, N, L]
          * L is batch dimension (batch-last)
          * Element-wise addition with GEMM output

    Output:
        - D = (A @ B) + Matrix: [M x N]

    Implementation:
        Multi-stage TMA pipeline with producer-consumer pattern. Producer warps
        asynchronously load matrix tiles to shared memory while consumer warps
        add to accumulator in registers. Overlaps loading with computation.

    Use cases:
        - Residual connections in neural networks
        - Fused GEMM + element-wise add in transformers
        - Skip connections requiring minimal overhead

    Example:
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTResidual(
        ...         acc_dtype=acc_dtype,
        ...         epi_dtype=cute.Float16,
        ...         tile_shape_mnk=tile_shape_mnk,
        ...         buffer_align_bytes=buffer_align_bytes,
        ...     )
        >>> epi_args = EVTResidual.EpilogueArguments(
        ...     mMatrix=residual_tensor,  # [M, N, L]
        ... )
    """

    EpilogueArguments = EVTMatrixLoad.EpilogueArguments
    EpilogueParams = EVTMatrixLoad.EpilogueParams
    EpilogueTensorsSMem = EVTMatrixLoad.EpilogueTensorsSMem
    EpilogueTensors = EVTMatrixLoad.EpilogueTensors
    EpilogueTensorsLoop = EVTMatrixLoad.EpilogueTensorsLoop
    EpiloguePipelines = EVTMatrixLoad.EpiloguePipelines

    def __init__(
        self,
        acc_dtype: type[cute.Numeric],
        epi_dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        buffer_align_bytes: int,
    ) -> None:
        super().__init__(
            acc_dtype=acc_dtype,
            epi_dtype=epi_dtype,
            container_dtype=epi_dtype,
            tile_shape_mnk=tile_shape_mnk,
            buffer_align_bytes=buffer_align_bytes,
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
            rC = tRS_rC.load()
            rD = tRS_rD.load()
            rC = dtype_utils.convert(rC, dtype=get_dtype(tRS_rD))
            tRS_rD.store(rD + rC)
        else:
            tRS_rC = epi_tensors_loop.tRS_rMatrix

        return self.EpilogueTensorsLoop(
            tRS_rMatrix=tRS_rC,
        )


class EVTMatrixLoad2X(EVTMatrixLoad):
    """
    Applies custom function to GEMM output using paired matrix elements.

    Loads matrix tensor with 2x elements per output position (along N dimension)
    and applies custom fusion function to GEMM accumulator.

    Inputs:
        - GEMM output: [M x N]
        - Matrix (mMatrix): [M, N*2, L]
          * L is batch dimension (batch-last)
          * 2x elements in N dimension: pairs at (2*i, 2*i+1) fused per output[i]

    Output:
        - D = fn((A @ B), Matrix[..., 2*i], Matrix[..., 2*i+1]): [M x N]
        - fn: Custom function taking (accumulator, elem0, elem1) -> result

    Implementation:
        Multi-stage TMA pipeline with producer-consumer pattern. Producer warps
        asynchronously load matrix tiles to shared memory while consumer warps
        apply the custom function in registers. Loads 2 matrix elements per
        output element (along N dimension), enabling efficient fused operations
        like gating, weighted combinations, or learned interpolations.
        Overlaps loading with computation.

    Use cases:
        - Gated activations or attention mechanisms with learnable gates
        - Weighted combinations of auxiliary tensors (e.g., dual residuals)
        - Fused multiply-add operations with dual matrix operands
        - Complex element-wise operations requiring paired inputs

    Example:
        >>> # Define custom function (e.g., weighted combination)
        >>> fn = lambda acc, x, y: acc + 0.3 * x - 0.7 * y
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTMatrixLoad2X(
        ...         fn=fn,
        ...         acc_dtype=acc_dtype,
        ...         epi_dtype=cute.Float16,
        ...         tile_shape_mnk=tile_shape_mnk,
        ...         buffer_align_bytes=buffer_align_bytes,
        ...     )
        >>> epi_args = EVTMatrixLoad2X.EpilogueArguments(
        ...     mMatrix=matrix_tensor,  # [M, N*2, L]
        ... )
    """

    EpilogueArguments = EVTMatrixLoad.EpilogueArguments
    EpilogueParams = EVTMatrixLoad.EpilogueParams
    EpilogueTensorsSMem = EVTMatrixLoad.EpilogueTensorsSMem
    EpilogueTensors = EVTMatrixLoad.EpilogueTensors
    EpilogueTensorsLoop = EVTMatrixLoad.EpilogueTensorsLoop
    EpiloguePipelines = EVTMatrixLoad.EpiloguePipelines

    def __init__(
        self,
        fn: cutlass.Constexpr[Callable],
        acc_dtype: type[cute.Numeric],
        epi_dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        buffer_align_bytes: int,
    ) -> None:
        misc_utils.static_assert(epi_dtype.width == 16)
        container_dtype = cute.Float32
        super().__init__(
            acc_dtype=acc_dtype,
            epi_dtype=epi_dtype,
            container_dtype=container_dtype,
            tile_shape_mnk=tile_shape_mnk,
            buffer_align_bytes=buffer_align_bytes,
        )
        self.arch = 90
        self.fn = fn

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
            misc_utils.static_assert(get_dtype(tRS_rC) is self.container_dtype)
            tRS_rC = cute.recast_tensor(tRS_rC, dtype=self.epi_dtype)
            tRS_rC = dtype_utils.convert(tRS_rC, dtype=get_dtype(tRS_rD))
            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                    tRS_rD[i] = self.fn(tRS_rD[i], tRS_rC[2 * i], tRS_rC[2 * i + 1])
            else:
                raise NotImplementedError
        else:
            tRS_rC = epi_tensors_loop.tRS_rMatrix

        return self.EpilogueTensorsLoop(
            tRS_rMatrix=tRS_rC,
        )
