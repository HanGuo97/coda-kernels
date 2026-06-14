import cutlass
import cutlass.cute as cute
from typing import NamedTuple
from dataclasses import dataclass

from coda.core.ops import misc_utils
from coda.core.ops import struct_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops import epilogue_utils
from coda.core.ops import reduction_utils
from coda.core.epilogue.base import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)

HOPPER_WARP_REDUCTION_WIDTH = 4


class EVTPartialCrossEntropy(EpilogueVisitorTree):
    """
    Computes per-N-tile partial log-sum-exp alongside full logit output.

    Maintains online-softmax (max, sse) state per row in registers during accumulation
    and folds to lse = max + log(sse) at end-of-tile. The resulting per-tile LSE vector
    must be reduced across tiles to obtain the row's final logsumexp.

    Inputs:
        - GEMM output: [M x N]
          * Typically M = batch_size * seq_len, N = vocab_size

    Outputs:
        - Logits (to C): [M x N] (unchanged GEMM output)
        - Partial LSE (mLSEVec): [L, M, num_N_tiles] in fp32
          * Per-row fused LSE for each N-tile: max + log(sum(exp(x - max)))
          * num_N_tiles = (N + tile_N - 1) // tile_N
          * L is batch dimension

    Implementation:
        consumer_visit calls online_softmax_combine_singleton per element to update
        the running (max, sse) pair in registers; OOB N-columns are skipped. At
        consumer_end, online_softmax_combine_warp merges the 4 Hopper lanes' partial
        states into a per-row (max, sse), which is folded to lse and stored. An
        empty-tile guard substitutes max=0 when no element was observed so the
        fastmath log(0) flows to a clean -inf.

    Use cases:
        - Fused GEMM + cross-entropy in transformer training
        - Partial-LSE handoff to a downstream row reduction (e.g. quack.cross_entropy)

    Example:
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTPartialCrossEntropy(dtype=acc_dtype, tile_shape_mnk=tile_shape_mnk)
        >>> epi_args = EVTPartialCrossEntropy.EpilogueArguments(
        ...     mLSEVec=lse_tensor,  # [L, M, num_N_tiles] fp32
        ... )
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mLSEVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mLSEVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        pass

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDrMaxVec: cute.Tensor | None
        tDrSSEVec: cute.Tensor | None
        tDcLogits: cute.Tensor | None
        m_offset_tile: cute.Int32
        n_offset_tile: cute.Int32

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tDrMaxVec_epi: cute.Tensor | None
        tDrSSEVec_epi: cute.Tensor | None
        tDcLogits_epi: cute.Tensor | None
        m_offset_tile: cute.Int32
        n_offset_tile: cute.Int32

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.arch = 90
        self.dtype = dtype
        self.tile_shape_mnk = tile_shape_mnk
        self.max_op = reduction_utils.get_registered_reduction_op(
            name="max",
            element_type=dtype,
        )
        self.add_op = reduction_utils.get_registered_reduction_op(
            name="add",
            element_type=dtype,
        )

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:

        if cutlass.const_expr(epi_args.mLSEVec is not None):
            mLSEVec = misc_utils.static_assert_is_Tensor(epi_args.mLSEVec)
            mLSEVec = layout_utils.assumed_align_stride(
                mLSEVec,
                assumed_align=4,
            )
        else:
            mLSEVec = None

        return self.EpilogueParams(
            mLSEVec=mLSEVec,
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
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mLSEVec is not None):
            rMaxVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(1, 0),
            )
            rMaxVec_view = creation_utils.allocate_tensor_from_layout(
                layout=rMaxVec_view_layout,
                dtype=cute.Float32,
                memspace="rmem",
                smem_allocator=None,
            )
            tDrMaxVec = partition_for_epilogue(rMaxVec_view)
            cute.filter_zeros(tDrMaxVec).fill(-cute.Float32.inf)

            rSSEVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(1, 0),
            )
            rSSEVec_view = creation_utils.allocate_tensor_from_layout(
                layout=rSSEVec_view_layout,
                dtype=cute.Float32,
                memspace="rmem",
                smem_allocator=None,
            )
            tDrSSEVec = partition_for_epilogue(rSSEVec_view)
            cute.filter_zeros(tDrSSEVec).fill(0.0)

            cLogits_view = cute.make_identity_tensor((tile_M, tile_N))
            tDcLogits = partition_for_epilogue(cLogits_view)
        else:
            tDrMaxVec = None
            tDrSSEVec = None
            tDcLogits = None

        return self.EpilogueTensors(
            tDrMaxVec=tDrMaxVec,
            tDrSSEVec=tDrSSEVec,
            tDcLogits=tDcLogits,
            m_offset_tile=m_idx * tile_M,
            n_offset_tile=n_idx * tile_N,
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
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mLSEVec is not None):
            mLSEVec = misc_utils.static_assert_is_Tensor(epi_params.mLSEVec)
            tDrMaxVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrMaxVec)
            tDrSSEVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrSSEVec)
            col_vec_limit_m = min(shape_mnk[0] - m_idx * tile_M, tile_M)
            col_vec_limit_n = mLSEVec.shape[2]

            # avoid duplicating operations on broadcast modes
            tDrMaxVec_filtered = cute.filter_zeros(tDrMaxVec)
            tDrSSEVec_filtered = cute.filter_zeros(tDrSSEVec)
            if cutlass.const_expr(self.arch != 100):
                # Hopper tensor core layout is such that each 4 consecutive
                # threads collectively hold the values of one row tile
                misc_utils.static_assert(cute.size(tDrMaxVec_filtered) == cute.size(tDrSSEVec_filtered))
                for i in cutlass.range_constexpr(cute.size(tDrMaxVec_filtered)):
                    tDrMaxVec_filtered[i], tDrSSEVec_filtered[i] = reduction_utils.online_softmax_combine_warp(
                        m=tDrMaxVec_filtered[i],
                        s=tDrSSEVec_filtered[i],
                        width=HOPPER_WARP_REDUCTION_WIDTH,
                    )
            else:
                # Don't need warp_reduce since we load from tmem with one thread per row
                raise NotImplementedError

            mLSEVec = mLSEVec[batch_idx, None, n_idx]
            gLSEVec = cute.local_tile(mLSEVec, (tile_M,), (m_idx,))
            cLSEVec = cute.make_identity_tensor((tile_M, tile_N))

            tDcLSEVec = partition_for_epilogue(cLSEVec)
            tDrMaxVec_m = layout_utils.select_nonzero_stride_modes(tDrMaxVec, tDrMaxVec.layout)
            tDrSSEVec_m = layout_utils.select_nonzero_stride_modes(tDrSSEVec, tDrSSEVec.layout)
            tDcLSEVec_m = layout_utils.select_nonzero_stride_modes(tDcLSEVec, tDrMaxVec.layout)

            if n_idx < col_vec_limit_n and tDcLSEVec_m[0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(tDcLSEVec_m, mode=[0])):
                    row_idx = tDcLSEVec_m[m][0]
                    if row_idx < col_vec_limit_m:
                        # Empty-tile guard: if no element was ever observed,
                        # max stays at -inf and sse at 0. Substitute 0 for max
                        # so the fastmath log(0) flows to a clean -inf instead
                        # of -inf + log(0), which is implementation-defined
                        # under fastmath.
                        row_max = (
                            tDrMaxVec_m[m]
                            if tDrMaxVec_m[m] > -cute.Float32.inf
                            else cute.Float32.zero
                        )
                        lse = row_max + cute.math.log(tDrSSEVec_m[m], fastmath=True)
                        gLSEVec[row_idx] = lse.to(dtype=misc_utils.get_dtype(gLSEVec))

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:

        if cutlass.const_expr(epi_tensors.tDrMaxVec is not None):
            tDrMaxVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrMaxVec)
            tDrMaxVec_cur = cute.group_modes(tDrMaxVec, 3, cute.rank(tDrMaxVec))
            tDrMaxVec_cur = tDrMaxVec_cur[None, None, None, epi_coord]

            tDrSSEVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrSSEVec)
            tDrSSEVec_cur = cute.group_modes(tDrSSEVec, 3, cute.rank(tDrSSEVec))
            tDrSSEVec_cur = tDrSSEVec_cur[None, None, None, epi_coord]

            tDcLogits = misc_utils.static_assert_is_Tensor(epi_tensors.tDcLogits)
            tDcLogits_cur = cute.group_modes(tDcLogits, 3, cute.rank(tDcLogits))
            tDcLogits_cur = tDcLogits_cur[None, None, None, epi_coord]
        else:
            tDrMaxVec_cur = None
            tDrSSEVec_cur = None
            tDcLogits_cur = None

        return (
            self.EpilogueTensorsLoop(
                tDrMaxVec_epi=tDrMaxVec_cur,
                tDrSSEVec_epi=tDrSSEVec_cur,
                tDcLogits_epi=tDcLogits_cur,
                m_offset_tile=epi_tensors.m_offset_tile,
                n_offset_tile=epi_tensors.n_offset_tile,
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

        if cutlass.const_expr(epi_tensors_loop.tDrMaxVec_epi is not None):
            misc_utils.static_assert(epi_tensors_loop.tDrSSEVec_epi is not None)
            tDrMaxVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrMaxVec_epi)
            tDrSSEVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrSSEVec_epi)
            tDcLogits_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDcLogits_epi)

            if cutlass.const_expr(self.arch < 100):
                misc_utils.static_assert(cute.size(tDrMaxVec_epi) == cute.size(tDrSSEVec_epi))
                for i in cutlass.range_constexpr(cute.size(tDrMaxVec_epi)):
                    # Skip OOB N-columns when N % tile_N != 0. Without this,
                    # OOB lanes feed `tRS_rD = 0` (GEMM accumulator default)
                    # into combine_singleton, anchoring max at 0 and adding
                    # spurious exp(0 - max) terms to the row's sse.
                    row_idx = tDcLogits_epi[i][0]
                    col_idx = tDcLogits_epi[i][1]
                    row_idx_offset = row_idx + epi_tensors_loop.m_offset_tile
                    col_idx_offset = col_idx + epi_tensors_loop.n_offset_tile

                    if col_idx_offset < shape_mnk[1]:
                        tDrMaxVec_epi[i], tDrSSEVec_epi[i] = reduction_utils.online_softmax_combine_singleton(
                            m0=tDrMaxVec_epi[i],
                            m1=tRS_rD[i],
                            s0=tDrSSEVec_epi[i],
                            s1=misc_utils.get_dtype(tDrSSEVec_epi)(1),
                        )
            else:
                raise NotImplementedError
        else:
            tDrMaxVec_epi = epi_tensors_loop.tDrMaxVec_epi
            tDrSSEVec_epi = epi_tensors_loop.tDrSSEVec_epi
            tDcLogits_epi = epi_tensors_loop.tDcLogits_epi

        return self.EpilogueTensorsLoop(
            tDrMaxVec_epi=tDrMaxVec_epi,
            tDrSSEVec_epi=tDrSSEVec_epi,
            tDcLogits_epi=tDcLogits_epi,
            m_offset_tile=epi_tensors_loop.m_offset_tile,
            n_offset_tile=epi_tensors_loop.n_offset_tile,
        )


class EVTSelectLogits(EpilogueVisitorTree):
    """
    Extracts target token logits at specified indices for each row.

    Inputs:
        - GEMM output: [M x N]
          * Typically M = batch_size * seq_len, N = vocab_size
        - Target indices (mTarget): [L, M] (L is batch dim)
          * Column indices to extract from each row, values in [0, N)

    Outputs:
        - Logits (to C): [M x N] (unchanged GEMM output)
        - Selected logits (mLogits): [L, M]
          * mLogits[b, i] = GEMM_output[i, target[b, i]]

    Implementation:
        Loads target indices to shared memory via cp.async, broadcasts to registers, then
        conditionally writes matching logits when col_idx == target[row_idx].

    Use cases:
        - Cross-entropy loss in LLM training
        - Ground-truth token probability extraction
        - Token-level metric computation

    Example:
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTSelectLogits(dtype=cute.Int32, tile_shape_mnk=tile_shape_mnk)
        >>> epi_args = EVTSelectLogits.EpilogueArguments(
        ...     mTarget=target_index_tensor,  # [L, M]
        ...     mLogits=target_logit_tensor,  # [L, M]
        ... )
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mTarget: cute.Tensor | None
        mLogits: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mTarget: cute.Tensor | None
        mLogits: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sTarget: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        gLogits: cute.Tensor | None
        tDsTarget: cute.Tensor | None
        tDcLogits: cute.Tensor | None
        m_offset_tile: cute.Int32
        n_offset_tile: cute.Int32

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        gLogits: cute.Tensor | None
        tDrTarget_epi: cute.Tensor | None
        tDcLogits_epi: cute.Tensor | None
        m_offset_tile: cute.Int32
        n_offset_tile: cute.Int32

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.arch = 90
        self.dtype = dtype
        self.tile_shape_mnk = tile_shape_mnk

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:

        if cutlass.const_expr(epi_args.mTarget is not None):
            mTarget = misc_utils.static_assert_is_Tensor(epi_args.mTarget)
            mTarget = layout_utils.assumed_align_stride(
                mTarget,
                assumed_align=4,
            )
        else:
            mTarget = None

        if cutlass.const_expr(epi_args.mLogits is not None):
            mLogits = misc_utils.static_assert_is_Tensor(epi_args.mLogits)
            mLogits = layout_utils.assumed_align_stride(
                mLogits,
                assumed_align=4,
            )
        else:
            mLogits = None

        return self.EpilogueParams(
            mTarget=mTarget,
            mLogits=mLogits,
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
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mTarget is not None):
            mTarget = misc_utils.static_assert_is_Tensor(epi_params.mTarget)
            sTarget = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sTarget)
            mTarget = mTarget[batch_idx, None]
            gTarget = cute.local_tile(mTarget, (tile_M,), (m_idx,))
            cTarget = cute.make_identity_tensor(tile_M)
            limit_m = min(mTarget.shape[0] - m_idx * tile_M, tile_M)
            memory_utils.g2s_copy_1d(
                src=gTarget,
                dst=sTarget,
                crd=cTarget,
                shape=(limit_m,),
                num_threads=epi_num_threads,
                thread_index=tidx,
            )
            sTarget_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(1, 0),
            )
            sTarget_view = cute.make_tensor(
                iterator=sTarget.iterator,
                layout=sTarget_view_layout,
            )
            tDsTarget = partition_for_epilogue(sTarget_view)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epi_barrier.arrive_and_wait()
        else:
            tDsTarget = None

        if cutlass.const_expr(epi_params.mLogits is not None):
            mLogits = misc_utils.static_assert_is_Tensor(epi_params.mLogits)
            mLogits = mLogits[batch_idx, None]
            gLogits = cute.local_tile(mLogits, (tile_M,), (m_idx,))

            cLogits_view = cute.make_identity_tensor((tile_M, tile_N))
            tDcLogits = partition_for_epilogue(cLogits_view)
        else:
            gLogits = None
            tDcLogits = None

        return self.EpilogueTensors(
            gLogits=gLogits,
            tDsTarget=tDsTarget,
            tDcLogits=tDcLogits,
            m_offset_tile=m_idx * tile_M,
            n_offset_tile=n_idx * tile_N,
        )

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:

        if cutlass.const_expr(epi_tensors.tDsTarget is not None):
            tDsTarget = misc_utils.static_assert_is_Tensor(epi_tensors.tDsTarget)
            tDsTarget_cur = cute.group_modes(tDsTarget, 3, cute.rank(tDsTarget))
            tDsTarget_cur = tDsTarget_cur[None, None, None, epi_coord]
            tDrTarget_cvt = memory_utils.s2r_copy_1d(tDsTarget_cur, dtype=self.dtype)
        else:
            tDrTarget_cvt = None

        if cutlass.const_expr(epi_tensors.tDcLogits is not None):
            tDcLogits = misc_utils.static_assert_is_Tensor(epi_tensors.tDcLogits)
            tDcLogits_cur = cute.group_modes(tDcLogits, 3, cute.rank(tDcLogits))
            tDcLogits_cur = tDcLogits_cur[None, None, None, epi_coord]
            tDcLogits_cvt = tDcLogits_cur
        else:
            tDcLogits_cvt = None

        return (
            self.EpilogueTensorsLoop(
                gLogits=epi_tensors.gLogits,
                tDrTarget_epi=tDrTarget_cvt,
                tDcLogits_epi=tDcLogits_cvt,
                m_offset_tile=epi_tensors.m_offset_tile,
                n_offset_tile=epi_tensors.n_offset_tile,
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

        if cutlass.const_expr(epi_tensors_loop.tDrTarget_epi is not None):
            misc_utils.static_assert(epi_tensors_loop.tDcLogits_epi is not None)
            gLogits = misc_utils.static_assert_is_Tensor(epi_tensors_loop.gLogits)
            tDrTarget_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrTarget_epi)
            tDcLogits_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDcLogits_epi)
            logit_type = misc_utils.get_dtype(gLogits)

            misc_utils.static_assert(cute.size(tDrTarget_epi) == cute.size(tDcLogits_epi))
            for i in cutlass.range_constexpr(cute.size(tDrTarget_epi)):
                target  = tDrTarget_epi[i]
                row_idx = tDcLogits_epi[i][0]
                col_idx = tDcLogits_epi[i][1]
                row_idx_offset = row_idx + epi_tensors_loop.m_offset_tile
                col_idx_offset = col_idx + epi_tensors_loop.n_offset_tile

                if row_idx_offset < shape_mnk[0] and col_idx_offset == target:
                    target_logit = logit_type(tRS_rD[i])
                    gLogits[row_idx] = target_logit

        else:
            gLogits = epi_tensors_loop.gLogits
            tDrTarget_epi = epi_tensors_loop.tDrTarget_epi
            tDcLogits_epi = epi_tensors_loop.tDcLogits_epi

        return self.EpilogueTensorsLoop(
            gLogits=gLogits,
            tDrTarget_epi=tDrTarget_epi,
            tDcLogits_epi=tDcLogits_epi,
            m_offset_tile=epi_tensors_loop.m_offset_tile,
            n_offset_tile=epi_tensors_loop.n_offset_tile,
        )

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:

        if cutlass.const_expr(epi_params.mTarget is not None):
            mTarget = misc_utils.static_assert_is_Tensor(epi_params.mTarget)
            col_vec_dtype = misc_utils.get_dtype(mTarget)
            col_vec_smem_size = epilogue_utils.get_smem_size_vector(
                mTensor=mTarget,
                epi_tile=self.tile_shape_mnk[0],
                epi_num_threads=epi_num_threads,
            )
        else:
            col_vec_dtype = cute.Int64
            col_vec_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            sTarget: cute.struct.Align[cute.struct.MemRange[col_vec_dtype, col_vec_smem_size], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mTarget is not None):
            sTarget_layout = cute.make_layout(self.tile_shape_mnk[0])
            sTarget = storage.sTarget.get_tensor(sTarget_layout)
        else:
            sTarget = None

        return self.EpilogueTensorsSMem(
            sTarget=sTarget,
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

        # we load the entire logits vectors upfront, hence the smem
        # storage is fixed and independent of stages
        if cutlass.const_expr(epi_args.mTarget is not None):
            mTarget = misc_utils.static_assert_is_Tensor(epi_args.mTarget)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + (
                epilogue_utils.get_epi_smem_bytes_per_stage_fixed_vector(
                    mTensor=mTarget,
                    epi_tile=self.tile_shape_mnk[0],
                    epi_num_threads=epi_num_threads,
                )
            )

        return (
            epi_smem_bytes_fixed,
            epi_smem_bytes_per_stage_cst,
            epi_smem_bytes_per_stage_pld,
        )
