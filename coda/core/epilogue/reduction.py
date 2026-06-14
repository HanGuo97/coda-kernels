import cutlass
import cutlass.cute as cute
from typing import NamedTuple
from dataclasses import dataclass

from coda.core.ops import misc_utils
from coda.core.ops import dtype_utils
from coda.core.ops import struct_utils
from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops import reduction_utils
from coda.core.epilogue.base import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)

# For HOPPER, we assume the warp layout is (8, 4)
HOPPER_WARP_REDUCTION_WIDTH = 4
HOPPER_WARP_REDUCTION_DEPTH = 8


class EVTRowOrColBlockReductionLoad(EpilogueVisitorTree):
    """
    Completes row/column reductions from tiled partial reductions.

    Loads per-tile partial reductions from memory, aggregates across all tiles,
    and fuses the result into GEMM output.

    Inputs:
        - GEMM output: [M x N]
        - Column block matrix (mColBlkLd): [L, M, num_N_tiles]
          * Partial column reductions, one per N-tile
          * num_N_tiles = (N + tile_N - 1) // tile_N
          * L is batch dimension

    Outputs:
        - GEMM output + reduction: D = (A @ B) + reduce(col_blocks): [M x N]
        - Final reduced vector (mColVecSt): [L, M] stored to memory

    Supported operations: sum, max, min, product (via reduction_op parameter)

    Implementation:
        Loads tiled partial reductions via cp.async to shared memory, reduces across
        tiles in registers, adds to accumulator during epilogue loop, then stores
        final reduced vector at consumer_end.

    Use cases:
        - Multi-tile row/column reductions for LayerNorm, RMSNorm
        - Online statistics aggregation across large matrices
        - Streaming reductions with two-pass computation

    Note: Row block matrices ([L, N, num_M_tiles]) are not yet implemented.

    Example:
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTRowOrColBlockReductionLoad(
        ...         acc_dtype=acc_dtype,
        ...         reduction_op='add',
        ...         tile_shape_mnk=tile_shape_mnk,
        ...     )
        >>> epi_args = EVTRowOrColBlockReductionLoad.EpilogueArguments(
        ...     mRowBlkLd=None,
        ...     mColBlkLd=col_block_tensor,  # [L, M, num_N_tiles]
        ...     mRowVecSt=None,
        ...     mColVecSt=col_vector_output,  # [L, M]
        ... )
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mRowBlkLd: cute.Tensor | None
        mColBlkLd: cute.Tensor | None
        mRowVecSt: cute.Tensor | None
        mColVecSt: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mRowBlkLd: cute.Tensor | None
        mColBlkLd: cute.Tensor | None
        mRowVecSt: cute.Tensor | None
        mColVecSt: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sRowBlkLd: cute.Tensor | None
        sColBlkLd: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDsRowBlkLd: cute.Tensor | None
        tDsColBlkLd: cute.Tensor | None
        tDrRowVecSt: cute.Tensor | None
        tDrColVecSt: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        # these are vectors from reduced block matrices
        tDrRowVec_epi: cute.Tensor | None
        tDrColVec_epi: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        acc_dtype: type[cute.Numeric],
        reduction_op: str,
        tile_shape_mnk: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.acc_dtype = acc_dtype
        self.reduction_op = reduction_op
        self.tile_shape_mnk = tile_shape_mnk

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:
        if cutlass.const_expr(epi_args.mRowBlkLd is not None):
            raise NotImplementedError
        else:
            mRowBlkLd = None
            mRowVecSt = None

        if cutlass.const_expr(epi_args.mColBlkLd is not None):
            mColBlkLd = misc_utils.static_assert_is_Tensor(epi_args.mColBlkLd)
            mColVecSt = misc_utils.static_assert_is_Tensor(epi_args.mColVecSt)
            col_blk_ld_dtype = misc_utils.get_dtype(mColBlkLd)
            col_vec_st_dtype = misc_utils.get_dtype(mColVecSt)
            col_blk_ld_count = mColBlkLd.shape[2]
            col_blk_ld_nbits = col_blk_ld_dtype.width * col_blk_ld_count
            misc_utils.static_assert(col_blk_ld_nbits % 128 == 0)
            misc_utils.static_assert(col_blk_ld_dtype is col_vec_st_dtype)
            mColBlkLd = layout_utils.assumed_align_stride(
                mColBlkLd,
                assumed_align=16,
            )
            mColVecSt = layout_utils.assumed_align_stride(
                mColVecSt,
                assumed_align=16,
            )
        else:
            mColBlkLd = None
            mColVecSt = None

        return self.EpilogueParams(
            mRowBlkLd=mRowBlkLd,
            mColBlkLd=mColBlkLd,
            mRowVecSt=mRowVecSt,
            mColVecSt=mColVecSt,
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
        # Don't need sync as we assume the previous epilogue has finished

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mRowBlkLd is not None):
            raise NotImplementedError
        else:
            tDsRowBlkLd = None
            tDrRowVecSt = None

        if cutlass.const_expr(epi_params.mColBlkLd is not None):
            mColBlkLd = misc_utils.static_assert_is_Tensor(epi_params.mColBlkLd)
            sColBlkLd = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sColBlkLd)

            col_blk_ld_count = mColBlkLd.shape[2]
            mColBlkLd = mColBlkLd[batch_idx, None, None]
            gColBlkLd = cute.local_tile(mColBlkLd, (tile_M, col_blk_ld_count), (m_idx, 0))
            cColBlkLd = cute.make_identity_tensor((tile_M, col_blk_ld_count))
            limit_m = min(mColBlkLd.shape[0] - m_idx * tile_M, tile_M)
            memory_utils.g2s_copy_2d_row_reduction(
                src=gColBlkLd,
                dst=sColBlkLd,
                crd=cColBlkLd,
                shape=(limit_m, col_blk_ld_count),
                num_threads=epi_num_threads,
                thread_index=tidx,
            )
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N, col_blk_count)
            sColBlkLd_view_layout = cute.make_layout(
                shape=(sColBlkLd.shape[0], tile_N, sColBlkLd.shape[1]),
                stride=(sColBlkLd.stride[0], 0, sColBlkLd.stride[1]),
            )
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            rColVecSt_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(1, 0),
            )
            sColBlkLd_view = cute.make_tensor(
                iterator=sColBlkLd.iterator,
                layout=sColBlkLd_view_layout,
            )
            rColVecSt_view = creation_utils.allocate_tensor_from_layout(
                layout=rColVecSt_view_layout,
                dtype=cute.Float32,
                memspace="rmem",
                smem_allocator=None,
            )
            tDsColBlkLd = partition_for_epilogue(sColBlkLd_view)
            tDrColVecSt = partition_for_epilogue(rColVecSt_view)
            cute.filter_zeros(tDrColVecSt).fill(0.0)
        else:
            tDsColBlkLd = None
            tDrColVecSt = None

        if cutlass.const_expr(
            (tDsRowBlkLd is not None) or
            (tDsColBlkLd is not None)
        ):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epi_barrier.arrive_and_wait()

        return self.EpilogueTensors(
            tDsRowBlkLd=tDsRowBlkLd,
            tDsColBlkLd=tDsColBlkLd,
            tDrRowVecSt=tDrRowVecSt,
            tDrColVecSt=tDrColVecSt,
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
        m_idx, _, _, batch_idx = tile_coord_mnkl
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mRowVecSt is not None):
            raise NotImplementedError

        if cutlass.const_expr(epi_params.mColVecSt is not None):
            mColVecSt = misc_utils.static_assert_is_Tensor(epi_params.mColVecSt)
            tDrColVecSt = misc_utils.static_assert_is_Tensor(epi_tensors.tDrColVecSt)
            col_vec_limit_m = min(shape_mnk[0] - m_idx * tile_M, tile_M)

            mColVecSt = mColVecSt[batch_idx, None]
            gColVecSt = cute.local_tile(mColVecSt, (tile_M,), (m_idx,))
            cColVecSt = cute.make_identity_tensor((tile_M, tile_N))

            tDcColVecSt = partition_for_epilogue(cColVecSt)
            tDrColVecSt_m = layout_utils.select_nonzero_stride_modes(tDrColVecSt, tDrColVecSt.layout)
            tDcColVecSt_m = layout_utils.select_nonzero_stride_modes(tDcColVecSt, tDrColVecSt.layout)
            if tDcColVecSt_m[0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(tDcColVecSt_m, mode=[0])):
                    row_idx = tDcColVecSt_m[m][0]
                    if row_idx < col_vec_limit_m:
                        gColVecSt[row_idx] = tDrColVecSt_m[m].to(dtype=misc_utils.get_dtype(gColVecSt))

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:
        if cutlass.const_expr(epi_tensors.tDsRowBlkLd is not None):
            raise NotImplementedError
        else:
            tDrRowVecSt_cvt = None

        if cutlass.const_expr(epi_tensors.tDsColBlkLd is not None):
            tDsColBlkLd = misc_utils.static_assert_is_Tensor(epi_tensors.tDsColBlkLd)
            tDrColVecSt = misc_utils.static_assert_is_Tensor(epi_tensors.tDrColVecSt)
            tDsColBlkLd_cur = cute.group_modes(tDsColBlkLd, 3, cute.rank(tDsColBlkLd) - 1)
            tDrColVecSt_cur = cute.group_modes(tDrColVecSt, 3, cute.rank(tDrColVecSt))
            tDsColBlkLd_cur = tDsColBlkLd_cur[None, None, None, epi_coord, None]
            tDrColVecSt_cur = tDrColVecSt_cur[None, None, None, epi_coord]
            tDrColBlkLd_cvt = memory_utils.s2r_copy_1d(tDsColBlkLd_cur, dtype=self.acc_dtype)

            reduction_ssa_op, _, reduction_init_value = reduction_utils.prepare_simple_block_reduction_op(
                name=self.reduction_op,
                element_type=self.acc_dtype,
            )
            tDrColBlk_ssa = tDrColBlkLd_cvt.load()
            tDrColBlk_red = tDrColBlk_ssa.reduce(
                op=reduction_ssa_op,
                init_val=reduction_init_value,
                # reducing the last dimension
                reduction_profile=(None, None, None, 1),
            )
            tDrColVecSt_cur.store(tDrColBlk_red)
        else:
            tDrColVecSt_cur = None

        return (
            self.EpilogueTensorsLoop(
                tDrRowVec_epi=tDrRowVecSt_cvt,
                tDrColVec_epi=tDrColVecSt_cur,
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
        if cutlass.const_expr(epi_tensors_loop.tDrRowVec_epi is not None):
            tDrRowVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrRowVec_epi)
            for i in cutlass.range_constexpr(cute.size(tDrRowVec_epi)):
                tRS_rD[i] = tRS_rD[i] + tDrRowVec_epi[i]
        else:
            tDrRowVec_epi = epi_tensors_loop.tDrRowVec_epi

        if cutlass.const_expr(epi_tensors_loop.tDrColVec_epi is not None):
            tDrColVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrColVec_epi)
            for i in cutlass.range_constexpr(cute.size(tDrColVec_epi)):
                tRS_rD[i] = tRS_rD[i] + tDrColVec_epi[i]
        else:
            tDrColVec_epi = epi_tensors_loop.tDrColVec_epi

        return self.EpilogueTensorsLoop(
            tDrRowVec_epi=tDrRowVec_epi,
            tDrColVec_epi=tDrColVec_epi,
        )

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:
        if cutlass.const_expr(epi_params.mRowBlkLd is not None):
            raise NotImplementedError
        else:
            row_blk_ld_dtype = cute.Float32
            row_blk_ld_smem_size = 0

        if cutlass.const_expr(epi_params.mColBlkLd is not None):
            mColBlkLd = misc_utils.static_assert_is_Tensor(epi_params.mColBlkLd)
            col_blk_ld_dtype = misc_utils.get_dtype(mColBlkLd)
            col_blk_ld_count = mColBlkLd.shape[2]
            col_blk_ld_smem_size = self.tile_shape_mnk[0] * col_blk_ld_count
        else:
            col_blk_ld_dtype = cute.Float32
            col_blk_ld_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            sRowBlkLd: cute.struct.Align[cute.struct.MemRange[row_blk_ld_dtype, row_blk_ld_smem_size], 16]
            sColBlkLd: cute.struct.Align[cute.struct.MemRange[col_blk_ld_dtype, col_blk_ld_smem_size], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mRowBlkLd is not None):
            raise NotImplementedError
        else:
            sRowBlkLd = None

        if cutlass.const_expr(epi_params.mColBlkLd is not None):
            mColBlkLd = misc_utils.static_assert_is_Tensor(epi_params.mColBlkLd)
            col_blk_ld_count = mColBlkLd.shape[2]
            sColBlk_ld_layout = layout_utils.make_ordered_layout(
                shape=(self.tile_shape_mnk[0], col_blk_ld_count),
                order="row",
            )
            sColBlkLd = storage.sColBlkLd.get_tensor(sColBlk_ld_layout)
        else:
            sColBlkLd = None

        return self.EpilogueTensorsSMem(
            sRowBlkLd=sRowBlkLd,
            sColBlkLd=sColBlkLd,
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

        # we load the entire row/col matrices upfront, hence the smem
        # storage is fixed and independent of stages
        if cutlass.const_expr(epi_args.mRowBlkLd is not None):
            raise NotImplementedError

        if cutlass.const_expr(epi_args.mColBlkLd is not None):
            mColBlkLd = misc_utils.static_assert_is_Tensor(epi_args.mColBlkLd)
            col_blk_ld_dtype = misc_utils.get_dtype(mColBlkLd)
            col_blk_ld_count = mColBlkLd.shape[2]
            col_blk_ld_smem_size = self.tile_shape_mnk[0] * col_blk_ld_count
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + (
                col_blk_ld_smem_size * col_blk_ld_dtype.width // 8
            )

        return (
            epi_smem_bytes_fixed,
            epi_smem_bytes_per_stage_cst,
            epi_smem_bytes_per_stage_pld,
        )


class EVTColBlockReductionStore(EpilogueVisitorTree):
    """
    Performs block-level column reductions of GEMM output within each CTA tile.

    Computes a partial column reduction during epilogue: one value per row per
    N-tile. A second pass is needed to aggregate across N-tiles for the full
    column reduction.

    Input:
        - GEMM output: [M x N]

    Output:
        - mColVec: [L, M, num_N_tiles]
          * num_N_tiles = (N + tile_N - 1) // tile_N

    Supported operations: sum, max, min, product (via reduction_op parameter)

    Implementation:
        Hopper output layout has 4 contiguous lanes per M row, so an intra-warp
        reduction across those lanes (width=4) suffices. Each warp owns disjoint
        M rows in the standard wgmma epilogue, so no inter-warp reduction is
        needed.
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        pass

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDrColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tDrColVec_epi: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        reduction_op: reduction_utils.BlockReductionOp,
        tile_shape_mnk: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.arch = 90
        self.reduction_op = reduction_op
        self.tile_shape_mnk = tile_shape_mnk

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:
        if cutlass.const_expr(epi_args.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_args.mColVec)
            mColVec = layout_utils.assumed_align_stride(
                mColVec,
                assumed_align=4,
            )
        else:
            mColVec = None

        return self.EpilogueParams(mColVec=mColVec)

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
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mColVec is not None):
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

        return self.EpilogueTensors(tDrColVec=tDrColVec)

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

        if cutlass.const_expr(epi_params.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_params.mColVec)
            tDrColVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrColVec)
            col_vec_limit_m = min(shape_mnk[0] - m_idx * tile_M, tile_M)
            col_vec_limit_n = mColVec.shape[2]

            # avoid duplicating operations on broadcast modes
            tDrColVec_filtered = cute.filter_zeros(tDrColVec)
            if cutlass.const_expr(self.arch != 100):
                # Hopper tensor core layout is such that each 4 consecutive
                # threads collectively hold the values of one row tile
                for i in cutlass.range_constexpr(cute.size(tDrColVec_filtered)):
                    tDrColVec_filtered[i] = self.reduction_op.warp_reduction_singleton(
                        tDrColVec_filtered[i],
                        width=HOPPER_WARP_REDUCTION_WIDTH,
                    )
            else:
                # Don't need warp_reduce since we load from tmem with one thread per row
                raise NotImplementedError

            mColVec = mColVec[batch_idx, None, n_idx]
            gColVec = cute.local_tile(mColVec, (tile_M,), (m_idx,))
            cColVec = cute.make_identity_tensor((tile_M, tile_N))

            tDcColVec = partition_for_epilogue(cColVec)
            tDrColVec_m = layout_utils.select_nonzero_stride_modes(tDrColVec, tDrColVec.layout)
            tDcColVec_m = layout_utils.select_nonzero_stride_modes(tDcColVec, tDrColVec.layout)
            if n_idx < col_vec_limit_n and tDcColVec_m[0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(tDcColVec_m, mode=[0])):
                    row_idx = tDcColVec_m[m][0]
                    if row_idx < col_vec_limit_m:
                        gColVec[row_idx] = tDrColVec_m[m].to(dtype=misc_utils.get_dtype(gColVec))

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:
        if cutlass.const_expr(epi_tensors.tDrColVec is not None):
            tDrColVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrColVec)
            tDrColVec_cur = cute.group_modes(tDrColVec, 3, cute.rank(tDrColVec))
            tDrColVec_cur = tDrColVec_cur[None, None, None, epi_coord]
        else:
            tDrColVec_cur = None

        return (
            self.EpilogueTensorsLoop(tDrColVec_epi=tDrColVec_cur),
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
        if cutlass.const_expr(epi_tensors_loop.tDrColVec_epi is not None):
            tDrColVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrColVec_epi)

            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tDrColVec_epi)):
                    tDrColVec_epi[i] = self.reduction_op.combine_fn_singleton(tDrColVec_epi[i], tRS_rD[i])
            else:
                raise NotImplementedError
        else:
            tDrColVec_epi = epi_tensors_loop.tDrColVec_epi

        return self.EpilogueTensorsLoop(tDrColVec_epi=tDrColVec_epi)


class EVTRowBlockReductionStore(EpilogueVisitorTree):
    """
    Performs block-level row reductions of GEMM output within each CTA tile.

    Computes a partial row reduction during epilogue: one value per N column
    per M-tile. A second pass is needed to aggregate across M-tiles for the
    full row reduction.

    Input:
        - GEMM output: [M x N]

    Output:
        - mRowVec: [L, N, num_M_tiles]
          * num_M_tiles = (M + tile_M - 1) // tile_M

    Supported operations: sum, max, min, product (via reduction_op parameter)

    Implementation:
        On Hopper, M lanes are strided by 4 within a warp. Reduction along M
        uses an explicit butterfly with offsets 16, 8, 4. All warps in the
        wgmma epilogue see the same N columns but disjoint M ranges, so an
        inter-warp combine via smem is required, sized exactly to
        tile_N * (epi_num_threads / WARP_SIZE - 1) fp32 entries.
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mRowVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mRowVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sRowVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDrRowVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tDrRowVec_epi: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        reduction_op: reduction_utils.BlockReductionOp,
        tile_shape_mnk: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.arch = 90
        self.reduction_op = reduction_op
        self.tile_shape_mnk = tile_shape_mnk

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:
        if cutlass.const_expr(epi_args.mRowVec is not None):
            mRowVec = misc_utils.static_assert_is_Tensor(epi_args.mRowVec)
            mRowVec = layout_utils.assumed_align_stride(
                mRowVec,
                assumed_align=4,
            )
        else:
            mRowVec = None

        return self.EpilogueParams(mRowVec=mRowVec)

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
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

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

        return self.EpilogueTensors(tDrRowVec=tDrRowVec)

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

        if cutlass.const_expr(epi_params.mRowVec is not None):
            mRowVec = misc_utils.static_assert_is_Tensor(epi_params.mRowVec)
            sRowVec = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sRowVec)
            tDrRowVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrRowVec)
            row_vec_limit_n = min(shape_mnk[1] - n_idx * tile_N, tile_N)
            row_vec_limit_m = mRowVec.shape[2]

            # avoid duplicating operations on broadcast modes
            tDrRowVec_filtered = cute.filter_zeros(tDrRowVec)
            if cutlass.const_expr(self.arch != 100):
                for i in cutlass.range_constexpr(cute.size(tDrRowVec_filtered)):
                    offset = cute.arch.WARP_SIZE // 2
                    while offset >= HOPPER_WARP_REDUCTION_WIDTH:
                        tDrRowVec_filtered[i] = self.reduction_op.combine_fn_singleton(
                            tDrRowVec_filtered[i],
                            cute.arch.shuffle_sync_bfly(
                                tDrRowVec_filtered[i],
                                offset=offset,
                            ),
                        )
                        offset = offset // 2
            else:
                raise NotImplementedError

            mRowVec = mRowVec[batch_idx, None, m_idx]
            gRowVec = cute.local_tile(mRowVec, (tile_N,), (n_idx,))
            cRowVec = cute.make_identity_tensor((tile_M, tile_N))

            tDcRowVec = partition_for_epilogue(cRowVec)
            tDrRowVec_n = layout_utils.select_nonzero_stride_modes(tDrRowVec, tDrRowVec.layout)
            tDcRowVec_n = layout_utils.select_nonzero_stride_modes(tDcRowVec, tDrRowVec.layout)

            misc_utils.static_assert(cute.size(tiled_copy_r2s) == epi_num_threads)
            num_warps = epi_num_threads // cute.arch.WARP_SIZE
            warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
            warps_in_M = num_warps
            warp_m_idx = warp_idx

            # lanes 0..lanes_in_N-1 of EVERY warp (warp-local "row 0" of the 8x4 fragment).
            # Used for the smem write so each warp deposits its per-warp partial.
            is_lane_m_leader  = cute.arch.lane_idx() < HOPPER_WARP_REDUCTION_WIDTH
            # same as warp_m_idx == 0 AND is_lane_m_leader
            should_write_gmem = tDcRowVec_n[0][0] == 0

            # Inter-warp reduction through smem
            if cutlass.const_expr(warps_in_M > 1):
                if warp_m_idx > 0 and is_lane_m_leader:
                    for n in cutlass.range_constexpr(cute.size(tDcRowVec_n, mode=[0])):
                        col_idx = tDcRowVec_n[n][1]
                        sRowVec[col_idx, warp_m_idx - 1] = tDrRowVec_n[n]
                epi_barrier.arrive_and_wait()
                if should_write_gmem:
                    for n in cutlass.range_constexpr(cute.size(tDcRowVec_n, mode=[0])):
                        col_idx = tDcRowVec_n[n][1]
                        for warp_m in cutlass.range_constexpr(1, warps_in_M):
                            tDrRowVec_n[n] = self.reduction_op.combine_fn_singleton(
                                tDrRowVec_n[n],
                                sRowVec[col_idx, warp_m - 1],
                            )

            if m_idx < row_vec_limit_m and should_write_gmem:
                for n in cutlass.range_constexpr(cute.size(tDcRowVec_n, mode=[0])):
                    col_idx = tDcRowVec_n[n][1]
                    if col_idx < row_vec_limit_n:
                        gRowVec[col_idx] = tDrRowVec_n[n].to(dtype=misc_utils.get_dtype(gRowVec))

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:
        if cutlass.const_expr(epi_tensors.tDrRowVec is not None):
            tDrRowVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrRowVec)
            tDrRowVec_cur = cute.group_modes(tDrRowVec, 3, cute.rank(tDrRowVec))
            tDrRowVec_cur = tDrRowVec_cur[None, None, None, epi_coord]
        else:
            tDrRowVec_cur = None

        return (
            self.EpilogueTensorsLoop(tDrRowVec_epi=tDrRowVec_cur),
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
        if cutlass.const_expr(epi_tensors_loop.tDrRowVec_epi is not None):
            tDrRowVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrRowVec_epi)

            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tDrRowVec_epi)):
                    tDrRowVec_epi[i] = self.reduction_op.combine_fn_singleton(tDrRowVec_epi[i], tRS_rD[i])
            else:
                raise NotImplementedError
        else:
            tDrRowVec_epi = epi_tensors_loop.tDrRowVec_epi

        return self.EpilogueTensorsLoop(tDrRowVec_epi=tDrRowVec_epi)

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:
        if cutlass.const_expr(epi_params.mRowVec is not None):
            warps_in_M = epi_num_threads // cute.arch.WARP_SIZE
            row_vec_dtype = cute.Float32
            row_vec_smem_size = self.tile_shape_mnk[1] * (warps_in_M - 1)
        else:
            row_vec_dtype = cute.Float32
            row_vec_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            sRowVec: cute.struct.Align[cute.struct.MemRange[row_vec_dtype, row_vec_smem_size], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mRowVec is not None):
            warps_in_M = epi_num_threads // cute.arch.WARP_SIZE
            sRowVec_layout = layout_utils.make_ordered_layout(
                shape=(self.tile_shape_mnk[1], warps_in_M - 1),
                order="row",
            )
            sRowVec = storage.sRowVec.get_tensor(sRowVec_layout)
        else:
            sRowVec = None

        return self.EpilogueTensorsSMem(sRowVec=sRowVec)

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

        if cutlass.const_expr(epi_args.mRowVec is not None):
            warps_in_M = epi_num_threads // cute.arch.WARP_SIZE
            row_vec_dtype = cute.Float32
            row_vec_smem_size = self.tile_shape_mnk[1] * (warps_in_M - 1)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + (
                row_vec_smem_size *
                row_vec_dtype.width // 8
            )

        return (
            epi_smem_bytes_fixed,
            epi_smem_bytes_per_stage_cst,
            epi_smem_bytes_per_stage_pld,
        )


class EVTColBlockReductionStore2X(EVTColBlockReductionStore):
    """
    Block-level column reduction with 2x output resolution per CTA tile.

    Computes two independent column-reduction values per tile instead of one,
    doubling output granularity. Values are packed as i64 = (f32, f32) so the
    register layout matches the 1X variant; they are unpacked before computation
    and repacked after.

    Input:
        - GEMM output: [M x N]

    Output:
        - mColVec: [L, M, 2 * num_N_tiles]
    """

    EpilogueArguments = EVTColBlockReductionStore.EpilogueArguments
    EpilogueParams = EVTColBlockReductionStore.EpilogueParams
    EpilogueTensorsSMem = EVTColBlockReductionStore.EpilogueTensorsSMem
    EpilogueTensors = EVTColBlockReductionStore.EpilogueTensors
    EpilogueTensorsLoop = EVTColBlockReductionStore.EpilogueTensorsLoop
    EpiloguePipelines = EVTColBlockReductionStore.EpiloguePipelines

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
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

        if cutlass.const_expr(epi_params.mColVec is not None):
            rColVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(1, 0),
            )
            rColVec_view = creation_utils.allocate_tensor_from_layout(
                layout=rColVec_view_layout,
                dtype=cute.Int64,
                memspace="rmem",
                smem_allocator=None,
            )
            tDrColVec = partition_for_epilogue(rColVec_view)
            cute.filter_zeros(tDrColVec).fill(
                dtype_utils.f32x2_to_i64(
                    cute.Float32(0.),
                    cute.Float32(0.),
                )
            )
        else:
            tDrColVec = None

        return self.EpilogueTensors(tDrColVec=tDrColVec)

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

        if cutlass.const_expr(epi_params.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_params.mColVec)
            col_vec_dtype = misc_utils.get_dtype(mColVec)
            if cutlass.const_expr(col_vec_dtype in (cute.Float16, cute.BFloat16)):
                mColVec = cute.recast_tensor(mColVec, dtype=cute.Float32)
            else:
                raise NotImplementedError

            tDrColVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDrColVec)
            tDrColVec_cvt = creation_utils.allocate_tensor_like(
                tDrColVec,
                memspace="rmem",
                smem_allocator=None,
                dtype=misc_utils.get_dtype(mColVec),
            )
            misc_utils.static_assert(misc_utils.get_dtype(tDrColVec) is cute.Int64)

            col_vec_limit_m = min(shape_mnk[0] - m_idx * tile_M, tile_M)
            col_vec_limit_n = mColVec.shape[2]

            # avoid duplicating operations on broadcast modes
            tDrColVec_filtered = cute.filter_zeros(tDrColVec)
            tDrColVec_cvt_filtered = cute.filter_zeros(tDrColVec_cvt)
            misc_utils.static_assert(tDrColVec_filtered.shape == tDrColVec_cvt_filtered.shape)
            if cutlass.const_expr(self.arch != 100):
                # Hopper tensor core layout is such that each 4 consecutive
                # threads collectively hold the values of one row tile
                for i in cutlass.range_constexpr(cute.size(tDrColVec_filtered)):
                    tDrColVec_filtered_0, tDrColVec_filtered_1 = self.reduction_op.warp_reduction(
                        dtype_utils.i64_to_f32x2(tDrColVec_filtered[i]),
                        width=HOPPER_WARP_REDUCTION_WIDTH,
                    )
                    # we need to perform type conversion before packing
                    tDrColVec_filtered_cvt_0 = dtype_utils.convert(
                        tDrColVec_filtered_0,
                        dtype=col_vec_dtype,
                    )
                    tDrColVec_filtered_cvt_1 = dtype_utils.convert(
                        tDrColVec_filtered_1,
                        dtype=col_vec_dtype,
                    )
                    tDrColVec_cvt_filtered[i] = dtype_utils.pack2(
                        tDrColVec_filtered_cvt_0,
                        tDrColVec_filtered_cvt_1,
                        src_dtype=col_vec_dtype,
                        dst_dtype=misc_utils.get_dtype(mColVec),
                    )
            else:
                # Don't need warp_reduce since we load from tmem with one thread per row
                raise NotImplementedError

            mColVec = mColVec[batch_idx, None, n_idx]
            gColVec = cute.local_tile(mColVec, (tile_M,), (m_idx,))
            cColVec = cute.make_identity_tensor((tile_M, tile_N))

            tDcColVec = partition_for_epilogue(cColVec)
            tDrColVec_m = layout_utils.select_nonzero_stride_modes(tDrColVec_cvt, tDrColVec_cvt.layout)
            tDcColVec_m = layout_utils.select_nonzero_stride_modes(tDcColVec    , tDrColVec_cvt.layout)
            if n_idx < col_vec_limit_n and tDcColVec_m[0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(tDcColVec_m, mode=[0])):
                    row_idx = tDcColVec_m[m][0]
                    if row_idx < col_vec_limit_m:
                        # we cannot cast since the types are just containers
                        gColVec[row_idx] = tDrColVec_m[m]#.to(dtype=misc_utils.get_dtype(gColVec))

    @cute.jit
    def consumer_visit(
        self,
        tRS_rD: cute.Tensor,
        shape_mnk: cute.Shape,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> EpilogueTensorsLoop:
        if cutlass.const_expr(epi_tensors_loop.tDrColVec_epi is not None):
            tDrColVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrColVec_epi)
            misc_utils.static_assert(misc_utils.get_dtype(tRS_rD) is cute.Float32)
            misc_utils.static_assert(misc_utils.get_dtype(tDrColVec_epi) is cute.Int64)

            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tDrColVec_epi)):
                    # unpack the vector into two separate elements, perform two separate
                    # reductions and pack the two output elements into a vector. The second
                    # reduction is just a dummy example, we need a better one in practice.
                    tDrColVec_epi[i] = dtype_utils.f32x2_to_i64(
                        *self.reduction_op.combine_fn(
                            dtype_utils.i64_to_f32x2(tDrColVec_epi[i]),
                            (
                                tRS_rD[i],
                                -tRS_rD[i],
                            ),
                        )
                    )
            else:
                raise NotImplementedError
        else:
            tDrColVec_epi = epi_tensors_loop.tDrColVec_epi

        return self.EpilogueTensorsLoop(tDrColVec_epi=tDrColVec_epi)
