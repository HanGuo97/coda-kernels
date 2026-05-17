import cutlass
import cutlass.cute as cute
from typing import NamedTuple
from dataclasses import dataclass

from hilt.dtype_utils import get_dtype
from rapier.ops import misc_utils
from rapier.ops import struct_utils
from rapier.ops import layout_utils
from rapier.ops import memory_utils
from rapier.ops import epilogue_utils
from rapier.epilogue.base import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)


class EVTRowOrColBias(EpilogueVisitorTree):
    """
    Broadcasts and adds row and/or column bias vectors to GEMM output.

    Inputs:
        - GEMM output: [M x N]
        - Row bias (optional): [L, N] (L is batch dim)
        - Column bias (optional): [L, M]

    Output:
        - D = (A @ B) + row_bias + col_bias: [M x N]
        - row_bias broadcast along M: [1 x N] -> [M x N]
        - col_bias broadcast along N: [M x 1] -> [M x N]

    Implementation:
        Loads bias vectors asynchronously via cp.async to shared memory, then broadcasts
        and adds in registers during epilogue.

    Example:
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTRowOrColBias(acc_dtype=acc_dtype, tile_shape_mnk=tile_shape_mnk)
        >>> epi_args = EVTRowOrColBias.EpilogueArguments(
        ...     mRowVec=bias_row_tensor,  # [L, N]
        ...     mColVec=bias_col_tensor,  # [L, M]
        ... )
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mRowVec: cute.Tensor | None
        mColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mRowVec: cute.Tensor | None
        mColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sRowVec: cute.Tensor | None
        sColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDsRowVec: cute.Tensor | None
        tDsColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tDrRowVec_epi: cute.Tensor | None
        tDrColVec_epi: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        acc_dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.acc_dtype = acc_dtype
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

        if cutlass.const_expr(epi_args.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_args.mColVec)
            mColVec = layout_utils.assumed_align_stride(
                mColVec,
                assumed_align=4,
            )
        else:
            mColVec = None

        return self.EpilogueParams(
            mRowVec=mRowVec,
            mColVec=mColVec,
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
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            sRowVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(0, 1),
            )
            sRowVec_view = cute.make_tensor(
                iterator=sRowVec.iterator,
                layout=sRowVec_view_layout,
            )
            tDsRowVec = partition_for_epilogue(sRowVec_view)
        else:
            tDsRowVec = None

        if cutlass.const_expr(epi_params.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_params.mColVec)
            sColVec = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sColVec)
            mColVec = mColVec[batch_idx, None]
            gColVec = cute.local_tile(mColVec, (tile_M,), (m_idx,))
            cColVec = cute.make_identity_tensor(tile_M)
            limit_m = min(mColVec.shape[0] - m_idx * tile_M, tile_M)
            memory_utils.g2s_copy_1d(
                src=gColVec,
                dst=sColVec,
                crd=cColVec,
                shape=(limit_m,),
                num_threads=epi_num_threads,
                thread_index=tidx,
            )
            sColVec_view_layout = cute.make_layout(
                shape=(tile_M, tile_N),
                stride=(1, 0),
            )
            sColVec_view = cute.make_tensor(
                iterator=sColVec.iterator,
                layout=sColVec_view_layout,
            )
            tDsColVec = partition_for_epilogue(sColVec_view)
        else:
            tDsColVec = None

        if cutlass.const_expr(
            (tDsRowVec is not None) or
            (tDsColVec is not None)
        ):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epi_barrier.arrive_and_wait()

        return self.EpilogueTensors(
            tDsRowVec=tDsRowVec,
            tDsColVec=tDsColVec,
        )

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

        if cutlass.const_expr(epi_tensors.tDsColVec is not None):
            tDsColVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDsColVec)
            tDsColVec_cur = cute.group_modes(tDsColVec, 3, cute.rank(tDsColVec))
            tDsColVec_cur = tDsColVec_cur[None, None, None, epi_coord]
            tDrColVec_cvt = memory_utils.s2r_copy_1d(tDsColVec_cur, dtype=self.acc_dtype)
        else:
            tDrColVec_cvt = None

        return (
            self.EpilogueTensorsLoop(
                tDrRowVec_epi=tDrRowVec_cvt,
                tDrColVec_epi=tDrColVec_cvt,
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

        if cutlass.const_expr(epi_params.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_params.mColVec)
            col_vec_dtype = get_dtype(mColVec)
            col_vec_smem_size = epilogue_utils.get_smem_size_vector(
                mTensor=mColVec,
                epi_tile=self.tile_shape_mnk[0],
                epi_num_threads=epi_num_threads,
            )
        else:
            col_vec_dtype = cute.Float32
            col_vec_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            sRowVec: cute.struct.Align[cute.struct.MemRange[row_vec_dtype, row_vec_smem_size], 16]
            sColVec: cute.struct.Align[cute.struct.MemRange[col_vec_dtype, col_vec_smem_size], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mRowVec is not None):
            sRowVec_layout = cute.make_layout(self.tile_shape_mnk[1])
            sRowVec = storage.sRowVec.get_tensor(sRowVec_layout)
        else:
            sRowVec = None

        if cutlass.const_expr(epi_params.mColVec is not None):
            sColVec_layout = cute.make_layout(self.tile_shape_mnk[0])
            sColVec = storage.sColVec.get_tensor(sColVec_layout)
        else:
            sColVec = None

        return self.EpilogueTensorsSMem(
            sRowVec=sRowVec,
            sColVec=sColVec,
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

        # we load the entire row/col vectors upfront, hence the smem
        # storage is fixed and independent of stages
        if cutlass.const_expr(epi_args.mRowVec is not None):
            mRowVec = misc_utils.static_assert_is_Tensor(epi_args.mRowVec)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + (
                epilogue_utils.get_epi_smem_bytes_per_stage_fixed_vector(
                    mTensor=mRowVec,
                    epi_tile=self.tile_shape_mnk[1],
                    epi_num_threads=epi_num_threads,
                )
            )

        if cutlass.const_expr(epi_args.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_args.mColVec)
            epi_smem_bytes_fixed = epi_smem_bytes_fixed + (
                epilogue_utils.get_epi_smem_bytes_per_stage_fixed_vector(
                    mTensor=mColVec,
                    epi_tile=self.tile_shape_mnk[0],
                    epi_num_threads=epi_num_threads,
                )
            )

        return (
            epi_smem_bytes_fixed,
            epi_smem_bytes_per_stage_cst,
            epi_smem_bytes_per_stage_pld,
        )
