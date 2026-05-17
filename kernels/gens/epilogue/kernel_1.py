import torch
import cutlass
import cutlass.cute as cute
from dataclasses import dataclass
from typing import Callable, NamedTuple

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


class EVTRMSNormScale(EpilogueVisitorTree):
    """
    Loads pre-computed reciprocal standard deviation R and multiplies the
    accumulator by R (per-row broadcast).

    Inputs:
        - GEMM output: [M x N]
        - R column vector (mColVec): [L, M] — reciprocal standard deviation in fp32

    Outputs:
        - D = (A @ B) * R: [M x N] (written to C by the default epilogue store)
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        mColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sColVec: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDsColVec: cute.Tensor | None

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
        if cutlass.const_expr(epi_args.mColVec is not None):
            mColVec = misc_utils.static_assert_is_Tensor(epi_args.mColVec)
            mColVec = layout_utils.assumed_align_stride(
                mColVec,
                assumed_align=4,
            )
        else:
            mColVec = None

        return self.EpilogueParams(
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

        def partition_for_epilogue(tensor: cute.Tensor) -> cute.Tensor:
            tensor_epi = cute.flat_divide(tensor, epi_tile)
            return thr_copy_r2s.partition_S(tensor_epi)

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

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epi_barrier.arrive_and_wait()
        else:
            tDsColVec = None

        return self.EpilogueTensors(
            tDsColVec=tDsColVec,
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

        if cutlass.const_expr(epi_tensors.tDsColVec is not None):
            tDsColVec = misc_utils.static_assert_is_Tensor(epi_tensors.tDsColVec)
            tDsColVec_cur = cute.group_modes(tDsColVec, 3, cute.rank(tDsColVec))
            tDsColVec_cur = tDsColVec_cur[None, None, None, epi_coord]
            # Load R from smem to registers, converting to accumulator dtype (fp32)
            tDrColVec_cvt = memory_utils.s2r_copy_1d(tDsColVec_cur, dtype=self.acc_dtype)
        else:
            tDrColVec_cvt = None

        return (
            self.EpilogueTensorsLoop(
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

        if cutlass.const_expr(epi_tensors_loop.tDrColVec_epi is not None):
            tDrColVec_epi = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDrColVec_epi)
            # Multiply accumulator by R (per-row scaling)
            for i in cutlass.range_constexpr(cute.size(tDrColVec_epi)):
                tRS_rD[i] = tRS_rD[i] * tDrColVec_epi[i]
        else:
            tDrColVec_epi = epi_tensors_loop.tDrColVec_epi

        return self.EpilogueTensorsLoop(
            tDrColVec_epi=tDrColVec_epi,
        )

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:
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
            sColVec: cute.struct.Align[cute.struct.MemRange[col_vec_dtype, col_vec_smem_size], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:
        if cutlass.const_expr(epi_params.mColVec is not None):
            sColVec_layout = cute.make_layout(self.tile_shape_mnk[0])
            sColVec = storage.sColVec.get_tensor(sColVec_layout)
        else:
            sColVec = None

        return self.EpilogueTensorsSMem(
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


def prepare_epilogue(
    shape_mnkl: tuple[int, int, int, int],
    tile_shape_mn: tuple[int, int],
    R: torch.Tensor,
) -> tuple[
    Callable[..., EpilogueVisitorTree],
    EpilogueVisitorTree.EpilogueArguments,
    dict,
    tuple,
]:
    """Prepare epilogue for GEMM with RMSNorm scaling.

    Computes:
        D = acc * R  (per-row broadcast multiplication)

    Args:
        shape_mnkl: Problem shape (M, N, K, L).
        tile_shape_mn: CTA tile shape (tile_M, tile_N).
        R: Pre-computed reciprocal std dev of shape (M,) in fp32.

    Returns:
        Tuple of (epi_cls, epi_args, epi_outs, epi_keys).
    """
    M, N, K, L = shape_mnkl

    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTRMSNormScale(
        acc_dtype=acc_dtype,
        tile_shape_mnk=tile_shape_mnk,
    )

    epi_args = EVTRMSNormScale.EpilogueArguments(
        mColVec=R,
    )

    epi_keys = (
        R.dtype,
        EVTRMSNormScale,
    )

    epi_outs = {}

    return epi_cls, epi_args, epi_outs, epi_keys
