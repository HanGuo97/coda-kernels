import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils

from hilt.dtype_utils import get_dtype
from coda.core.ops import misc_utils
from coda.core.ops import gemm_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils


def prepare_copy_r2s_sm90(
    tiled_copy_r2s: cute.TiledCopy,
    tidx: cute.Int32,
    dst: cute.Tensor,
    epi_layout: cutlass.utils.LayoutEnum,
    epi_dtype: type[cute.Numeric],
    acc_dtype: type[cute.Numeric],
) -> tuple[cute.TiledCopy, cute.ThrCopy, cute.Tensor]:
    copy_atom_postact_r2s = sm90_utils.sm90_get_smem_store_op(
        layout_d=epi_layout,
        elem_ty_d=epi_dtype,
        elem_ty_acc=acc_dtype,
    )
    tiled_copy_postact_r2s = cute.make_tiled_copy_S(
        atom=copy_atom_postact_r2s,
        tiled_copy=tiled_copy_r2s,
    )
    thr_copy_postact_r2s = tiled_copy_postact_r2s.get_slice(tidx)
    dst_thread = thr_copy_postact_r2s.partition_D(dst)
    return (
        tiled_copy_postact_r2s,
        thr_copy_postact_r2s,
        dst_thread,
    )


def prepare_copy_s2r_sm90(
    tiled_mma: cute.TiledMma,
    tidx: cute.Int32,
    src: cute.Tensor,
    dst_layout: cute.Layout,
    epi_dtype: type[cute.Numeric],
    container_dtype: type[cute.Numeric],
    epi_gmem_layout: cutlass.utils.LayoutEnum,
    epi_num_matrices: int,
) -> tuple[cute.TiledCopy, cute.ThrCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(
            epi_gmem_layout.is_m_major_c(),
            num_matrices=epi_num_matrices,
        ),
        epi_dtype,
    )
    tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
        atom=copy_atom,
        mma=tiled_mma,
    )
    copy_atom_s2r = gemm_utils.sm90_get_smem_load_op(
        layout_c=epi_gmem_layout,
        elem_ty_c=container_dtype,
    )
    tiled_copy_s2r = cute.make_tiled_copy_S(
        atom=copy_atom_s2r,
        tiled_copy=tiled_copy_C_atom,
    )
    thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
    src_thread = thr_copy_s2r.partition_S(src)
    dst_thread = creation_utils.allocate_tensor_from_layout(
        layout=dst_layout,
        dtype=container_dtype,
        memspace="rmem",
        smem_allocator=None,
    )
    dst_thread_view = thr_copy_s2r.retile(dst_thread)
    return (
        tiled_copy_s2r,
        thr_copy_s2r,
        src_thread,
        dst_thread,
        dst_thread_view,
    )


def prepare_tma(
    tma_op: str,
    epi_tile: cute.Tile,
    epi_stage: int,
    epi_tensor: cute.Tensor,
) -> tuple[cutlass.utils.LayoutEnum, cute.Layout, cute.CopyAtom, cute.Tensor]:
    epi_dtype = get_dtype(epi_tensor)
    epi_coord = cute.make_identity_layout(epi_tensor.shape)
    epi_gmem_layout = cutlass.utils.LayoutEnum.from_tensor(epi_tensor)
    epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
        epi_dtype=epi_dtype,
        epi_layout=epi_gmem_layout,
        epi_tile=epi_tile,
        epi_stage=epi_stage,
    )

    epi_smem_tile = cute.composition(epi_coord, epi_tile)
    epi_tma_atom, epi_tma_tensor = memory_utils.make_tma_atoms_and_tensors(
        op=tma_op,
        tensor=epi_tensor,
        smem_layout_staged=epi_smem_layout_staged,
        smem_tile=epi_smem_tile,
    )
    return (
        epi_gmem_layout,
        epi_smem_layout_staged,
        epi_tma_atom,
        epi_tma_tensor,
    )


def prepare_epi_load_pipeline(
    epi_load_stage: int,
    epi_dtype: type[cute.Numeric],
    epi_num_warps: int,
    epi_smem_layout: cute.Layout,
    epi_load_pipeline_mbar_ptr: cute.Pointer,
) -> tuple[cutlass.pipeline.PipelineTmaAsync, cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
    # Threads/warps participating in this pipeline
    epi_load_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
        agent=cutlass.pipeline.Agent.Thread,
    )
    # Each warp will contribute 1 to the arrive count
    epi_load_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
        agent=cutlass.pipeline.Agent.Thread,
        size=epi_num_warps,
    )
    tma_copy_bytes = cute.size_in_bytes(
        dtype=epi_dtype,
        layout=epi_smem_layout,
    )
    epi_load_pipeline = cutlass.pipeline.PipelineTmaAsync.create(
        barrier_storage=epi_load_pipeline_mbar_ptr,
        num_stages=epi_load_stage,
        producer_group=epi_load_pipeline_producer_group,
        consumer_group=epi_load_pipeline_consumer_group,
        tx_count=tma_copy_bytes,
        defer_sync=True,
    )
    epi_load_consumer_state = cutlass.pipeline.make_pipeline_state(
        type=cutlass.pipeline.PipelineUserType.Consumer,
        stages=epi_load_stage,
    )
    epi_load_producer_state = cutlass.pipeline.make_pipeline_state(
        type=cutlass.pipeline.PipelineUserType.Producer,
        stages=epi_load_stage,
    )
    return (
        epi_load_pipeline,
        epi_load_consumer_state,
        epi_load_producer_state,
    )


def get_minimum_vector_size(dtype: type[cute.Numeric]) -> int:
    if cutlass.const_expr(dtype.width == 16):
        return 2
    elif cutlass.const_expr(dtype.width >= 32):
        # it seems like setting this to 1 could occasionally lead to OOB
        return 2
    else:
        raise NotImplementedError


def get_smem_size_vector(
    mTensor: cute.Tensor,
    epi_tile: cute.Tile | int,
    epi_num_threads: int,
) -> int:
    mTensor = misc_utils.static_assert_is_Tensor(mTensor)
    epi_dtype = get_dtype(mTensor)
    # we at least need certain number of smem size to avoid OOB smem access
    # https://github.com/NVIDIA/cutlass/issues/2980
    # we also need to make sure consistency between
    # `get_smem_struct` and `get_smem_bytes_per_stage`
    vec_min_smem_size = (
        epi_num_threads *
        get_minimum_vector_size(epi_dtype)
    )
    vec_smem_size = cutlass.max(
        epi_tile,
        vec_min_smem_size,
    )
    return vec_smem_size


def get_epi_smem_bytes_per_stage_fixed_vector(
    mTensor: cute.Tensor,
    epi_tile: cute.Tile | int,
    epi_num_threads: int,
) -> int:
    mTensor = misc_utils.static_assert_is_Tensor(mTensor)
    epi_dtype = get_dtype(mTensor)
    vec_smem_size = get_smem_size_vector(
        mTensor=mTensor,
        epi_tile=epi_tile,
        epi_num_threads=epi_num_threads,
    )
    epi_smem_bytes_fixed = (
        vec_smem_size *
        epi_dtype.width //
        8
    )
    return epi_smem_bytes_fixed


def get_epi_smem_bytes_per_stage_matrix(
    mTensor: cute.Tensor,
    epi_tile: cute.Tile,
) -> int:
    mTensor = misc_utils.static_assert_is_Tensor(mTensor)
    epi_dtype = get_dtype(mTensor)
    epi_smem_bytes_per_stage = (
        cute.size(epi_tile) *
        epi_dtype.width //
        8
    )
    return epi_smem_bytes_per_stage
