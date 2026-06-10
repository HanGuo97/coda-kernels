import math
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.utils import LayoutEnum

from coda.core.ops.misc_utils import static_assert


def make_smem_layouts(
    tile_shape_mnk: tuple[int, int, int],
    epi_tile: tuple[int, int],
    a_dtype: type[cute.Numeric],
    a_layout: LayoutEnum,
    b_dtype: type[cute.Numeric],
    b_layout: LayoutEnum,
    ab_stage: int,
    c_dtype: type[cute.Numeric],
    c_layout: LayoutEnum,
    epi_stage: int,
) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
    """Create shared memory layouts for A, B, and C tensors.

    :param tile_shape_mnk: CTA tile shape (M,N,K)
    :type tile_shape_mnk: Tuple[int, int, int]
    :param epi_tile: Epilogue tile shape
    :type epi_tile: Tuple[int, int]
    :param a_dtype: Data type for matrix A
    :type a_dtype: type[cute.Numeric]
    :param a_layout: Layout enum for matrix A
    :type a_layout: LayoutEnum
    :param b_dtype: Data type for matrix B
    :type b_dtype: type[cute.Numeric]
    :param b_layout: Layout enum for matrix B
    :type b_layout: LayoutEnum
    :param ab_stage: Number of stages for A/B tensors
    :type ab_stage: int
    :param c_dtype: Data type for output matrix C
    :type c_dtype: type[cute.Numeric]
    :param c_layout: Layout enum for the output matrix C
    :type c_layout: LayoutEnum
    :param epi_stage: Number of epilogue stages
    :type epi_stage: int

    :return: Tuple of shared memory layouts for A, B, and C
    :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
    """
    a_smem_layout_staged = sm90_utils.make_smem_layout_a(
        a_layout,
        tile_shape_mnk,
        a_dtype,
        ab_stage,
    )

    b_smem_layout_staged = sm90_utils.make_smem_layout_b(
        b_layout,
        tile_shape_mnk,
        b_dtype,
        ab_stage,
    )

    epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
        c_dtype,
        c_layout,
        epi_tile,
        epi_stage,
    )

    return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged


def compute_stages(
    tile_shape_mnk: tuple[int, int, int],
    epi_tile: tuple[int, int],
    a_dtype: type[cute.Numeric],
    b_dtype: type[cute.Numeric],
    d_dtype: type[cute.Numeric],
    smem_capacity: int,
    occupancy: int,
    epi_smem_bytes_per_stage: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Computes the number of stages for A/B/C operands based on heuristics.

    :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
    :type tile_shape_mnk: tuple[int, int, int]
    :param a_dtype: Data type of operand A.
    :type a_dtype: type[cute.Numeric]
    :param b_dtype: Data type of operand B.
    :type b_dtype: type[cute.Numeric]
    :param smem_capacity: Total available shared memory capacity in bytes.
    :type smem_capacity: int
    :param occupancy: Target number of CTAs per SM (occupancy).
    :type occupancy: int

    :return: A tuple containing the computed number of stages for:
                (A/B operand stages, epilogue stages)
    :rtype: tuple[int, int]
    """

    epi_stage_cst = 4 if cutlass.const_expr(epi_tile[1] <= 16) else 2
    epi_stage_pld = 4 if cutlass.const_expr(epi_tile[1] <= 16) else 2

    (
        epi_smem_bytes_fixed,
        epi_smem_bytes_per_stage_cst,
        epi_smem_bytes_per_stage_pld,
    ) = epi_smem_bytes_per_stage
    epi_smem_bytes_per_stage_cst = (
        epi_smem_bytes_per_stage_cst +
        cute.size(epi_tile) * d_dtype.width // 8
    )
    epi_bytes = (
        epi_smem_bytes_fixed +
        epi_smem_bytes_per_stage_cst * epi_stage_cst +
        epi_smem_bytes_per_stage_pld * epi_stage_pld
    )

    a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
    b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
    ab_bytes_per_stage = (
        cute.size(a_shape) * a_dtype.width // 8 +
        cute.size(b_shape) * b_dtype.width // 8
    )
    # This is used to account for bytes due to (mis)alignment and others
    misc_bytes = 1
    mbar_helpers_bytes = 1024

    remaining_bytes = (
        smem_capacity // occupancy -
        misc_bytes -
        mbar_helpers_bytes -
        epi_bytes
    )
    ab_stage = remaining_bytes // ab_bytes_per_stage

    # Refine epilogue stages:
    # Calculate remaining smem after allocating for A/B stages and reserved bytes
    # Add remaining unused smem to epilogue
    if cutlass.const_expr(epi_smem_bytes_per_stage_cst > 0):
        epi_remaining_bytes = (remaining_bytes - ab_bytes_per_stage * ab_stage)
        epi_stage_cst = epi_stage_cst + epi_remaining_bytes // epi_smem_bytes_per_stage_cst

    return ab_stage, epi_stage_cst, epi_stage_pld


def compute_grid(
    c: cute.Tensor,
    tile_shape_mnk: tuple[int, int, int],
    cluster_shape_mn: tuple[int, int],
) -> tuple[int, int, int]:
    """Compute grid shape for the output tensor C.

    :param c: The output tensor C
    :type c: cute.Tensor
    :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
    :type tile_shape_mnk: tuple[int, int, int]
    :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
    :type cluster_shape_mn: tuple[int, int]

    :return: Grid shape for kernel launch.
    :rtype: tuple[int, int, int]
    """

    c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
    gc = cute.zipped_divide(c, tiler=c_shape)
    cluster_shape_mnl = (*cluster_shape_mn, 1)
    clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl)
    grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))
    return grid


def is_valid_dtypes(
    a_dtype: type[cute.Numeric],
    b_dtype: type[cute.Numeric],
    acc_dtype: type[cute.Numeric],
    c_dtype: type[cute.Numeric],
    a_major: str,
    b_major: str,
) -> bool:
    """
    Check if the dtypes are valid

    :param a_dtype: The data type of tensor A
    :type a_dtype: type[cute.Numeric]
    :param b_dtype: The data type of tensor B
    :type b_dtype: type[cute.Numeric]
    :param acc_dtype: The data type of the accumulator
    :type acc_dtype: type[cute.Numeric]
    :param c_dtype: The data type of the output tensor
    :type c_dtype: type[cute.Numeric]
    :param a_major: major mode of tensor A
    :type a_major: str
    :param b_major: major mode of tensor B
    :type b_major: str

    :return: True if the dtypes are valid, False otherwise
    :rtype: bool
    """
    is_valid = True

    valid_ab_dtypes = {
        cutlass.Float16,
        cutlass.BFloat16,
    }
    if cutlass.const_expr(a_dtype not in valid_ab_dtypes):
        is_valid = False
    if cutlass.const_expr(b_dtype not in valid_ab_dtypes):
        is_valid = False

    # make sure a_dtype == b_dtype for Float16
    if cutlass.const_expr(a_dtype.width == 16 and a_dtype != b_dtype):
        is_valid = False
    if cutlass.const_expr(a_dtype.width != b_dtype.width):
        is_valid = False
    if cutlass.const_expr(not a_dtype.is_same_kind(b_dtype)):
        is_valid = False

    # for 8-bit types, this implementation only supports k-major layout
    if cutlass.const_expr(
        (a_dtype.width == 8 and a_major != "k") or
        (b_dtype.width == 8 and b_major != "k")
    ):
        is_valid = False

    # Define compatibility mapping between accumulator type and AB type
    acc_ab_compatibility = {
        cutlass.Float32: {
            cutlass.Float16,
            cutlass.BFloat16,
        },
    }
    # Check compatibility between accumulator type and A type
    if cutlass.const_expr(a_dtype not in acc_ab_compatibility[acc_dtype]):
        is_valid = False

    # Define compatibility mapping between accumulator type and C type
    acc_c_compatibility = {
        cutlass.Float32: {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
        },
    }
    # Check compatibility between accumulator type and C type
    if cutlass.const_expr(c_dtype not in acc_c_compatibility[acc_dtype]):
        is_valid = False

    return is_valid


def is_valid_tensor_alignment(
    m: int,
    n: int,
    k: int,
    l: int,
    ab_dtype: type[cute.Numeric],
    c_dtype: type[cute.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
) -> bool:
    """
    Check if the tensor alignment is valid

    :param m: The number of rows in the A tensor
    :type m: int
    :param n: The number of columns in the B tensor
    :type n: int
    :param k: The number of columns in the A tensor
    :type k: int
    :param l: The number of columns in the C tensor
    :type l: int
    :param ab_dtype: The data type of the A and B operands
    :type ab_dtype: type[cute.Numeric]
    :param c_dtype: The data type of the output tensor
    :type c_dtype: type[cute.Numeric]
    :param a_major: The major axis of the A tensor
    :type a_major: str
    :param b_major: The major axis of the B tensor
    :type b_major: str
    :param c_major: The major axis of the C tensor
    :type c_major: str

    :return: True if the problem shape is valid, False otherwise
    :rtype: bool
    """
    is_valid = True

    def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
        major_mode_idx = 0 if is_mode0_major else 1
        num_major_elements = tensor_shape[major_mode_idx]
        num_contiguous_elements = 16 * 8 // dtype.width
        return num_major_elements % num_contiguous_elements == 0

    if cutlass.const_expr(
        not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
        or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
        or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
    ):
        is_valid = False
    return is_valid


def get_major(
    tensor: cute.Tensor,
    dims: tuple[str, str, str],
) -> str:
    stride = tensor.stride
    static_assert(len(tensor.shape) == 3)
    static_assert(len(stride) == 3)
    static_assert(len(dims) == 3)
    if cutlass.const_expr(stride[0] == 1):
        major = dims[0]
    elif cutlass.const_expr(stride[1] == 1):
        major = dims[1]
    else:
        raise ValueError
    return major


# https://github.com/Dao-AILab/quack/blob/main/quack/utils.py
def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (
            acc_layout_col_major.shape[0][1],
            acc_layout_col_major.shape[1],
        ),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (
            acc_layout_col_major.stride[0][1],
            acc_layout_col_major.stride[1],
        ),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    acc_layout_mn = cute.make_layout(
        shape=shape,
        stride=stride,
    )
    return cute.composition(acc_layout, acc_layout_mn)


# https://github.com/Dao-AILab/quack/blob/main/quack/utils.py
def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    layout = convert_layout_acc_mn(acc.layout)
    return cute.make_tensor(
        iterator=acc.iterator,
        layout=layout,
    )


def get_smem_load_op(
    layout_c: LayoutEnum,
    elem_ty_c: type[cute.Numeric],
) -> cute.CopyAtom:

    def validate_type(ty, ty_name):
        if not isinstance(ty, cutlass.cutlass_dsl.NumericMeta):
            raise TypeError(f"{ty_name} must be a Numeric, but got {ty}")

    validate_type(elem_ty_c, "elem_ty_d")

    is_m_major = layout_c.is_m_major_c()

    if elem_ty_c.width == 16:
        return cute.make_copy_atom(
            op=cute.nvgpu.warp.LdMatrix8x8x16bOp(is_m_major, 4),
            copy_internal_type=elem_ty_c,
        )
    else:
        return cute.make_copy_atom(
            op=cute.nvgpu.CopyUniversalOp(),
            copy_internal_type=elem_ty_c,
        )


sm90_get_smem_load_op = get_smem_load_op


def check_tile_sizes(
    tile_M: int,
    tile_N: int,
    pingpong: bool,
) -> None:
    # check the cta tile shape
    if not pingpong:

        if tile_M not in [64, 128, 192, 256, 320]:
            raise ValueError("CTA tile shape M must be 64/128/192/256/320")

        # special case
        if tile_M in [192, 320]:
            if tile_M == 192:
                tile_N_max = 256
            else:
                tile_N_max = 160

            if not (
                tile_N % 32 == 0 and
                tile_N <= tile_N_max
            ):
                raise ValueError(
                    f"If tile_m == {tile_M}, CTA tile shape N "
                    f"must be divisible by 32 and <= {tile_N_max}")

        else:
            if not (
                (tile_N % 16 == 0 and tile_N <= 256) or
                (tile_N % 32 == 0 and tile_N <= 512)
            ):
                raise ValueError(
                    "CTA tile shape N must be divisible by 16 "
                    "and <= 256, or divisible by 32 and <= 512")

    else:
        if tile_M not in [64, 128, 192]:
            raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")

        if tile_M == 64:
            tile_N_max = 256
        elif tile_M == 128:
            tile_N_max = 208
        else:
            tile_N_max = 128

        if not (
            tile_N % 16 == 0 and
            tile_N <= tile_N_max
        ):
            raise ValueError(f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}")


def get_atom_layout(
    tile_M: int,
    tile_N: int,
    pingpong: bool,
) -> tuple[int, int]:
    if not pingpong:

        # tile_M / 64 is not even so we have to split along N
        if tile_M == 320:
            atom_layout_m = 1
            atom_layout_n = 2

        elif tile_M == 192:
            if tile_N <= 128:
                atom_layout_m = 3
                atom_layout_n = 1
            else:
                atom_layout_m = 1
                atom_layout_n = 2

        else:
            if tile_M < 256:
                atom_layout_m = tile_M // 64
            else:
                atom_layout_m = 2

            atom_layout_n = 1

        assert atom_layout_m in (1, 2, 3)
        assert atom_layout_n in (1, 2)

    else:
        atom_layout_m = 1
        atom_layout_n = 1

    return atom_layout_m, atom_layout_n


def get_mma_warp_groups(
    atom_layout_mnk: tuple[int, int, int],
    pingpong: bool,
) -> int:
    mma_warp_groups = math.prod(atom_layout_mnk)
    if pingpong:
        mma_warp_groups = mma_warp_groups * 2
        assert mma_warp_groups == 2
    assert mma_warp_groups in [1, 2, 3]
    return mma_warp_groups


def get_num_epi_warps(
    mma_warp_groups: int,
    pingpong: bool,
) -> int:
    if not pingpong:
        num_epi_warp_groups = mma_warp_groups
    else:
        num_epi_warp_groups = 1

    return num_epi_warp_groups * 4


def get_register_allocations(
    tile_shape_mnk: tuple[int, int, int],
    atom_layout_mnk: tuple[int, int, int],
    mma_warp_groups: int,
    num_threads_per_warp_group: int,
) -> tuple[int, int]:
    num_registers = math.prod(tile_shape_mnk[:2])
    num_atoms = math.prod(atom_layout_mnk)
    num_threads = num_atoms * num_threads_per_warp_group
    num_registers_per_thread = num_registers // num_threads

    if mma_warp_groups == 3:
        num_registers_producer = 32
        num_registers_consumer = 160

    else:
        heavy_register_pressure = num_registers_per_thread >= 208
        if not heavy_register_pressure:
            num_registers_producer = 40
            num_registers_consumer = 232
        else:
            num_registers_producer = 24
            num_registers_consumer = 240

    return num_registers_producer, num_registers_consumer


def sm90_compute_tile_shape_or_override(
    tile_shape_mnk: tuple[int, int, int],
    atom_layout_mnk: tuple[int, int, int],
    element_type: type[cute.Numeric] | None,
    epi_tile_override: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if epi_tile_override is not None:
        return epi_tile_override

    if tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
        tile_m = math.gcd(128, cute.size(tile_shape_mnk, mode=[0]))
        tile_n = math.gcd(32, cute.size(tile_shape_mnk, mode=[1]))

    elif tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
        tile_m = math.gcd(192, cute.size(tile_shape_mnk, mode=[0]))
        tile_n = math.gcd(32, cute.size(tile_shape_mnk, mode=[1]))

    else:
        # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
        # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
        # M dimension first, then move to the N dimension. But the accumulator in registers
        # iterate along the N dimension first, then move to the M dimension.
        # We could change the epilogue to accommodate this,
        # but it's easier to just set epi_tile_m = 64.
        n_perf = 64 if element_type is not None and element_type.width == 8 else 32
        tile_m = math.gcd(64, cute.size(tile_shape_mnk, mode=[0]))
        tile_n = math.gcd(n_perf, cute.size(tile_shape_mnk, mode=[1]))

    return (tile_m, tile_n)


def make_mainloop_pipeline(
    num_stages: int,
    num_tma_load_bytes: int,
    num_mcast_ctas_a: int,
    num_mcast_ctas_b: int,
    tiled_mma: cute.TiledMma,
    cluster_layout_vmnk: cute.Layout,
    mainloop_pipeline_mbar_ptr: cute.Pointer,
) -> cutlass.pipeline.PipelineTmaAsync:
    # Threads/warps participating in this pipeline
    producer_size = 1
    pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread,
        size=producer_size,
    )
    # Each warp will contribute to the arrive count with the number of mcast size
    mcast_size = num_mcast_ctas_a + num_mcast_ctas_b - 1
    consumer_arrive_size = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
    pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread,
        size=consumer_arrive_size,
    )
    return cutlass.pipeline.PipelineTmaAsync.create(
        barrier_storage=mainloop_pipeline_mbar_ptr,
        num_stages=num_stages,
        producer_group=pipeline_producer_group,
        consumer_group=pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )


def make_scheduler_pipeline(
    pingpong: bool,
    num_stages: int,
    dma_warps: int,
    mma_warp_groups: int,
    cluster_layout_mnk: cute.Layout,
    scheduler_pipeline_mbar_ptr: cute.Pointer,
) -> cutlass.pipeline.PipelineAsync:
    # Threads/warps participating in this pipeline
    scheduler_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread,
    )
    cluster_size = cute.size(cluster_layout_mnk)
    # Each warp will contribute 1 to the arrive count. If pingpong and varlen_k,
    # then all 8 mma warps will participate in the scheduler barrier at each
    # round. If pingpong and not varlen_k, then only 4 mma warp will participate.
    if cutlass.const_expr(not pingpong):
        num_warps = dma_warps + 4 * mma_warp_groups
    else:
        num_warps = dma_warps + 4
    consumer_arrive_size = num_warps * cluster_size
    scheduler_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread,
        size=consumer_arrive_size,
    )
    return cutlass.pipeline.PipelineAsync.create(
        barrier_storage=scheduler_pipeline_mbar_ptr,
        num_stages=num_stages,
        producer_group=scheduler_pipeline_producer_group,
        consumer_group=scheduler_pipeline_consumer_group,
        # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
        consumer_mask=None if cutlass.const_expr(cluster_size == 1) else 0,
        defer_sync=True,
    )
