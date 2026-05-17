# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py

import enum
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import if_generate
from typing import Callable

from quack.cute_dsl_utils import ParamsBase
from quack.tile_scheduler import (
    TileSchedulerOptions,
    TileSchedulerArguments,
    TileScheduler,
    PersistenceMode,
)

from rapier.ops import misc_utils
from rapier.ops import gemm_utils
from rapier.ops import layout_utils
from rapier.ops import memory_utils
from rapier.ops import pipeline_utils
from rapier.ops.launch_utils import launch_check
from rapier.epilogue import EpilogueVisitorTree


"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    # For mainloop load warps to signal that the epilogue load warp can start.
    # This is to avoid loading C too early, interfering with loading A and B.
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    TmemPtr = enum.auto()


class HopperWgmmaGemmKernelQuack:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernelQuack(
        ...     acc_dtype=Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    arch = 90

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mnk: tuple[int, int, int],
        epilogue_visitor_tree_cls: Callable[[type[cute.Numeric], tuple], EpilogueVisitorTree],
        pingpong: bool = False,
        is_persistent: bool = True,
        add_to_output: bool = False,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.add_to_output = add_to_output
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"

        self.cluster_shape_mnk = cluster_shape_mnk
        # K dimension is deferred in _setup_attributes
        self.cta_tile_shape_mnk = (*tile_shape_mn, 1)

        gemm_utils.check_tile_sizes(
            tile_M=self.cta_tile_shape_mnk[0],
            tile_N=self.cta_tile_shape_mnk[1],
            pingpong=self.pingpong,
        )
        atom_layout_m, atom_layout_n = gemm_utils.get_atom_layout(
            tile_M=self.cta_tile_shape_mnk[0],
            tile_N=self.cta_tile_shape_mnk[1],
            pingpong=self.pingpong,
        )
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)
        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.mma_warp_groups = gemm_utils.get_mma_warp_groups(
            atom_layout_mnk=self.atom_layout_mnk,
            pingpong=self.pingpong,
        )
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_90")

        self.num_epi_warps = gemm_utils.get_num_epi_warps(
            mma_warp_groups=self.mma_warp_groups,
            pingpong=self.pingpong,
        )
        self.num_ab_load_warps = 1
        self.ab_load_warp_id = self.mma_warp_groups * 4
        self.num_registers_load, self.num_registers_mma = gemm_utils.get_register_allocations(
            tile_shape_mnk=self.cta_tile_shape_mnk,
            atom_layout_mnk=self.atom_layout_mnk,
            mma_warp_groups=self.mma_warp_groups,
            num_threads_per_warp_group=self.num_threads_per_warp_group,
        )

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.epilogue_visitor_tree: EpilogueVisitorTree = epilogue_visitor_tree_cls(
            acc_dtype=self.acc_dtype,
            tile_shape_mnk=self.cta_tile_shape_mnk,
            buffer_align_bytes=self.buffer_align_bytes,
        )

    def _setup_attributes(self, epi_args: EpilogueVisitorTree.EpilogueArguments):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1]),
        )
        if cutlass.const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            tile_N = self.cta_tile_shape_mnk[1]
            atom_N = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, tile_N // atom_N // 8, atom_N),
                order=(0, 2, 1),
            )
            self.tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(self.tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )

        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.cta_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.epi_tile = gemm_utils.sm90_compute_tile_shape_or_override(
            self.cta_tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )

        # Compute stage before compute smem layout
        epi_smem_bytes_per_stage = self.epilogue_visitor_tree.get_smem_bytes_per_stage(
            epi_tile=self.epi_tile,
            epi_num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
            epi_args=epi_args,
        )
        self.ab_stage, self.epi_stage, self.epi_c_stage = gemm_utils.compute_stages(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}"),  # smem_capacity
            self.occupancy,
            epi_smem_bytes_per_stage=epi_smem_bytes_per_stage,
        )

        if cutlass.const_expr(self.pingpong):
            self.scheduler_stage = 2
        else:
            self.scheduler_stage = 1

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = gemm_utils.make_smem_layouts(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        d: cute.Tensor,
        epi_args: tuple,  # EpilogueVisitorTree.EpilogueArguments
        scheduler_args: TileSchedulerOptions,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.d_dtype = d.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.d_layout = utils.LayoutEnum.from_tensor(d)

        if cutlass.const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

        # Assume all strides are divisible by 128 bits except the last stride
        a = layout_utils.assumed_align_stride(
            tensor=a,
            assumed_align=16,
        )
        d = layout_utils.assumed_align_stride(
            tensor=d,
            assumed_align=16,
        )

        self._setup_attributes(epi_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout) +
            cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )

        tma_atom_a, tma_tensor_a = memory_utils.make_tma_atoms_and_tensors(
            op="g2s",
            tensor=a,
            smem_layout_staged=self.a_smem_layout_staged,
            smem_tile=(self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
            num_multicast=self.cluster_shape_mnk[1],
        )
        tma_atom_b, tma_tensor_b = memory_utils.make_tma_atoms_and_tensors(
            op="g2s",
            tensor=b,
            smem_layout_staged=self.b_smem_layout_staged,
            smem_tile=(self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
            num_multicast=self.cluster_shape_mnk[0],
        )
        epi_coord = cute.make_identity_layout(d.shape)
        epi_smem_tile = cute.composition(epi_coord, self.epi_tile)
        tma_atom_d, tma_tensor_d = memory_utils.make_tma_atoms_and_tensors(
            op="s2g" if cutlass.const_expr(not self.add_to_output) else "s2g-add",
            tensor=d,
            smem_layout_staged=self.epi_smem_layout_staged,
            smem_tile=epi_smem_tile,
        )

        # Epilogue
        epi_params = self.epilogue_visitor_tree.to_underlying_arguments(
            epi_tile=self.epi_tile,
            epi_stage=self.epi_stage,
            epi_load_stage=self.epi_c_stage,
            epi_args=epi_args,
        )

        # Scheduler
        tile_scheduler_args = self.get_scheduler_arguments(a, b, d, scheduler_args)
        tile_scheduler_params = TileScheduler.to_underlying_arguments(tile_scheduler_args)
        grid = TileScheduler.get_grid_shape(
            tile_scheduler_params,
            scheduler_args.max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            scheduler_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.scheduler_stage * 2]
            scheduler_data: cute.struct.MemRange[cutlass.Int32, self.scheduler_stage * 4]

            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, cute.cosize(self.epi_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Epilogue
            epilogue: self.epilogue_visitor_tree.get_smem_struct(
                epi_load_stage=self.epi_c_stage,
                epi_num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
                epi_params=epi_params,
            )

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        kernel = self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            epi_params,
            self.tiled_mma,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_scheduler_params,
        )
        kernel.launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        launch_check(kernel)

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        epi_params: EpilogueVisitorTree.EpilogueParams,
        tiled_mma: cute.TiledMma,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_scheduler_params: TileScheduler.Params,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_d: TMA copy atom for D tensor
        :type tma_atom_d: cute.CopyAtom
        :param mD_mnl: Output tensor D
        :type mD_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cluster_layout_mnk: CTA layout
        :type cluster_layout_mnk: cute.Layout
        :param a_smem_layout: Shared memory layout for A
        :type a_smem_layout: cute.ComposedLayout
        :param b_smem_layout: Shared memory layout for B
        :type b_smem_layout: cute.ComposedLayout
        :param epi_smem_layout: Shared memory layout for epilogue
        :type epi_smem_layout: cute.ComposedLayout
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.ab_load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            self.epilogue_visitor_tree.prefetch_tma_descriptors(epi_params=epi_params)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline = gemm_utils.make_mainloop_pipeline(
            num_stages=self.ab_stage,
            num_tma_load_bytes=self.num_tma_load_bytes,
            num_mcast_ctas_a=self.num_mcast_ctas_a,
            num_mcast_ctas_b=self.num_mcast_ctas_b,
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            mainloop_pipeline_mbar_ptr=storage.mainloop_pipeline_array_ptr.data_ptr(),
        )

        scheduler_pipeline = None
        scheduler_data = None
        if cutlass.const_expr(self.is_persistent):
            # Dynamic persistent scheduler
            scheduler_pipeline = gemm_utils.make_scheduler_pipeline(
                pingpong=self.pingpong,
                num_stages=self.scheduler_stage,
                dma_warps=self.num_ab_load_warps,
                mma_warp_groups=self.mma_warp_groups,
                cluster_layout_mnk=cluster_layout_mnk,
                scheduler_pipeline_mbar_ptr=storage.scheduler_pipeline_array_ptr.data_ptr(),
            )
            scheduler_data = storage.scheduler_data.get_tensor((4, self.scheduler_stage))

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)

        # Epilogue smem tensors
        epi_tensors_smem = self.epilogue_visitor_tree.get_smem_tensors(
            storage=storage.epilogue,
            epi_num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
            epi_params=epi_params,
        )
        epi_pipelines = self.epilogue_visitor_tree.prepare_pipelines(
            epi_load_stage=self.epi_c_stage,
            epi_num_warps=self.num_epi_warps,
            epi_params=epi_params,
            epi_tensors_smem=epi_tensors_smem,
        )

        k_tile_cnt = cute.ceil_div(mA_mkl.shape[1], self.cta_tile_shape_mnk[2])
        c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

        # Cluster wait for barrier init
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_registers_load)
            if (
                warp_idx >= self.ab_load_warp_id and
                warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                is_tma_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id

                # ///////////////////////////////////////////////////////////////////////////////
                # Get mcast mask
                # ///////////////////////////////////////////////////////////////////////////////
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if cutlass.const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0

                tile_scheduler = TileScheduler.create(
                    params=tile_scheduler_params,
                    sched_smem=scheduler_data,
                    scheduler_pipeline=scheduler_pipeline,
                )

                work_tile = tile_scheduler.initial_work_tile_info()
                mainloop_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )

                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]

                    # ///////////////////////////////////////////////////////////////////////////
                    #  Local_tile partition global tensors
                    # ///////////////////////////////////////////////////////////////////////////

                    # (bM, bK, RestK)
                    gA_mk = cute.local_tile(
                        mA_mkl[None, None, batch_idx],
                        cute.select(self.cta_tile_shape_mnk, [0, 2]),
                        (tile_coord_mnkl[0], None),
                    )

                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        mB_nkl[None, None, batch_idx],
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    # //////////////////////////////////////////////////////////////////////////
                    #  Partition shared tensor for TMA load A/B
                    # //////////////////////////////////////////////////////////////////////////
                    #  TMA load A partition_S/D
                    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_mnk, (0, None, 0)).shape)
                    a_cta_crd = block_in_cluster_coord_mnk[1]
                    sA_group_rank_smem = cutlass.const_expr(cute.rank(sA) - 1)
                    gA_group_rank_gmem = cutlass.const_expr(cute.rank(gA_mk) - 1)
                    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
                    tAsA, tAgA_mk = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_a,
                        a_cta_crd,
                        a_cta_layout,
                        cute.group_modes(sA, 0, sA_group_rank_smem),
                        cute.group_modes(gA_mk, 0, gA_group_rank_gmem),
                    )

                    # TMA load B partition_S/D
                    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape)
                    b_cta_crd = block_in_cluster_coord_mnk[0]
                    sB_group_rank_smem = cutlass.const_expr(cute.rank(sB) - 1)
                    gB_group_rank_gmem = cutlass.const_expr(cute.rank(gB_nk) - 1)
                    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
                    tBsB, tBgB_nk = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_b,
                        b_cta_crd,
                        b_cta_layout,
                        cute.group_modes(sB, 0, sB_group_rank_smem),
                        cute.group_modes(gB_nk, 0, gB_group_rank_gmem),
                    )

                    mainloop_producer_state = self.load_AB(
                        mainloop_pipeline=mainloop_pipeline,
                        mainloop_producer_state=mainloop_producer_state,
                        tma_atom_a=tma_atom_a,
                        tma_atom_b=tma_atom_b,
                        tAgA_mk=tAgA_mk,
                        tBgB_nk=tBgB_nk,
                        tAsA=tAsA,
                        tBsB=tBsB,
                        a_mcast_mask=a_mcast_mask,
                        b_mcast_mask=b_mcast_mask,
                        k_tile_cnt=k_tile_cnt,
                    )

                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop

                if cutlass.const_expr(self.pingpong):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()

                mainloop_pipeline.producer_tail(mainloop_producer_state)

                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        if warp_idx < self.ab_load_warp_id:
            cute.arch.setmaxregister_increase(self.num_registers_mma)
            is_tma_warp = cute.Boolean(
                (not self.pingpong and warp_idx == 0) or
                (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            # //////////////////////////////////////////////////////////////////////////////
            #  Partition global tensor for TiledMMA_A/B/C
            # //////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if cutlass.const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if cutlass.const_expr(not self.pingpong) else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )

            # //////////////////////////////////////////////////////////////////////////////
            #  Make fragments
            # //////////////////////////////////////////////////////////////////////////////
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))

            acc_shape = tiled_mma.partition_shape_C(
                cute.select(self.cta_tile_shape_mnk, mode=[0, 1])
            )
            acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            if cutlass.const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            # Threads/warps participating in tma store pipeline
            num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
            epi_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_epi_threads
            )
            epi_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage, producer_group=epi_store_producer_group
            )

            tile_scheduler = TileScheduler.create(
                params=tile_scheduler_params,
                sched_smem=scheduler_data,
                scheduler_pipeline=scheduler_pipeline,
            )

            work_tile = tile_scheduler.initial_work_tile_info()
            if cutlass.const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    mainloop_consumer_read_state = pipeline_utils.advance_n(
                        state=mainloop_consumer_read_state,
                        num_iterations=k_tile_cnt,
                    )
                    epi_pipelines = self.epilogue_visitor_tree.advance_pipelines(
                        tile_count=c_tile_cnt,
                        epi_params=epi_params,
                        epi_pipelines=epi_pipelines,
                    )
                    # TODO: do we need to check if work_tile is valid?
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()

            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                mainloop_consumer_read_state, tiled_mma = self.mma(
                    mainloop_pipeline=mainloop_pipeline,
                    mainloop_consumer_read_state=mainloop_consumer_read_state,
                    tiled_mma=tiled_mma,
                    tCrA=tCrA,
                    tCrB=tCrB,
                    acc=acc,
                    k_tile_cnt=k_tile_cnt,
                    warp_group_idx=warp_group_idx,
                )

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOGUE
                # /////////////////////////////////////////////////////////////////////////////
                if cutlass.const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")

                epilogue_barrier = pipeline.NamedBarrier(
                    barrier_id=int(NamedBarrierGemm.Epilogue),
                    num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
                )

                # (bM, bN)
                gD_mn = cute.local_tile(
                    mD_mnl[None, None, batch_idx],
                    self.cta_tile_shape_mnk[:2],
                    tile_coord_mnkl[:2],
                )
                tDgD_for_tma_partition = cute.zipped_divide(gD_mn, self.epi_tile)

                sD_group_rank_smem = cutlass.const_expr(cute.rank(sD) - 1)
                gD_group_rank_gmem = cutlass.const_expr(cute.rank(tDgD_for_tma_partition) - 1)
                # thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
                # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_d,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sD, 0, sD_group_rank_smem),
                    cute.group_modes(tDgD_for_tma_partition, 0, gD_group_rank_gmem),
                )

                # Doesn't work with tile_N % 8 == 0 but tile_n % 16 != since this always
                # get st.matrix with num_matrices=4
                if cutlass.const_expr(self.epi_tile[1] % 16 == 0):
                    epi_num_matrices = 4
                else:
                    epi_num_matrices = 2

                # Partition for epilogue
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.d_layout,
                    elem_ty_d=self.d_dtype,
                    elem_ty_acc=self.acc_dtype,
                )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.d_layout.is_m_major_c(),
                        num_matrices=epi_num_matrices,
                    ),
                    self.d_dtype,
                )

                tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_atom,
                )

                # (R2S, R2S_M, R2S_N, PIPE_D)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                # (t)hread-partition for (r)egister to (s)mem copy (tRS_)
                tRS_sD = thr_copy_r2s.partition_D(sD)
                sD_shape = sD.shape[:2]
                tRS_rD_shape = thr_copy_r2s.partition_S(cute.make_identity_tensor(sD_shape)).shape
                tRS_rD = cute.make_rmem_tensor(tRS_rD_shape, self.acc_dtype)

                # (R2S, R2S_M, R2S_N)
                tRS_rAcc = tiled_copy_r2s.retile(acc)

                epi_pipelines = self.epilogue(
                    epi_params=epi_params,
                    epi_pipelines=epi_pipelines,
                    epi_tensors_smem=epi_tensors_smem,
                    epi_store_pipeline=epi_store_pipeline,
                    epi_tile=self.epi_tile,
                    epi_num_matrices=epi_num_matrices,
                    epi_barrier=epilogue_barrier,
                    tRS_rAcc=tRS_rAcc,
                    tRS_rD=tRS_rD,
                    tRS_sD=tRS_sD,
                    bSG_sD=bSG_sD,
                    bSG_gD=bSG_gD,
                    shape_mnk=(
                        mD_mnl.shape[0],
                        mD_mnl.shape[1],
                        mA_mkl.shape[1],
                    ),
                    tma_atom_d=tma_atom_d,
                    tiled_mma=tiled_mma,
                    tiled_copy_r2s=tiled_copy_r2s,
                    tile_coord_mnkl=tile_coord_mnkl,
                    tile_scheduler=tile_scheduler,
                    tidx=tidx,
                    is_tma_warp=is_tma_warp,
                )

                if cutlass.const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

                if cutlass.const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                    work_tile = tile_scheduler.get_current_work()
                    # Update starting mainloop pipeline state for the next tile
                    mainloop_consumer_read_state = pipeline_utils.advance_n(
                        state=mainloop_consumer_read_state,
                        num_iterations=k_tile_cnt,
                    )
                    # Update starting load/store pipeline states for the next tile
                    epi_pipelines = self.epilogue_visitor_tree.advance_pipelines(
                        tile_count=c_tile_cnt,
                        epi_params=epi_params,
                        epi_pipelines=epi_pipelines,
                    )

                # End of persistent scheduler loop

            # Wait for D store complete
            if cutlass.const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

    @cute.jit
    def load_AB(
        self,
        mainloop_pipeline: cutlass.pipeline.PipelineAsync,
        mainloop_producer_state: cutlass.pipeline.PipelineState,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tAgA_mk: cute.Tensor,
        tBgB_nk: cute.Tensor,
        tAsA: cute.Tensor,
        tBsB: cute.Tensor,
        a_mcast_mask: cute.Int16,
        b_mcast_mask: cute.Int16,
        k_tile_cnt: cute.Int32,
    ) -> cutlass.pipeline.PipelineState:

        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = cute.Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = mainloop_pipeline.producer_try_acquire(mainloop_producer_state)
        # /////////////////////////////////////////////////////////////////////////
        # TMA load
        # /////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            mainloop_pipeline.producer_acquire(mainloop_producer_state, peek_ab_empty_status)
            tma_bar_ptr = mainloop_pipeline.producer_get_barrier(mainloop_producer_state)
            smem_idx = mainloop_producer_state.index
            cute.copy(
                tma_atom_a,
                tAgA_mk[None, k_tile],
                tAsA[None, smem_idx],
                tma_bar_ptr=tma_bar_ptr,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_b,
                tBgB_nk[None, k_tile],
                tBsB[None, smem_idx],
                tma_bar_ptr=tma_bar_ptr,
                mcast_mask=b_mcast_mask,
            )

            # Mainloop pipeline's producer commit is a NOP
            mainloop_pipeline.producer_commit(mainloop_producer_state)
            mainloop_producer_state.advance()

            peek_ab_empty_status = cute.Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = mainloop_pipeline.producer_try_acquire(mainloop_producer_state)

        return mainloop_producer_state

    @cute.jit
    def mma(
        self,
        mainloop_pipeline: cutlass.pipeline.PipelineAsync,
        mainloop_consumer_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        acc: cute.Tensor,
        k_tile_cnt: cute.Int32,
        warp_group_idx: cute.Int32,
    ) -> tuple[cutlass.pipeline.PipelineState, cute.TiledMma]:
        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1
        mainloop_consumer_release_state = mainloop_consumer_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        if cutlass.const_expr(self.pingpong):
            self.pingpong_barrier_sync(warp_group_idx, stage="mma")
        peek_ab_full_status = cute.Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state,
            )
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])

        for k_tile in cutlass.range(num_prologue_mma):
            # Wait for TMA copies to complete
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state,
                peek_ab_full_status,
            )
            # WGMMA
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    acc,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cute.Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state,
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            # Wait for TMA copies to complete
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state,
                peek_ab_full_status,
            )
            # WGMMA
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    acc,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()
            peek_ab_full_status = cute.Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state,
                )

        if cutlass.const_expr(self.pingpong):
            # Cue for next WG's MMA to start
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")

        cute.nvgpu.warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
            mainloop_consumer_release_state.advance()

        # If we don't return the tiled_mma, we get compiler error
        # "operand #0 does not dominate this use"
        return mainloop_consumer_read_state, tiled_mma

    @cute.jit
    def epilogue(
        self,
        epi_params: EpilogueVisitorTree.EpilogueParams,
        epi_pipelines: EpilogueVisitorTree.EpiloguePipelines,
        epi_tensors_smem: EpilogueVisitorTree.EpilogueTensorsSMem,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_tile: cute.Tile,
        epi_num_matrices: int,
        epi_barrier: cutlass.pipeline.NamedBarrier,
        tRS_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        tRS_sD: cute.Tensor,
        bSG_sD: cute.Tensor,
        bSG_gD: cute.Tensor,
        shape_mnk: tuple[int, int, int],
        tma_atom_d: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        tile_scheduler: TileScheduler,
        tidx: cute.Int32,
        is_tma_warp: cute.Boolean,
    ) -> EpilogueVisitorTree.EpiloguePipelines:

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1, 0))
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        # Pre-loop fusion callback entry point
        epi_tensors = self.epilogue_visitor_tree.consumer_begin(
            tiled_copy_r2s=tiled_copy_r2s,
            tile_coord_mnkl=tile_coord_mnkl,
            tidx=tidx,
            tiled_mma=tiled_mma,
            tRS_rD_layout=tRS_rD.layout,
            epi_tile=epi_tile,
            epi_num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
            epi_num_matrices=epi_num_matrices,
            epi_barrier=epi_barrier,
            epi_params=epi_params,
            epi_tensors_smem=epi_tensors_smem,
        )
        epi_pipelines = self.epilogue_visitor_tree.producer_begin(
            is_tma_warp=is_tma_warp,
            epi_load_stage=self.epi_c_stage,
            epi_tile_num=epi_tile_num,
            epi_tile_layout=epi_tile_layout,
            epi_params=epi_params,
            epi_tensors=epi_tensors,
            epi_pipelines=epi_pipelines,
        )

        def tma_store_fn(src_idx, dst_idx, epi_loop_tensors):
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            # Copy from shared memory to global memory
            if is_tma_warp:
                cute.copy(
                    tma_atom_d,
                    bSG_sD[None, src_idx],
                    bSG_gD[None, dst_idx],
                )
                self.epilogue_visitor_tree.consumer_tma_store(
                    epi_coord=dst_idx,
                    epi_buffer=src_idx,
                    epi_params=epi_params,
                    epi_tensors_loop=epi_loop_tensors,
                )

            # Can't use if statement here, epi_store_pipeline object isn't captured somehow
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_commit())
            if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_acquire())
            epi_barrier.arrive_and_wait()

        # We could delay the TMA store by 1 epi tile to better overlap the non-TMA ops
        # with the TMA store. However, currently this doesn't seem to improve perf.
        delay_tma_store = False
        src_idx_prev = None
        dst_idx_prev = None
        epi_tensors_loop_prev = None

        for epi_idx in cutlass.range_constexpr(epi_tile_num):

            # The global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)

            # Per-loop fusion callback entry point
            epi_tensors_loop, epi_pipelines = self.epilogue_visitor_tree.consumer_begin_loop(
                epi_coord=gmem_coord,
                epi_params=epi_params,
                epi_tensors=epi_tensors,
                epi_pipelines=epi_pipelines,
            )
            epi_pipelines = self.epilogue_visitor_tree.producer_tma_load(
                is_tma_warp=is_tma_warp,
                epi_idx=epi_idx,
                epi_load_stage=self.epi_c_stage,
                epi_tile_num=epi_tile_num,
                epi_tile_layout=epi_tile_layout,
                epi_params=epi_params,
                epi_tensors=epi_tensors,
                epi_pipelines=epi_pipelines,
            )

            # Copy from acc to D registers
            for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]

            # perform thread-local computations
            epi_tensors_loop = self.epilogue_visitor_tree.consumer_visit(
                tRS_rD=tRS_rD,
                shape_mnk=shape_mnk,
                epi_params=epi_params,
                epi_tensors_loop=epi_tensors_loop,
            )

            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            if cutlass.const_expr(delay_tma_store):
                if cutlass.const_expr(epi_idx > 0):
                    tma_store_fn(
                        src_idx=src_idx_prev,
                        dst_idx=dst_idx_prev,
                        epi_loop_tensors=epi_tensors_loop_prev,
                    )
                src_idx_prev = epi_buffer
                dst_idx_prev = gmem_coord
                epi_tensors_loop_prev = epi_tensors_loop

            # Type conversion
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD, self.d_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.d_dtype))

            # Copy from D registers to shared memory
            cute.copy(
                tiled_copy_r2s,
                tRS_rD_out,
                tRS_sD[None, None, None, epi_buffer],
            )

            # Pre TMA store callback entry point and smem async fence.
            # Smem stores usually performed here. Upon exit, all smem
            # stores for TMA must have been issued
            self.epilogue_visitor_tree.consumer_smem_store(
                epi_coord=gmem_coord,
                epi_buffer=epi_buffer,
                epi_params=epi_params,
                epi_tensors_loop=epi_tensors_loop,
            )

            if cutlass.const_expr(not delay_tma_store):
                tma_store_fn(
                    src_idx=epi_buffer,
                    dst_idx=gmem_coord,
                    epi_loop_tensors=epi_tensors_loop,
                )

            # Per-loop fusion callback entry point
            self.epilogue_visitor_tree.consumer_end_loop(gmem_coord)

        if cutlass.const_expr(delay_tma_store):
            tma_store_fn(
                src_idx=src_idx_prev,
                dst_idx=dst_idx_prev,
                epi_loop_tensors=epi_tensors_loop_prev,
            )

        # Post-loop fusion callback entry point
        self.epilogue_visitor_tree.consumer_end(
            tiled_copy_r2s=tiled_copy_r2s,
            tile_coord_mnkl=tile_coord_mnkl,
            tidx=tidx,
            shape_mnk=shape_mnk,
            epi_tile=epi_tile,
            epi_num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
            epi_barrier=epi_barrier,
            epi_params=epi_params,
            epi_tensors=epi_tensors,
            epi_tensors_smem=epi_tensors_smem,
        )
        return epi_pipelines

    def get_scheduler_arguments(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        scheduler_args,
    ):
        """Create scheduler arguments. Override in subclasses for custom schedulers."""
        if cutlass.const_expr(not self.is_persistent):
            persistence_mode = PersistenceMode.NONE
        else:
            if cutlass.const_expr(self.arch >= 100):
                raise NotImplementedError
            elif cutlass.const_expr(scheduler_args.tile_count_semaphore is not None):
                persistence_mode = PersistenceMode.DYNAMIC
            else:
                persistence_mode = PersistenceMode.STATIC

        num_problems = mD.shape[2]
        problem_shape_ntile_mnl = (
            cute.ceil_div(mA.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(mB.shape[0], self.cta_tile_shape_mnk[1]),
            num_problems,
        )
        tile_scheduler_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=problem_shape_ntile_mnl,
            raster_order=scheduler_args.raster_order,
            group_size=scheduler_args.max_swizzle_size,
            cluster_shape_mnk=self.cluster_shape_mnk,
            tile_count_semaphore=scheduler_args.tile_count_semaphore,
            batch_idx_permute=scheduler_args.batch_idx_permute,
            persistence_mode=persistence_mode,
        )

        return tile_scheduler_args

    def pingpong_barrier_sync(
        self,
        warp_group_idx: cute.Int32,
        stage: str,
    ) -> None:
        misc_utils.static_assert(stage in ["mma", "epi"])
        if cutlass.const_expr(stage == "mma"):
            barrier = NamedBarrierGemm.MmaWG0
        else:
            barrier = NamedBarrierGemm.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(
        self,
        warp_group_idx: cute.Int32,
        stage: str,
    ) -> None:
        misc_utils.static_assert(stage in ["mma", "epi"])
        if cutlass.const_expr(stage == "mma"):
            barrier = NamedBarrierGemm.MmaWG0
        else:
            barrier = NamedBarrierGemm.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )
