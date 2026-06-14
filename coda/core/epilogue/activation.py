import cutlass
import cutlass.cute as cute
from typing import Callable, NamedTuple
from dataclasses import dataclass

from coda.core.ops import misc_utils
from coda.core.ops import dtype_utils
from coda.core.ops import struct_utils
from coda.core.ops import layout_utils
from coda.core.ops import creation_utils
from coda.core.ops import epilogue_utils
from coda.core.epilogue.base import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)


class EVTActivationWithDualOutputs(EpilogueVisitorTree):
    """
    Applies activation functions with shape-preserving or dimension-changing modes.

    Produces dual outputs: pre-activation (C) and post-activation (mPostAct) in one pass.
    Output dimensions depend on mode.

    Input:
        - GEMM output: [M x N]

    Output modes:
        - **elementwise**: 1→1 pointwise transform
          * Pre-activation (C): [M x N]
          * Post-activation (mPostAct): [M x N]

        - **pairwise**: 2→2 pairwise transform
          * Pre-activation (C): [M x N]
          * Post-activation (mPostAct): [M x N]

        - **contraction**: 2→1 pairwise reduction
          * Pre-activation (C): [M x N]
          * Post-activation (mPostAct): [M x N/2]

        - **expansion**: 1→2 unpacking
          * Pre-activation (C): [M x N]
          * Post-activation (mPostAct): [M x 2N]

    Implementation:
        Pre-activation via standard epilogue, post-activation via TMA store to separate buffer.

    Use cases:
        - Fused activation with forward/backward storage
        - Gated activations in transformer FFNs
        - Memory-efficient training requiring both values

    Example (elementwise):
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTActivationWithDualOutputs(
        ...         fn=lambda x: ...,
        ...         ftype="elementwise",
        ...         acc_dtype=acc_dtype,
        ...         post_act_dtype=cute.Float16,
        ...         tile_shape_mnk=tile_shape_mnk,
        ...         buffer_align_bytes=buffer_align_bytes,
        ...     )

    Example (contraction):
        >>> epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: \\
        ...     EVTActivationWithDualOutputs(
        ...         fn=lambda x0, x1: ...,
        ...         ftype="contraction",
        ...         acc_dtype=acc_dtype,
        ...         post_act_dtype=cute.Float16,
        ...         tile_shape_mnk=tile_shape_mnk,
        ...         buffer_align_bytes=buffer_align_bytes,
        ...     )
        >>> epi_args = EVTActivationWithDualOutputs.EpilogueArguments(
        ...     mPostAct=post_act_tensor,  # Shape depends on ftype
        ... )
    """

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        mPostAct: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        mPostAct: cute.Tensor | None
        epi_tma_atom: cute.CopyAtom
        epi_gmem_layout: cutlass.utils.LayoutEnum
        epi_smem_layout_staged: cute.Layout

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        sPostAct: cute.Tensor | None

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        tDsPostAct: cute.Tensor
        tDgPostAct: cute.Tensor
        # tRS_rPostAct: cute.Tensor | None
        tRS_sPostAct: cute.Tensor
        epi_tma_atom: cute.CopyAtom
        tiled_copy_postact_r2s: cute.TiledCopy

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        tDsPostAct: cute.Tensor
        tDgPostAct: cute.Tensor
        tRS_rPostAct: cute.Tensor | None
        tRS_sPostAct: cute.Tensor
        epi_tma_atom: cute.CopyAtom
        tiled_copy_postact_r2s: cute.TiledCopy

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        pass

    def __init__(
        self,
        fn: cutlass.Constexpr[Callable],
        ftype: str,
        acc_dtype: type[cute.Numeric],
        post_act_dtype: type[cute.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        buffer_align_bytes: int,
    ) -> None:
        super().__init__()

        misc_utils.static_assert(ftype in ["elementwise", "pairwise", "contraction", "expansion"])
        if cutlass.const_expr(ftype == "contraction"):
            misc_utils.static_assert(post_act_dtype.width == 16)
            misc_utils.static_assert(tile_shape_mnk[1] % 32 == 0)
        if cutlass.const_expr(ftype == "expansion"):
            misc_utils.static_assert(post_act_dtype.width == 16)
            container_dtype = cute.Float32
        else:
            container_dtype = post_act_dtype

        self.arch = 90
        self.fn = fn
        self.ftype = ftype
        self.acc_dtype = acc_dtype
        self.post_act_dtype = post_act_dtype
        self.container_dtype = container_dtype
        self.tile_shape_mnk = tile_shape_mnk
        self.buffer_align_bytes = buffer_align_bytes

    @staticmethod
    def postprocess_epi_tile(epi_tile: cute.Tile) -> cute.Tile:
        misc_utils.static_assert(isinstance(epi_tile, tuple))
        misc_utils.static_assert(isinstance(epi_tile[0], int))
        misc_utils.static_assert(isinstance(epi_tile[1], int))
        misc_utils.static_assert(len(epi_tile) == 2)
        return (
            epi_tile[0],
            epi_tile[1] // 2,
        )

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:

        if cutlass.const_expr(self.ftype == "contraction"):
            epi_tile = self.postprocess_epi_tile(epi_tile)

        if cutlass.const_expr(epi_args.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_args.mPostAct)
            misc_utils.static_assert(misc_utils.get_dtype(mPostAct) is self.container_dtype)
            (
                epi_gmem_layout,
                epi_smem_layout_staged,
                epi_tma_atom,
                epi_tma_tensor,
            ) = epilogue_utils.prepare_tma(
                tma_op="s2g",
                epi_tile=epi_tile,
                # we use `epi_stage` here
                # since this is for storing
                epi_stage=epi_stage,
                epi_tensor=mPostAct,
            )

        return self.EpilogueParams(
            mPostAct=epi_tma_tensor,
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

        if cutlass.const_expr(self.ftype == "contraction"):
            tile_N = tile_N // 2
            epi_tile = self.postprocess_epi_tile(epi_tile)

        if cutlass.const_expr(epi_params.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_params.mPostAct)
            sPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_smem.sPostAct)
            # RMem -> SMem Copy
            tiled_copy_postact_r2s, _, tRS_sPostAct = epilogue_utils.prepare_copy_r2s_sm90(
                tiled_copy_r2s=tiled_copy_r2s,
                tidx=tidx,
                dst=sPostAct,
                epi_layout=epi_params.epi_gmem_layout,
                epi_dtype=self.container_dtype,
                acc_dtype=self.acc_dtype,
            )
            # SMem -> GMem Copy
            # (bM, bN)
            gPostAct = mPostAct[None, None, batch_idx]
            gPostAct = cute.local_tile(gPostAct, (tile_M, tile_N), (m_idx, n_idx))
            gPostAct = cute.zipped_divide(gPostAct, epi_tile)

            # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
            tDsPostAct, tDgPostAct = cute.nvgpu.cpasync.tma_partition(
                atom=epi_params.epi_tma_atom,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
                gmem_tensor=cute.group_modes(gPostAct, 0, cute.rank(gPostAct) - 1),
            )
            filter_zeros = cutlass.const_expr(False)
            if cutlass.const_expr(filter_zeros):
                tDsPostAct = cute.filter_zeros(tDsPostAct)
                tDgPostAct = cute.filter_zeros(tDgPostAct)

        return self.EpilogueTensors(
            tDsPostAct=tDsPostAct,
            tDgPostAct=tDgPostAct,
            # tRS_rPostAct=None,
            tRS_sPostAct=tRS_sPostAct,
            epi_tma_atom=epi_params.epi_tma_atom,
            tiled_copy_postact_r2s=tiled_copy_postact_r2s,
        )

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:
        return (
            self.EpilogueTensorsLoop(
                tDsPostAct=epi_tensors.tDsPostAct,
                tDgPostAct=epi_tensors.tDgPostAct,
                tRS_rPostAct=None,
                tRS_sPostAct=epi_tensors.tRS_sPostAct,
                epi_tma_atom=epi_tensors.epi_tma_atom,
                tiled_copy_postact_r2s=epi_tensors.tiled_copy_postact_r2s,
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

        if cutlass.const_expr(self.ftype == "elementwise"):
            tRS_rPostAct = creation_utils.allocate_tensor_like(
                tensor=tRS_rD,
                memspace="rmem",
                smem_allocator=None,
                dtype=self.acc_dtype,
            )
            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tRS_rPostAct)):
                    tRS_rPostAct[i] = self.fn(tRS_rD[i])
            else:
                raise NotImplementedError

            tRS_rPostAct = dtype_utils.convert(
                tRS_rPostAct,
                dtype=self.post_act_dtype,
            )

        elif cutlass.const_expr(self.ftype == "pairwise"):
            tRS_rPostAct = creation_utils.allocate_tensor_like(
                tensor=tRS_rD,
                memspace="rmem",
                smem_allocator=None,
                dtype=self.acc_dtype,
            )
            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tRS_rPostAct) // 2):
                    tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = self.fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
            else:
                raise NotImplementedError

            tRS_rPostAct = dtype_utils.convert(
                tRS_rPostAct,
                dtype=self.post_act_dtype,
            )

        elif cutlass.const_expr(self.ftype == "contraction"):
            tRS_rPostAct = creation_utils.allocate_tensor_from_recast_layout(
                layout=tRS_rD.layout,
                new_type_bits=2,
                old_type_bits=1,
                memspace="rmem",
                smem_allocator=None,
                dtype=self.acc_dtype,
            )
            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tRS_rPostAct)):
                    tRS_rPostAct[i] = self.fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
            else:
                raise NotImplementedError

            tRS_rPostAct = dtype_utils.convert(
                tRS_rPostAct,
                dtype=self.post_act_dtype,
            )
            if cutlass.const_expr(self.arch == 90):
                # Only need this if we're using STSM
                layout_utils.permute_gated_Cregs_b16(tRS_rPostAct)

        elif cutlass.const_expr(self.ftype == "expansion"):
            misc_utils.static_assert(misc_utils.get_dtype(tRS_rD).width == 32)
            misc_utils.static_assert(self.post_act_dtype.width == 16)
            misc_utils.static_assert(self.container_dtype.width == 32)
            tRS_rPostAct = creation_utils.allocate_tensor_from_recast_layout(
                layout=tRS_rD.layout,
                new_type_bits=self.post_act_dtype.width,
                old_type_bits=misc_utils.get_dtype(tRS_rD).width,
                memspace="rmem",
                smem_allocator=None,
                dtype=self.acc_dtype,
            )

            if cutlass.const_expr(self.arch < 100):
                for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                    tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = self.fn(tRS_rD[i])
            else:
                raise NotImplementedError

            tRS_rPostAct = dtype_utils.convert(
                tRS_rPostAct,
                dtype=self.post_act_dtype,
            )
            tRS_rPostAct = cute.recast_tensor(
                tRS_rPostAct,
                dtype=self.container_dtype,
            )

        else:
            raise NotImplementedError

        return self.EpilogueTensorsLoop(
            tDsPostAct=epi_tensors_loop.tDsPostAct,
            tDgPostAct=epi_tensors_loop.tDgPostAct,
            tRS_rPostAct=tRS_rPostAct,
            tRS_sPostAct=epi_tensors_loop.tRS_sPostAct,
            epi_tma_atom=epi_tensors_loop.epi_tma_atom,
            tiled_copy_postact_r2s=epi_tensors_loop.tiled_copy_postact_r2s,
        )

    @cute.jit
    def consumer_smem_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        tiled_copy = epi_tensors_loop.tiled_copy_postact_r2s
        tRS_rPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_rPostAct)
        tRS_sPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tRS_sPostAct)
        src = tiled_copy.retile(tRS_rPostAct)
        dst = tRS_sPostAct[None, None, None, epi_buffer]
        cute.copy(atom=tiled_copy, src=src, dst=dst)

    @cute.jit
    def consumer_tma_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        atom = epi_tensors_loop.epi_tma_atom
        tDsPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDsPostAct)
        tDgPostAct = misc_utils.static_assert_is_Tensor(epi_tensors_loop.tDgPostAct)
        src = tDsPostAct[None, epi_buffer]
        dst = tDgPostAct[None, epi_coord]
        cute.copy(atom=atom, src=src, dst=dst)

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:

        if cutlass.const_expr(epi_params.mPostAct is not None):
            post_act_smem_size = cute.cosize(epi_params.epi_smem_layout_staged)
        else:
            post_act_smem_size = 0

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            sPostAct: cute.struct.Align[cute.struct.MemRange[self.container_dtype, post_act_smem_size], self.buffer_align_bytes]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:

        if cutlass.const_expr(epi_params.mPostAct is not None):
            sPostAct = storage.sPostAct.get_tensor(
                epi_params.epi_smem_layout_staged.outer,
                swizzle=epi_params.epi_smem_layout_staged.inner,
            )
        else:
            sPostAct = None

        return self.EpilogueTensorsSMem(
            sPostAct=sPostAct,
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

        if cutlass.const_expr(self.ftype == "contraction"):
            epi_tile = self.postprocess_epi_tile(epi_tile)

        # we store post-activation tensors, hence this is under
        # epilogue consumers, and since this is done in
        # stages, the smem storage is per stage
        if cutlass.const_expr(epi_args.mPostAct is not None):
            mPostAct = misc_utils.static_assert_is_Tensor(epi_args.mPostAct)
            misc_utils.static_assert(misc_utils.get_dtype(mPostAct) is self.container_dtype)
            # when TMA is used, the tile size should be `cute.size(epi_tile)`
            epi_smem_bytes_per_stage_cst = epi_smem_bytes_per_stage_cst + (
                epilogue_utils.get_epi_smem_bytes_per_stage_matrix(
                    mTensor=mPostAct,
                    epi_tile=epi_tile,
                )
            )

        return (
            epi_smem_bytes_fixed,
            epi_smem_bytes_per_stage_cst,
            epi_smem_bytes_per_stage_pld,
        )
