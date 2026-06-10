import cutlass
import cutlass.cute as cute
from typing import NamedTuple
from dataclasses import dataclass

from coda.core.ops import misc_utils
from coda.core.ops import struct_utils
from coda.core.epilogue.base import (
    EpilogueVisitorTree,
    EpilogueSharedStorage,
)


class EVTList(EpilogueVisitorTree):

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        list_: list[EpilogueVisitorTree.EpilogueArguments]

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(EpilogueVisitorTree.EpilogueParams):
        # Device side
        list_: list[EpilogueVisitorTree.EpilogueParams]

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(EpilogueVisitorTree.EpilogueTensorsSMem):
        list_: list[EpilogueVisitorTree.EpilogueTensorsSMem]

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(EpilogueVisitorTree.EpilogueTensors):
        list_: list[EpilogueVisitorTree.EpilogueTensors]

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(EpilogueVisitorTree.EpilogueTensorsLoop):
        list_: list[EpilogueVisitorTree.EpilogueTensorsLoop]

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(EpilogueVisitorTree.EpiloguePipelines):
        list_: list[EpilogueVisitorTree.EpiloguePipelines]

    def __init__(
        self,
        evts: list[EpilogueVisitorTree],
    ) -> None:
        super().__init__()
        self.evts = evts

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:
        misc_utils.static_assert(len(self.evts) == len(epi_args.list_))
        epi_params_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_params = self.evts[i].to_underlying_arguments(
                epi_tile=epi_tile,
                epi_stage=epi_stage,
                epi_load_stage=epi_load_stage,
                epi_args=epi_args.list_[i],
            )
            epi_params_list.append(epi_params)

        return self.EpilogueParams(list_=epi_params_list)

    @cute.jit
    def prefetch_tma_descriptors(
        self,
        epi_params: EpilogueParams,
    ) -> None:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        for i in cutlass.range_constexpr(len(self.evts)):
            self.evts[i].prefetch_tma_descriptors(
                epi_params=epi_params.list_[i],
            )

    @cute.jit
    def prepare_pipelines(
        self,
        epi_load_stage: int,
        epi_num_warps: int,
        epi_params: EpilogueParams,
        epi_tensors_smem: EpilogueTensorsSMem,
    ) -> EpiloguePipelines:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors_smem.list_))
        epi_pipelines_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_pipelines = self.evts[i].prepare_pipelines(
                epi_load_stage=epi_load_stage,
                epi_num_warps=epi_num_warps,
                epi_params=epi_params.list_[i],
                epi_tensors_smem=epi_tensors_smem.list_[i],
            )
            epi_pipelines_list.append(epi_pipelines)

        return self.EpiloguePipelines(list_=epi_pipelines_list)

    @cute.jit
    def advance_pipelines(
        self,
        tile_count: int,
        epi_params: EpilogueParams,
        epi_pipelines: EpiloguePipelines,
    ) -> EpiloguePipelines:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_pipelines.list_))
        epi_pipelines_new_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_pipelines_new = self.evts[i].advance_pipelines(
                tile_count=tile_count,
                epi_params=epi_params.list_[i],
                epi_pipelines=epi_pipelines.list_[i],
            )
            epi_pipelines_new_list.append(epi_pipelines_new)

        return self.EpiloguePipelines(list_=epi_pipelines_new_list)

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
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors_smem.list_))
        epi_tensors_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_tensors = self.evts[i].consumer_begin(
                tiled_copy_r2s=tiled_copy_r2s,
                tile_coord_mnkl=tile_coord_mnkl,
                tidx=tidx,
                tiled_mma=tiled_mma,
                tRS_rD_layout=tRS_rD_layout,
                epi_tile=epi_tile,
                epi_num_threads=epi_num_threads,
                epi_num_matrices=epi_num_matrices,
                epi_barrier=epi_barrier,
                epi_params=epi_params.list_[i],
                epi_tensors_smem=epi_tensors_smem.list_[i],
            )
            epi_tensors_list.append(epi_tensors)

        return self.EpilogueTensors(list_=epi_tensors_list)

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
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors_smem.list_))
        for i in cutlass.range_constexpr(len(self.evts)):
            self.evts[i].consumer_end(
                tiled_copy_r2s=tiled_copy_r2s,
                tile_coord_mnkl=tile_coord_mnkl,
                tidx=tidx,
                shape_mnk=shape_mnk,
                epi_tile=epi_tile,
                epi_num_threads=epi_num_threads,
                epi_barrier=epi_barrier,
                epi_params=epi_params.list_[i],
                epi_tensors=epi_tensors.list_[i],
                epi_tensors_smem=epi_tensors_smem.list_[i],
            )

    @cute.jit
    def consumer_begin_loop(
        self,
        epi_coord: cute.Coord,
        epi_params: EpilogueParams,
        epi_tensors: EpilogueTensors,
        epi_pipelines: EpiloguePipelines,
    ) -> tuple[EpilogueTensorsLoop, EpiloguePipelines]:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_pipelines.list_))
        epi_tensors_loop_list = []
        epi_pipelines_new_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_tensors_loop, epi_pipelines_new = self.evts[i].consumer_begin_loop(
                epi_coord=epi_coord,
                epi_params=epi_params.list_[i],
                epi_tensors=epi_tensors.list_[i],
                epi_pipelines=epi_pipelines.list_[i],
            )
            epi_tensors_loop_list.append(epi_tensors_loop)
            epi_pipelines_new_list.append(epi_pipelines_new)

        return (
            self.EpilogueTensorsLoop(list_=epi_tensors_loop_list),
            self.EpiloguePipelines(list_=epi_pipelines_new_list),
        )

    @cute.jit
    def consumer_end_loop(
        self,
        *args,
        **kwargs,
    ) -> None:
        for i in cutlass.range_constexpr(len(self.evts)):
            self.evts[i].consumer_end_loop(*args, **kwargs)

    @cute.jit
    def consumer_visit(
        self,
        tRS_rD: cute.Tensor,
        shape_mnk: cute.Shape,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> EpilogueTensorsLoop:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors_loop.list_))
        epi_tensors_loop_new_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_tensors_loop_new = self.evts[i].consumer_visit(
                tRS_rD=tRS_rD,
                shape_mnk=shape_mnk,
                epi_params=epi_params.list_[i],
                epi_tensors_loop=epi_tensors_loop.list_[i],
            )
            epi_tensors_loop_new_list.append(epi_tensors_loop_new)

        return self.EpilogueTensorsLoop(list_=epi_tensors_loop_new_list)

    @cute.jit
    def consumer_smem_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors_loop.list_))
        for i in cutlass.range_constexpr(len(self.evts)):
            self.evts[i].consumer_smem_store(
                epi_coord=epi_coord,
                epi_buffer=epi_buffer,
                epi_params=epi_params.list_[i],
                epi_tensors_loop=epi_tensors_loop.list_[i],
            )

    @cute.jit
    def consumer_tma_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors_loop.list_))
        for i in cutlass.range_constexpr(len(self.evts)):
            self.evts[i].consumer_tma_store(
                epi_coord=epi_coord,
                epi_buffer=epi_buffer,
                epi_params=epi_params.list_[i],
                epi_tensors_loop=epi_tensors_loop.list_[i],
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
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_pipelines.list_))
        epi_pipelines_new_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_pipelines_new = self.evts[i].producer_begin(
                is_tma_warp=is_tma_warp,
                epi_load_stage=epi_load_stage,
                epi_tile_num=epi_tile_num,
                epi_tile_layout=epi_tile_layout,
                epi_params=epi_params.list_[i],
                epi_tensors=epi_tensors.list_[i],
                epi_pipelines=epi_pipelines.list_[i],
            )
            epi_pipelines_new_list.append(epi_pipelines_new)

        return self.EpiloguePipelines(list_=epi_pipelines_new_list)

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
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_tensors.list_))
        misc_utils.static_assert(len(self.evts) == len(epi_pipelines.list_))
        epi_pipelines_new_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            epi_pipelines_new = self.evts[i].producer_tma_load(
                is_tma_warp=is_tma_warp,
                epi_idx=epi_idx,
                epi_load_stage=epi_load_stage,
                epi_tile_num=epi_tile_num,
                epi_tile_layout=epi_tile_layout,
                epi_params=epi_params.list_[i],
                epi_tensors=epi_tensors.list_[i],
                epi_pipelines=epi_pipelines.list_[i],
            )
            epi_pipelines_new_list.append(epi_pipelines_new)

        return self.EpiloguePipelines(list_=epi_pipelines_new_list)

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        smem_struct_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            smem_struct = self.evts[i].get_smem_struct(
                epi_load_stage=epi_load_stage,
                epi_num_threads=epi_num_threads,
                epi_params=epi_params.list_[i],
            )
            smem_struct_list.append(smem_struct)

        class SharedStorage(EpilogueSharedStorage):
            pass

        for i, smem_struct in enumerate(smem_struct_list):
            SharedStorage.__annotations__[f"_evt_{i}"] = smem_struct

        SharedStorage = cute.struct(SharedStorage)
        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:
        misc_utils.static_assert(len(self.evts) == len(epi_params.list_))
        epi_tensors_smem_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            evt_storage = getattr(storage, f"_evt_{i}")
            epi_tensors_smem = self.evts[i].get_smem_tensors(
                storage=evt_storage,
                epi_num_threads=epi_num_threads,
                epi_params=epi_params.list_[i],
            )
            epi_tensors_smem_list.append(epi_tensors_smem)

        return self.EpilogueTensorsSMem(list_=epi_tensors_smem_list)

    @cute.jit
    def get_smem_bytes_per_stage(
        self,
        epi_tile: cute.Tile,
        epi_num_threads: int,
        epi_args: EpilogueArguments,
    ) -> tuple[int, int, int]:
        misc_utils.static_assert(len(self.evts) == len(epi_args.list_))
        epi_smem_bytes_fixed_list = []
        epi_smem_bytes_per_stage_cst_list = []
        epi_smem_bytes_per_stage_pld_list = []
        for i in cutlass.range_constexpr(len(self.evts)):
            (
                epi_smem_bytes_fixed,
                epi_smem_bytes_per_stage_cst,
                epi_smem_bytes_per_stage_pld,
            ) = self.evts[i].get_smem_bytes_per_stage(
                epi_tile=epi_tile,
                epi_num_threads=epi_num_threads,
                epi_args=epi_args.list_[i],
            )
            epi_smem_bytes_fixed_list.append(epi_smem_bytes_fixed)
            epi_smem_bytes_per_stage_cst_list.append(epi_smem_bytes_per_stage_cst)
            epi_smem_bytes_per_stage_pld_list.append(epi_smem_bytes_per_stage_pld)

        return (
            sum(epi_smem_bytes_fixed_list),
            sum(epi_smem_bytes_per_stage_cst_list),
            sum(epi_smem_bytes_per_stage_pld_list),
        )
