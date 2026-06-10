import cutlass
import cutlass.cute as cute
from typing import NamedTuple
from dataclasses import dataclass

from coda.core.ops import struct_utils


class EpilogueSharedStorage(object):
    pass


class EpilogueVisitorTree(object):

    @struct_utils.mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        # Host side
        pass

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueParams(struct_utils.PyTreeDynamicExpression):
        # Device side
        pass

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsSMem(struct_utils.PyTreeDynamicExpression):
        pass

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensors(struct_utils.PyTreeDynamicExpression):
        pass

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpilogueTensorsLoop(struct_utils.PyTreeDynamicExpression):
        pass

    @struct_utils.register_pytree_dataclass
    @dataclass
    class EpiloguePipelines(struct_utils.PyTreeDynamicExpression):
        pass

    @cute.jit
    def to_underlying_arguments(
        self,
        epi_tile: cute.Tile,
        epi_stage: int,
        epi_load_stage: int,
        epi_args: EpilogueArguments,
    ) -> EpilogueParams:
        return self.EpilogueParams()

    @cute.jit
    def prefetch_tma_descriptors(
        self,
        epi_params: EpilogueParams,
    ) -> None:
        pass

    @cute.jit
    def prepare_pipelines(
        self,
        epi_load_stage: int,
        epi_num_warps: int,
        epi_params: EpilogueParams,
        epi_tensors_smem: EpilogueTensorsSMem,
    ) -> EpiloguePipelines:
        return self.EpiloguePipelines()

    @cute.jit
    def advance_pipelines(
        self,
        tile_count: int,
        epi_params: EpilogueParams,
        epi_pipelines: EpiloguePipelines,
    ) -> EpiloguePipelines:
        return self.EpiloguePipelines()

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
        return self.EpilogueTensors()

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
        return self.EpilogueTensorsLoop(), self.EpiloguePipelines()

    @cute.jit
    def consumer_end_loop(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    @cute.jit
    def consumer_visit(
        self,
        tRS_rD: cute.Tensor,
        shape_mnk: cute.Shape,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> EpilogueTensorsLoop:
        return self.EpilogueTensorsLoop()

    @cute.jit
    def consumer_smem_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        pass

    @cute.jit
    def consumer_tma_store(
        self,
        epi_coord: cute.Coord,
        epi_buffer: cute.Int32,
        epi_params: EpilogueParams,
        epi_tensors_loop: EpilogueTensorsLoop,
    ) -> None:
        pass

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
        return self.EpiloguePipelines()

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
        return self.EpiloguePipelines()

    @cute.jit
    def get_smem_struct(
        self,
        epi_load_stage: int,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> type[EpilogueSharedStorage]:

        @cute.struct
        class SharedStorage(EpilogueSharedStorage):
            _placeholder: cute.struct.Align[cute.struct.MemRange[cute.Float32, 0], 16]

        return SharedStorage

    @cute.jit
    def get_smem_tensors(
        self,
        storage: EpilogueSharedStorage,
        epi_num_threads: int,
        epi_params: EpilogueParams,
    ) -> EpilogueTensorsSMem:
        return self.EpilogueTensorsSMem()

    @cute.jit
    def get_smem_bytes_per_stage(
        self,
        epi_tile: cute.Tile,
        epi_num_threads: int,
        epi_args: EpilogueArguments,
    ) -> tuple[int, int, int]:
        return 0, 0, 0


class EVTNoOp(EpilogueVisitorTree):
    pass
