# Copyright (c) 2025, Tri Dao.
# https://github.com/Dao-AILab/quack/blob/main/quack/pipeline.py

import cutlass
import cutlass.cute as cute


@cute.jit
def advance_n(
    state: cutlass.pipeline.PipelineState,
    num_iterations: cute.Int32,
) -> cutlass.pipeline.PipelineState:
    # https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/pipeline/sm90_pipeline.hpp
    state._count += num_iterations
    new_index = state._index + num_iterations

    # Number of iterations cross over the stage boundary => flipped phase
    if ((num_iterations < state.stages) and new_index >= state.stages):
        state._phase ^= 1

    # How many times number of iterations cross over the stage boundary and
    # end up on a odd number => flipped phase
    if ((num_iterations >= state.stages) and ((new_index // state.stages) % 2) == 1):
        state._phase ^= 1

    state._index = new_index % state.stages
    return state
