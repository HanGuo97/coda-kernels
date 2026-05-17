# CODA

<p align="center">
  <img src="figs/icon.jpg" width="200" />
</p>

**CODA** is a GPU kernel abstraction that expresses memory-bound Transformer computations as GEMM-plus-epilogue programs, eliminating intermediate global memory traffic by fusing surrounding operators into the GEMM tile while it remains on chip.

## The Problem

Transformer training is dominated by matrix multiplications, but a significant fraction of wall-clock time is spent in the surrounding operators: RMS norm, activations, residual adds, reductions, RoPE. These kernels are individually cheap but collectively expensive because each one reads and writes large tensors through global memory. In an optimized training stack, this data movement becomes the bottleneck.

## The Approach

CODA reparameterizes these operators as **GEMM epilogues** — computations that run while the GEMM output tile is still in registers and shared memory, before it is written back to global memory. The GEMM mainloop is fixed; the surrounding work is expressed using a small set of composable primitives:

- **Scaling** (e.g., RMS norm scale factors)
- **Reductions** (block-wise row/column aggregations)
- **Pairwise transformations** (e.g., SwiGLU gating over two streams)
- **Accumulation** (residual add, bias, cross-entropy)

Composing these covers nearly all non-attention computation in a standard Transformer block, in both forward and backward passes.

## Naming

The project was originally named **Rapier** (because it is built on CUTLASS). The name was later changed to **CODA**. The infrastructure library in `rapier/` retains its original name.

## Repository Structure

```
coda-kernels/
├── models/          # High-level API
│   ├── ops.py       # CODA layer implementations (forward + backward)
│   └── ops2.py      # Corresponding implementations in PyTorch
├── kernels/
│   ├── gens/        # LLM-authored CuTeDSL kernel implementations
│   ├── refs/        # PyTorch reference implementations
│   ├── tests/
│   └── benchmarks/
└── rapier/          # GEMM-plus-epilogue kernel infrastructure
    ├── gemm/        # WGMMA GEMM kernels and PyTorch wrapper
    ├── epilogue/    # Composable epilogue visitors
    ├── ops/         # Low-level utilities
    ├── examples/    # Standalone usage examples
    └── docs/        # Docs for LLM (LLM-generated, somewhat deprecated)
```

### `models/ops.py` — fused layer ops

The three ops together cover the full Transformer block (excluding attention):

| Op | Operations fused into one kernel |
|----|----------------------------------|
| `layer_pre` | Embedding → RMS norm → QKV projection → RoPE |
| `layer` | Attn out-proj → residual add → RMS norm → SwiGLU gate+up → RoPE |
| `layer_post` | MLP down-proj → residual add → RMS norm → output GEMM → cross-entropy |

### `kernels/`

`gens/` contains LLM-authored CuTeDSL kernels (one file per op, with individual epilogue visitors in `gens/epilogue/`). `refs/` contains the corresponding PyTorch reference implementations.

### `rapier/` — the kernel infrastructure

Rapier is the library that implements the GEMM-plus-epilogue abstraction on top of [CUTLASS CuTeDSL](https://github.com/NVIDIA/cutlass), targeting NVIDIA Hopper (H100) GPUs.

**GEMM backends (`rapier/gemm/`)**

| Module | Description |
|--------|-------------|
| `gemm_quack` | Persistent warp-specialized WGMMA kernel with ping-pong buffering |
| `gemm_interface` | PyTorch wrapper: compilation caching, layout management, autotuning |

**Epilogue visitors (`rapier/epilogue/`)**

| Module | Description |
|--------|-------------|
| `base` | Abstract visitor interface defining the full lifecycle |
| `bias` | Row/column bias addition |
| `reduction` | Block-level row/column reductions (store, store-2X, load variants) |
| `activation` | Dual-output activations: elementwise, pairwise, contraction, expansion |
| `matrix` | TMA-pipelined matrix load with residual-add; 2X paired-tile variant |
| `cross_entropy` | Online softmax + target logit selection, fused into the output tile |
| `composite` | Chains multiple visitors into a single unified epilogue |

**Utilities (`rapier/ops/`)** — tensor allocation in register/shared memory, TMA descriptor creation, hierarchical reductions, dtype conversion, layout construction, pipeline state management, profiling, and benchmarking.
