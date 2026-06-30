# CODA: GPU Kernels as GEMM-plus-Epilogue Programs

<p align="center">
  <img src="figs/icon.jpg" width="350" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19269"><img src="https://img.shields.io/badge/arXiv-2605.19269-b31b1b.svg" alt="arXiv"></a>
</p>

**CODA** is a GPU kernel abstraction that expresses Transformer operators as GEMM-plus-epilogue programs, fusing normalization, activations, residual updates, and reductions into the GEMM output tile before it is written to global memory, combining framework-level productivity with hardware-level efficiency. CODA is built on [CUTLASS CuTeDSL](https://github.com/NVIDIA/cutlass) and targets NVIDIA Hopper (H100) GPUs.

<p align="center">
  <img src="figs/reparameterization.png" width="700" />
</p>


## Updates
- June 23, 2026. We are restructuring CODA. For legacy version, please check `v1` tag.

## Installation

```bash
git clone https://github.com/HanGuo97/coda-kernels.git
cd coda-kernels
pip install -e .
```


## Quick Start

> [!NOTE]
> We autotune each kernel the first time it sees a new input configuration (shape, dtype, etc.), so the initial call may take a while.


### Functional level

`coda/kernels/functional/` exposes the fused kernels as differentiable `torch.autograd.Function`s with hand-written backward passes.

#### `linear_swiglu`

Fused linear projection and SwiGLU activation: `swiglu(x @ weight.T)`, where the projection produces a `gate || up` pre-activation and `swiglu(gate || up) = silu(gate) * up`.

| Argument | Shape | Description |
|----------|-------|-------------|
| `x` | `(M, K)` | Input activations. |
| `weight` | `(N, K)` | Gate+up projection weight (`out_features, in_features`); `N` must be even. |

**Returns** `(M, N // 2)` — the SwiGLU output. Differentiable in both `x` and `weight`.
