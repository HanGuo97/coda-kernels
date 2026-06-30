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

#### `linear_swiglu(x, weight)`

Fused linear projection followed by a SwiGLU activation. Computes `swiglu(x @ weight.T)`: the projection yields a `gate||up` pre-activation, which SwiGLU collapses to half the width.

| Argument | Shape | Description |
|----------|-------|-------------|
| `x` | `(M, K)` | Input activations. |
| `weight` | `(N, K)` | Gate+up projection weight (`out_features, in_features`). `N` must be even. |

**Returns:** a `(M, N // 2)` tensor.

**Backward:** the forward saves `x`, `weight`, and the `gate||up` pre-activation. Given the output gradient, the SwiGLU derivative recovers the pre-activation gradient `grad_pre` of shape `(M, N)`, then two GEMMs produce `dx = grad_pre @ weight` and `dweight = grad_pre.T @ x`. Saving the pre-activation rather than the activation lets SwiGLU be recomputed locally instead of stored.

```python
import torch
from coda.kernels.functional.swiglu import linear_swiglu

x      = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda", requires_grad=True)
weight = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda", requires_grad=True)

out = linear_swiglu(x, weight)   # (4096, 4096)
out.sum().backward()             # populates x.grad and weight.grad
```
