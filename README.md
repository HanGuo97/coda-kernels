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
pip install coda-kernels
```

Or from source:

```bash
git clone https://github.com/open-lm-engine/coda-kernels.git
cd coda-kernels
pip install -e .
```


## Functional API

`coda.kernels.functional` exposes fully fused, autograd-complete operators: each forward runs as one GEMM with a fused epilogue (plus at most one elementwise pass), and each backward is built from the same fused kernels.

### `linear_swiglu`

Fused linear projection + SwiGLU: `silu(gate) * up`, where `gate || up = x @ weight.T`.

```python
import torch
from coda.kernels.functional.swiglu import linear_swiglu

x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)

out = linear_swiglu(x, weight)  # (4096, 4096)
out.sum().backward()
```

### `linear_cross_entropy`

Fused linear projection + cross-entropy loss; the `(M, V)` logits are never materialized.

```python
import torch
from coda.kernels.functional.cross_entropy import linear_cross_entropy

x = torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(131072, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
target = torch.randint(0, 131072, (8192,), device="cuda", dtype=torch.int32)

loss = linear_cross_entropy(x, weight, target, ignore_index=-100, reduction="mean")
loss.backward()
```

`linear_cross_entropy_forward` is the gradient-free variant for evaluation.

### `linear_qknorm_rope`

Fused QKV projection + per-head QK RMSNorm + RoPE. Returns `(q, k, v)` as views of the projection -- V passes through untouched and no copies are made.

```python
import torch
from coda.kernels.functional.qknorm_rope import linear_qknorm_rope

M, K, head_dim = 8192, 2048, 128
num_heads_q, num_heads_k = 8, 8
num_heads = num_heads_q + 2 * num_heads_k  # Q + K + V heads

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(num_heads * head_dim, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
gamma = torch.ones(head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
positions = torch.arange(M, device="cuda", dtype=torch.int32)
inv_freq = 10000.0 ** (-torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim)
frequencies = inv_freq.repeat(num_heads_q + num_heads_k)  # one frequency per rotation pair

q, k, v = linear_qknorm_rope(x, weight, gamma, positions, frequencies, num_heads_q, num_heads_k, head_dim, 1e-6)
# q: (M, num_heads_q * head_dim), k: (M, num_heads_k * head_dim), v: (M, num_heads_k * head_dim)
(q.sum() + k.sum() + v.sum()).backward()
```
