# Profiling CuTeDSL Kernels with NVIDIA Nsight Compute

A practical guide to profiling GPU kernels using NCU.

---

## Structuring Code for Profiling

Structure your profiling script with three phases: setup, warmup, and profile.

```python
import torch

if __name__ == "__main__":
    # 1. Setup: Create tensors and kernel instance
    fn = lambda: kernel(...)

    # 2. Warmup: Run without profiling
    for _ in range(num_warmup):
        fn()

    # 3. Profile: Enable profiler, run, disable profiler
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_profile):  # always set `num_profile = 1`
        fn()
    torch.cuda.cudart().cudaProfilerStop()
```

**Important:**
- Profile one kernel per run (multiple kernels will break CSV parsing)
- Remove all print statements from your script (stdout must be clean for parser)

---

## Running NCU

Use a dedicated device to avoid interference from other workloads (set via `--device`).

```bash
# Details mode (JSON metrics output with full section data)
python -m coda.core.ops.profiling_utils script.py --mode details --device 1

# Source mode (SASS-level analysis with stall reasons)
python -m coda.core.ops.profiling_utils script.py --mode source --device 1
```

**Options:**
- `--mode`: Profiling mode - `details` (full metrics, default) or `source` (SASS analysis with stall sampling)
- `--device`: CUDA device ID (default: 0)
- `--output`: Output file name (default: out)
- `--ncu-executable`: Path to ncu executable (default: ncu)
