# NVIDIA Compute Sanitizer for GPU Kernels

Detect memory errors, race conditions, and synchronization issues in CUDA kernels.

---

## Basic Setup

No special setup required. Write a standard Python script that launches your kernel:

```python
import torch

if __name__ == "__main__":
    x = torch.randn(1024, device='cuda')
    result = kernel(x, ...)
    torch.cuda.synchronize()  # Catch async errors at exact call site
```

### Compile with Line Info

Enable line-level debugging information in generated kernels for precise error localization:

```python
kernel = cute.compile(..., options="--generate-line-info")
```

**Benefits:**
- Compute Sanitizer reports exact line numbers in generated CUDA code
- Pinpoint errors to specific operations rather than entire kernel ranges
- Essential for debugging complex memory access patterns

---

## Available Tools

Four specialized tools for different error classes:

```bash
# Memory errors: out-of-bounds, invalid addresses, misalignment
compute-sanitizer --tool memcheck python script.py

# Race conditions: shared/global memory races
compute-sanitizer --tool racecheck python script.py

# Uninitialized memory reads
compute-sanitizer --tool initcheck python script.py

# Synchronization errors: illegal barriers, deadlocks
compute-sanitizer --tool synccheck python script.py
```

### Useful Options

```bash
--device <id>         # Target specific GPU
--print-limit <n>     # Limit error output (default: 1000)
--save <file>         # Save report
--log-file <file>     # Full log with pass/fail summary
```

**Example:**
```bash
compute-sanitizer --tool memcheck --device 1 --print-limit 10 python script.py
```

---

## Tool Selection

| Tool | Detects | Overhead | Use When |
|------|---------|----------|----------|
| `memcheck` | Out-of-bounds access, invalid addresses, alignment | Low | First line of defense |
| `synccheck` | Illegal barriers, deadlocks | Low | Kernel hangs |
| `initcheck` | Uninitialized memory reads | Medium | Undefined behavior |
| `racecheck` | Shared/global memory races | High | Non-deterministic results |

**Recommended workflow:** Start with `memcheck`, then `synccheck`. Use `initcheck` and `racecheck` only when investigating specific issues.
