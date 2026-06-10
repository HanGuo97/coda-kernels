import torch
import warnings

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda

from coda.core.ops import layout_utils
from coda.core.ops import memory_utils
from coda.core.ops import creation_utils
from coda.core.ops import reduction_utils
from coda.core.ops.misc_utils import static_assert
from coda.core.ops.debug_utils import isnan, check_nan
from coda.core.ops.launch_utils import launch_check

ALLOWED_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@cute.kernel
def nan_debug_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    vector_size: cutlass.Constexpr,
    use_fix: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()

    idX = cute.make_identity_tensor(mX.shape)
    gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
    gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
    cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

    def _tiled_copy(
        src: cute.Tensor,
        dst: cute.Tensor | str,
    ) -> cute.Tensor:
        copy_config = memory_utils.MemoryCopyConfig(
            op="universal",
            dtype=src.element_type,
            num_bits_per_copy=src.element_type.width * vector_size,
            tiler_mn=tiler_mn,
            layout_tv=tv_layout,
        )

        # We assume that all copy operations will use the same
        # number of elements as X and hence the same predicate
        crd = cX
        shape = mX.shape

        copy_outputs = memory_utils.copy(
            src=src,
            dst=dst,
            crd=crd,
            shape=shape,
            config=copy_config,
            thread_index=tidx,
            smem_allocator=allocator,
        )
        return copy_outputs.dst_thread

    tXrX = _tiled_copy(
        src=gX,
        dst="rmem",
    )
    tYrY = creation_utils.allocate_tensor_like(
        tensor=tXrX,
        memspace="rmem",
        smem_allocator=allocator,
        dtype=mY.element_type,
    )
    x = tXrX.load().to(cute.Float32)
    o = creation_utils.ones_like(x)

    thread_shape = (
        tv_layout.shape[0][1],  # threads_per_col
        tv_layout.shape[0][0],  # threads_per_row
    )

    def _combine_fn(
        lhs: tuple[cute.Numeric, cute.Numeric],
        rhs: tuple[cute.Numeric, cute.Numeric],
    ) -> tuple[cute.Numeric, cute.Numeric]:
        """Combine two (max, scaled_sum) pairs.

        BUG: When both max_lhs and max_rhs are -inf, we get:
            new_max = max(-inf, -inf) = -inf
            scale_lhs = exp((-inf) - (-inf)) = exp(NaN) = NaN

        This happens when both partial results are empty (no valid data).
        The arithmetic (-inf) - (-inf) is undefined and produces NaN.
        """
        max_lhs, sum_lhs = lhs
        max_rhs, sum_rhs = rhs
        new_max = cute.arch.fmax(max_lhs, max_rhs)
        # Direct computation could cause NaN when both inputs are -inf
        scale_lhs = cute.math.exp(max_lhs - new_max)
        scale_rhs = cute.math.exp(max_rhs - new_max)

        if cutlass.const_expr(use_fix):
            # FIX: Check if new_max is finite before computing scaling factors
            # When new_max = -inf (both inputs empty), use identity scale factor
            has_nan_scale_lhs = isnan(scale_lhs)
            has_nan_scale_rhs = isnan(scale_rhs)
            static_assert(isinstance(has_nan_scale_lhs, cute.Boolean))
            static_assert(isinstance(has_nan_scale_rhs, cute.Boolean))
            static_assert(isinstance(max_lhs, cute.Float32 | float))
            static_assert(isinstance(max_rhs, cute.Float32 | float))
            if has_nan_scale_lhs:
                scale_lhs = cute.Float32(1.0)
            if has_nan_scale_rhs:
                scale_rhs = cute.Float32(1.0)
        else:
            # Use check_nan to detect NaN during computation.
            # When device assertions are enabled, this will abort the process
            # if NaN is detected. Otherwise, it behaves as a no-op.
            check_nan(scale_lhs, error=True)
            check_nan(scale_rhs, error=True)

        new_sum = sum_lhs * scale_lhs + sum_rhs * scale_rhs
        return (new_max, new_sum)

    # Initialize reduction with -inf for max, 0.0 for sum
    init_value = (
        -x.element_type.inf,
        x.element_type.zero,
    )

    op = reduction_utils.BlockReductionOp(
        combine_fn=_combine_fn,
        reduce_ssa=None,
        reduce_wrp=None,
        init_value=init_value,
    )

    (_, x_reduced), _ = reduction_utils.reduce(
        (x, o),
        op=op,
        thread_shape=thread_shape,
        smem_allocator=allocator,
        reduction_buffer=None,
    )

    y = creation_utils.full_like(x, fill_value=x_reduced)
    tYrY.store(y.to(tYrY.element_type))

    # Copy the results back
    _tiled_copy(
        src=tYrY,
        dst=gY,
    )


@cute.jit
def nan_debug(
    mX: cute.Tensor,
    mY: cute.Tensor,
    stream: cuda.CUstream,
    use_fix: cutlass.Constexpr,
) -> None:
    vector_size = cutlass.const_expr(
        128 //
        cutlass.max(
            mX.element_type.width,
            mY.element_type.width,
        )
    )

    tiler_mn, tv_layout = layout_utils.make_2D_row_reduction_layout(
        size=mX.shape[1],
        dtype=mX.element_type,
        num_threads=None,
        cluster_size=1,
        num_bits_per_copy=(
            vector_size *
            mX.element_type.width
        ),
    )

    num_blocks = cute.ceil_div(mX.shape[0], tiler_mn[0])
    num_threads = cute.size(tv_layout, mode=[0])

    kernel = nan_debug_kernel(
        mX,
        mY,
        tiler_mn,
        tv_layout,
        vector_size,
        use_fix,
    )
    kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1],
        cluster=None,
        smem=None,
        stream=stream,
    )
    launch_check(kernel)


def to_cute_tensor(torch_tensor: torch.Tensor, assumed_align: int = 16, dynamic: bool = True) -> cute.Tensor:
    torch_tensor = torch_tensor.detach()
    cute_tensor = cute.runtime.from_dlpack(
        torch_tensor,
        assumed_align=assumed_align,
    )
    if dynamic:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=0,
            stride_order=torch_tensor.dim_order(),
        )

    return cute_tensor


def _nan_debug_op(
    x: torch.Tensor,
    y: torch.Tensor,
    use_fix: bool,
    device_assert: bool,
) -> None:
    if x.dim() != 2:
        raise ValueError("Input must be 2D")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    if x.dtype not in ALLOWED_DTYPES:
        raise ValueError("Unsupported dtype")

    _, N = x.shape
    x_tensor = to_cute_tensor(x)
    y_tensor = to_cute_tensor(y)

    current_stream = cutlass_torch.current_stream()
    compile_key = (
        N,
        x.dtype,
        y.dtype,
        use_fix,
        device_assert,
    )
    if compile_key not in _nan_debug_op.compile_cache:
        warnings.warn(
            f"Kernel compilation cache miss for N={N}, x.dtype={x.dtype}, "
            f"y.dtype={y.dtype}, use_fix={use_fix}. This will trigger JIT compilation.",
        )
        # NOTE: Device assertions must be enabled for `check_nan` to detect and abort on NaN
        if device_assert:
            warnings.warn(
                "Enabling device-side assertions for debugging. "
                "Performance will be significantly degraded."
            )
            compile_options = {
                "options": "--enable-device-assertions",
            }
        else:
            compile_options = {}

        _nan_debug_op.compile_cache[compile_key] = cute.compile(
            nan_debug,
            x_tensor,
            y_tensor,
            current_stream,
            use_fix,
            **compile_options,
        )

    _nan_debug_op.compile_cache[compile_key](
        x_tensor,
        y_tensor,
        current_stream,
    )


_nan_debug_op.compile_cache = {}


def nan_debug_op(
    x: torch.Tensor,
    use_fix: bool,
    device_assert: bool,
) -> torch.Tensor:
    y = torch.empty_like(x)
    _nan_debug_op(
        x=x,
        y=y,
        use_fix=use_fix,
        device_assert=device_assert,
    )
    return y


if __name__ == "__main__":
    print("=" * 70)
    print("NaN Debugging Example")
    print("=" * 70)
    print("\nThis example demonstrates how NaN can occur during reductions when")
    print("combining values with -inf, and how to detect and fix such issues.")
    print()

    M, N = 128, 4096
    dtype = torch.float16
    device = "cuda"

    print(f"Testing with M={M}, N={N}, dtype={dtype}\n")

    torch.random.manual_seed(0)
    x = 0.1 * torch.randn(M, N, device=device, dtype=dtype)

    # Test 1: Buggy version (should produce NaN)
    print("1. Running BUGGY version (without fix)...")
    print("   Computing normalized sum Σ exp(x) with max-based scaling.")
    print("   Bug: exp((-inf) - (-inf)) = exp(NaN) when combining empty partials.\n")
    try:
        out_buggy = nan_debug_op(x, use_fix=False)
        has_nan = torch.isnan(out_buggy).any().item()
        has_inf = torch.isinf(out_buggy).any().item()
        print(f"   Result: Has NaN: {has_nan}, Has Inf: {has_inf}")

        if has_nan:
            print("   ⚠️  NaN detected! The check_nan() calls in the kernel would help")
            print("       pinpoint where the NaN originated.")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Test 2: Fixed version
    print("2. Running FIXED version (with isfinite check)...")
    print("   This version checks if new_max is finite before computing exp(max - max),")
    print("   avoiding the (-inf) - (-inf) = NaN issue.\n")
    try:
        out_fixed = nan_debug_op(x, use_fix=True)
        has_nan = torch.isnan(out_fixed).any().item()
        has_inf = torch.isinf(out_fixed).any().item()
        print(f"   Result: Has NaN: {has_nan}, Has Inf: {has_inf}")

        if not has_nan:
            print("   ✅ Success! No NaN values in output.")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()
    print("=" * 70)
    print("Key Takeaway:")
    print("=" * 70)
    print("When writing custom combine functions for reductions, be careful with")
    print("special values like ±inf. Operations like (-inf) - (-inf) produce NaN.")
    print()
    print("Use check_nan() from rapier.ops.debug_utils during development to catch")
    print("these issues early and identify exactly where NaN values originate.")
    print("=" * 70)
