import torch
import cutlass
import cutlass.cute.testing as testing
from cuda.bindings import driver
from typing import Callable


class ExtendedHardwareInfo(cutlass.utils.HardwareInfo):

    def get_memory_clock_rate(self) -> int:
        # Query memory clock rate (in kHz)
        return self._checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                self.device,
            )
        )

    def get_global_memory_bus_width(self) -> int:
        # Query bus width (in bits)
        return self._checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                self.device,
            )
        )

    def get_peak_memory_bandwidth_in_bytes(self) -> int:
        # Peak bandwidth in GB/s = (mem_clock_rate_khz * mem_bus_width * 2) / 8
        return (
            self.get_memory_clock_rate() *
            self.get_global_memory_bus_width() *
            1000. *  # kHz to Hz
            2. /     # double data rate
            8.       # bits to bytes
        )


def assert_throughput_efficiency(
    avg_time_us: float,
    total_bytes: int,
    throughput: float,
) -> None:
    # Calculate memory throughput
    avg_time_s = avg_time_us / 1e6
    throughput_gb_s = (total_bytes / 1e9) / avg_time_s

    # Get theoretical peak memory bandwidth from device
    hw_info = ExtendedHardwareInfo()
    peak_bandwidth_gb_s = hw_info.get_peak_memory_bandwidth_in_bytes() / 1e9

    # Assert throughput efficiency is >= target
    efficiency = throughput_gb_s / peak_bandwidth_gb_s
    message = (
        f"Memory throughput efficiency {efficiency * 100.:.2f}% is below {throughput * 100.:.2f}%. "
        f"Achieved {throughput_gb_s:.2f} GB/s out of {peak_bandwidth_gb_s:.2f} GB/s peak "
        f"(Memory clock: {hw_info.get_memory_clock_rate()/1e6:.2f} GHz, "
        f"Bus width: {hw_info.get_global_memory_bus_width()} bits)."
    )
    assert efficiency >= throughput, message


def check_memory_throughput(
    fn: Callable,
    fn_ref: Callable,
    workspace_generator: Callable[[], testing.JitArguments],
    throughput: float,
    seed: int = 0,
    iterations: int = 100,
    warmup_iterations: int = 2,
) -> None:
    torch.random.manual_seed(seed)

    # Generate reference inputs
    workspace = workspace_generator()
    outputs = fn_ref(*workspace.args, **workspace.kwargs)
    if isinstance(outputs, torch.Tensor):
        total_bytes = (
            outputs.numel() *
            outputs.element_size()
        )
    else:
        total_bytes = 0

    for workspace_arg in workspace.args:
        if isinstance(workspace_arg, torch.Tensor):
            total_bytes = (
                total_bytes +
                workspace_arg.numel() *
                workspace_arg.element_size()
            )
    for (_, workspace_kwarg) in workspace.kwargs.items():
        if isinstance(workspace_kwarg, torch.Tensor):
            total_bytes = (
                total_bytes +
                workspace_kwarg.numel() *
                workspace_kwarg.element_size()
            )

    workspace_count = testing.get_workspace_count(
        one_workspace_bytes=total_bytes,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    avg_time_us = testing.benchmark(
        fn,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    assert_throughput_efficiency(
        avg_time_us=avg_time_us,
        total_bytes=total_bytes,
        throughput=throughput,
    )


DEFAULT_LAUNCH_EVENTS = (
    "cuLaunchKernel",
    "cudaLaunchKernel",
    "cuLaunchKernelEx",
    "cudaLaunchKernelExC",
)


def count_cuda_launch_calls(
    fn: Callable,
    *args: object,
    warmup: int = 1,
    synchronize: bool = True,
    **kwargs: object,
) -> tuple[int, dict[str, int]]:

    for _ in range(warmup):
        fn(*args, **kwargs)

    if synchronize:
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        fn(*args, **kwargs)
        if synchronize:
            torch.cuda.synchronize()

    counts = {name: 0 for name in DEFAULT_LAUNCH_EVENTS}
    for event in prof.events():
        key = getattr(event, "key", None)
        name = getattr(event, "name", None)
        if any([
            name != key,
            name is None,
            callable(name),
        ]):
            raise NotImplementedError
        if name in counts.keys():
            counts[name] = counts[name] + 1
        else:
            assert "launch" not in name.lower()

    total = sum(counts.values())
    return total, counts
