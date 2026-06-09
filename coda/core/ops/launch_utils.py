import cutlass
from cuda.bindings import driver


def launch_check(kernel: cutlass.cutlass_dsl.KernelLauncher) -> None:
    info = cutlass.utils.HardwareInfo()
    max_shared_memory_per_block = info._checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            info.device,
        )
    )
    smem_usage = kernel.smem_usage()
    if smem_usage > max_shared_memory_per_block:
        raise ValueError(
            f"Insufficient shared memory for kernel launch:\n"
            f"  Required:  {smem_usage:,} bytes ({smem_usage / 1024:.2f} KB)\n"
            f"  Available: {max_shared_memory_per_block:,} bytes ({max_shared_memory_per_block / 1024:.2f} KB)\n"
            f"  Deficit:   {smem_usage - max_shared_memory_per_block:,} bytes\n"
            f"Consider reducing shared memory allocations or adjusting kernel configuration."
        )
