"""GPU memory statistics through NVML."""

import pynvml


def memory_used():
    """Return the bytes of GPU memory in use on device 0, or None without NVML."""
    if pynvml is None:
        return None
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()
