import pynvml
import logging
def get_gpu_stats(device_index=0):
    """
    Returns current GPU stats:
    - total_memory_MB: Total memory in MB
    - free_memory_MB: Free memory in MB
    - used_memory_MB: Used memory in MB
    - power_usage_W: Current power draw in watts (if supported)
    - utilization_gpu_percent: GPU utilization percentage

    Parameters
    ----------
    device_index : int
        GPU index (default: 0)

    Returns
    -------
    dict
        Dictionary of GPU stats
    """
    pynvml.nvmlInit()

    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory_MB = mem_info.total / 1024 ** 2
    free_memory_MB = mem_info.free / 1024 ** 2
    used_memory_MB = mem_info.used / 1024 ** 2

    power_usage_W = None
    try:
        power_usage_W = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # milliwatts → watts
    except pynvml.NVMLError:
        power_usage_W = -1  # Not supported on all GPUs

    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    utilization_gpu_percent = utilization.gpu

    pynvml.nvmlShutdown()

    return {
        "total_memory_MB": round(total_memory_MB, 2),
        "free_memory_MB": round(free_memory_MB, 2),
        "used_memory_MB": round(used_memory_MB, 2),
        "power_usage_W": round(power_usage_W, 2) if power_usage_W != -1 else None,
        "utilization_gpu_percent": utilization_gpu_percent,
    }


def log_gpu_status(prefix="[GPU STATUS]"):
    """
    Logs GPU status using `logging.debug2_status` convention.
    """
    stats = get_gpu_stats()

    msg = (
        f"{prefix} Total={stats['total_memory_MB']}MB | "
        f"Free={stats['free_memory_MB']}MB | "
        f"Used={stats['used_memory_MB']}MB | "
        f"Power={stats['power_usage_W']}W | "
        f"Load={stats['utilization_gpu_percent']}%"
    )

    logging.debug2_status(msg, overwrite=True)