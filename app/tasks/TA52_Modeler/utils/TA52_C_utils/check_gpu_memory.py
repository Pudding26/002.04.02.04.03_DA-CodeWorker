import pynvml
pynvml.nvmlInit()

def check_gpu_memory(min_free_mb=500):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_mb = meminfo.free / 1024**2
    if free_mb < min_free_mb:
        raise MemoryError(f"Low GPU memory: only {free_mb:.1f} MB free")