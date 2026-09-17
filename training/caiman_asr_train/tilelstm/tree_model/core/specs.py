from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from beartype import beartype


@beartype
@dataclass(frozen=True)
class GpuSpec:
    gpu: str  # device name as torch reports it
    arch_sm: int  # 89 / 80 / 120 -- numeric arch (major*10+minor)
    gemm_flops: float  # tensor-core peak, fp16 in / fp32 accumulate (FLOP/s)
    cuda_flops: float  # CUDA-core fp32 peak (FLOP/s)
    mem_bw: float  # global-memory bandwidth (bytes/s)
    sms: int  # streaming multiprocessors
    l2_bytes: float  # L2 capacity (bytes)


# Measure real throughput peaks by timing a big GEMM / copy, not a spec sheet.


@beartype
def _bench_ms(fn: Callable, warmup: int = 10, rep: int = 50) -> float:
    """
    Median ms of ``fn()`` via CUDA events (a fresh launch each rep, no graph capture).
    """
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rep):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


@beartype
def _measured_gemm_flops(device, n: int = 8192) -> float:
    """
    fp16-in / fp32-accumulate tensor-core GEMM ceiling (FLOP/s) from an n×n matmul.
    """
    import torch

    a = torch.randn(n, n, device=device, dtype=torch.float16)
    b = torch.randn(n, n, device=device, dtype=torch.float16)
    with torch.inference_mode():
        ms = _bench_ms(lambda: a @ b)
    return (2 * n**3) / (ms * 1e-3)


@beartype
def _measured_cuda_flops(device, n: int = 8192) -> float:
    """
    fp32 (TF32 disabled) GEMM ceiling (FLOP/s) -- a CUDA-core fp32 throughput proxy.
    """
    import torch

    mm = torch.backends.cuda.matmul
    prev, mm.fp32_precision = mm.fp32_precision, "ieee"
    try:
        a = torch.randn(n, n, device=device, dtype=torch.float32)
        b = torch.randn(n, n, device=device, dtype=torch.float32)
        with torch.inference_mode():
            ms = _bench_ms(lambda: a @ b)
    finally:
        mm.fp32_precision = prev
    return (2 * n**3) / (ms * 1e-3)


@beartype
def _measured_mem_bw(device, mib: int = 256) -> float:
    """
    Memory-bandwidth ceiling (bytes/s) from a large device-to-device copy.
    """
    import torch

    n = mib * 1024 * 1024 // 2  # fp16 elements
    src = torch.randn(n, device=device, dtype=torch.float16)
    dst = torch.empty_like(src)
    with torch.inference_mode():
        ms = _bench_ms(lambda: dst.copy_(src))
    return (2 * n * 2) / (ms * 1e-3)  # 2 bytes each, read + write


@cache
@beartype
def host_spec(index: int = 0) -> GpuSpec:
    """
    ``GpuSpec`` for the GPU at ``index``: throughput peaks measured, sms / l2 / arch queried
    from ``torch.cuda``. Requires CUDA. ``@cache``d on ``index`` since the GEMM/copy
    benchmarks are costly. Runs under ``torch.cuda.device(index)`` so it measures that GPU
    even when it is not the current device (e.g. a DDP rank on its own cuda:index).
    """
    import torch

    device = torch.device("cuda", index)
    props = torch.cuda.get_device_properties(device)
    with torch.cuda.device(index):
        gemm_flops = _measured_gemm_flops(device)
        cuda_flops = _measured_cuda_flops(device)
        mem_bw = _measured_mem_bw(device)
    return GpuSpec(
        gpu=torch.cuda.get_device_name(device),
        arch_sm=props.major * 10 + props.minor,
        gemm_flops=gemm_flops,
        cuda_flops=cuda_flops,
        mem_bw=mem_bw,
        sms=props.multi_processor_count,
        l2_bytes=float(props.L2_cache_size),
    )
