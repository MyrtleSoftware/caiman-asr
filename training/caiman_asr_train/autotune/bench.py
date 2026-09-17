from collections.abc import Callable

import torch
from beartype import beartype


@beartype
def tl_log(label: str, msg: str) -> None:
    """
    Print one autotuner progress line, tagged ``[TileTune/<label>]``. ``label`` is
    the backend name or a selection group ("auto"/"bwd"/...).
    """
    print(f"[TileTune/{label:<6}] {msg}")


_BENCH_SKIP_ERRORS = (
    "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE",  # cooperative grid past residency
    "CUDA_ERROR_INVALID_VALUE",  # smem attribute set fails
    "Failed to launch kernel",  # NVRTC-path launch failure
    "Failed to set max dynamic shared memory",  # NVRTC-path smem failure
    # TVM-FFI execution-backend path (cuda_module.cc): the dynamic SMEM exceeds the
    # per-block opt-in cap. The caller's SMEM filter prunes these pre-compile, so this
    # is a launch-time backstop; the shorter substring also matches "... size to N".
    "Failed to set the allowed dynamic shared memory",
)


@beartype
def n_repeat_for(tune_batch: int) -> int:
    """
    Default lean-bench timed-launch count for ``tune_batch``: ``768*10 // tune_batch``
    clamped to ``[10, 256]``.
    """
    return min(max(768 * 10 // tune_batch, 10), 256)


@beartype
def lean_bench(thunk: Callable, n_warmup: int = 3, n_repeat: int = 10) -> float | None:
    """
    A lean fixed-iteration median timing (ms), or None if the kernel can't launch.

    ``n_warmup`` warm-up launches, then ``n_repeat`` CUDA-event-timed launches each preceded
    by a 256 MB L2 flush (cold-cache, like ``do_bench``), returning the median elapsed ms.
    """
    try:
        flush = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")
        for _ in range(max(1, n_warmup)):
            thunk()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        for i in range(n_repeat):
            flush.zero_()  # evict L2 so each timed launch reloads weights cold
            starts[i].record()
            thunk()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        return times[len(times) // 2]  # median
    except RuntimeError as e:
        if not any(pat in str(e) for pat in _BENCH_SKIP_ERRORS):
            raise
        torch.cuda.empty_cache()
        return None
