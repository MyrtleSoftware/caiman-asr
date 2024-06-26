#!/usr/bin/env python3
import subprocess
from pathlib import Path

import yappi
from beartype import beartype
from beartype.typing import List

from caiman_asr_train.train_utils.distributed import get_rank_or_zero, is_rank_zero


@beartype
def save_timings(
    dataloading_total: float,
    feat_proc_total: float,
    forward_backward_total: float,
    output_dir: str,
    step: int,
    timestamp: str,
) -> None:
    benchmark_dir = Path(output_dir) / "benchmark"
    with open(
        benchmark_dir / f"timings_step{step}_rank{get_rank_or_zero()}_{timestamp}.txt",
        "w",
    ) as f:
        f.write(f"{dataloading_total=}\n")
        f.write(f"{feat_proc_total=}\n")
        f.write(f"{forward_backward_total=}\n")


@beartype
def set_up_profiling(
    profiler: bool, output_dir: str, timestamp: str
) -> List[subprocess.Popen]:
    benchmark_dir = Path(output_dir) / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    if not profiler:
        return []
    yappi.set_clock_type("wall")
    yappi.start()
    if is_rank_zero():
        system_info_logfile = benchmark_dir / f"system_info_{timestamp}.txt"
        subprocess.run(
            ["../training/scripts/profile/record_system_info.sh", system_info_logfile]
        )
        nvidia_smi_logfile = benchmark_dir / f"nvidia_smi_log_{timestamp}.txt"
        nvidia_smi_process = subprocess.Popen(
            ["../training/scripts/profile/record_nvidia_smi.bash", nvidia_smi_logfile]
        )
        top_logfile = benchmark_dir / f"top_log_{timestamp}.html"
        top_process = subprocess.Popen(
            ["../training/scripts/profile/record_top.bash", top_logfile]
        )
        return [nvidia_smi_process, top_process]
    else:
        return []


@beartype
def finish_profiling(
    profiler: bool, output_dir: str, profilers: List[subprocess.Popen], timestamp: str
) -> None:
    if profiler:
        benchmark_dir = Path(output_dir) / "benchmark"
        yappi.get_func_stats().save(
            benchmark_dir / f"program{get_rank_or_zero()}_{timestamp}.prof",
            type="pstat",
        )
        for p in profilers:
            p.terminate()
