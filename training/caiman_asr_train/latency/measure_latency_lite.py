#!/usr/bin/env python3

from statistics import mean, median, pstdev

from beartype import beartype
from beartype.typing import Dict, List


@beartype
def compute_latency_metrics(
    latencies: List[float],
    sil_latency: List[float],
    eos_latency: List[float],
    percentiles: List[float | int] = [90, 99],
) -> Dict[str, float]:
    latency_metrics = {}
    latency_num = len(latencies)

    if sil_latency:
        latency_metrics["mean-SIL-latency"] = mean(sil_latency)
        latency_metrics["median-SIL-latency"] = median(sil_latency)
        latency_metrics["stdev-SIL-latency"] = pstdev(sil_latency)

    if eos_latency:
        latency_metrics["mean-EOS-latency"] = mean(eos_latency)
        latency_metrics["stdev-EOS-latency"] = pstdev(eos_latency)
        latency_metrics["median-EOS-latency"] = median(eos_latency)

    if not latency_num:
        return latency_metrics

    latencies = sorted(latencies)
    latency_metrics["mean-emission-latency"] = mean(latencies)
    latency_metrics["stdev-emission-latency"] = pstdev(latencies)
    latency_metrics["median-emission-latency"] = median(latencies)

    for perc in percentiles:
        latency_metrics[f"p{perc}-emission-latency"] = latencies[
            int(latency_num * perc / 100)
        ]

    return latency_metrics
