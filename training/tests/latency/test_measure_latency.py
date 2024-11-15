#!/usr/bin/env python3

from caiman_asr_train.latency.measure_latency import CTMTimestamp, align_transcripts


def test_align_transcripts():
    align_transcripts(
        [CTMTimestamp("hello", 1.0, "file1")],
        [CTMTimestamp("world", 2.0, "file1")],
        None,
    )
