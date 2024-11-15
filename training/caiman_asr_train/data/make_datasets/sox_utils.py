#!/usr/bin/env python3
from pathlib import Path

import sox
from beartype import beartype


@beartype
def convert_to_standard_format(input_path: str, output_path: str) -> None:
    # Calling build() multiple times with the same tfm object can
    # corrupt the output (especially if the input is multi-channel)
    tfm = sox.Transformer()
    tfm.convert(samplerate=16000, n_channels=1, bitdepth=16)
    tfm.build(input_path, output_path)


@beartype
def concat_and_convert_to_standard_format(
    input_paths: list[str], output_path: str
) -> None:
    assert len(input_paths), "No input paths provided"
    if len(input_paths) == 1:
        # The combiner requires at least two input audios
        convert_to_standard_format(input_paths[0], output_path)
        return
    # Calling build() multiple times with the same combiner object
    # may be safe, but out of caution, don't reuse the object
    combiner = sox.Combiner()
    combiner.convert(samplerate=16000, n_channels=1, bitdepth=16)
    # Without this line, sox will autodetect the file type,
    # but it'll print noisy warnings:
    combiner.set_input_format(file_type=[Path(path).suffix[1:] for path in input_paths])
    combiner.build(input_paths, output_path, "concatenate")


@beartype
def trim_and_convert_to_standard_format(
    input_path: str, output_path: str, start_time: float, end_time: float
) -> None:
    tfm = sox.Transformer()
    tfm.convert(samplerate=16000, n_channels=1, bitdepth=16)
    tfm.trim(start_time=start_time, end_time=end_time)
    tfm.build(input_path, output_path)
