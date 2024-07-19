#!/usr/bin/env python3
import sox
from beartype import beartype


@beartype
def convert_to_standard_format(input_path: str, output_path: str) -> None:
    # Calling build() multiple times with the same tfm object can
    # corrupt the output (especially if the input is multi-channel)
    tfm = sox.Transformer()
    tfm.convert(samplerate=16000, n_channels=1, bitdepth=16)
    tfm.build(input_path, output_path)
