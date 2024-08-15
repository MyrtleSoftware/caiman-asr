#!/usr/bin/env python3
import tempfile

import numpy as np
from beartype import beartype
from beartype.typing import Optional
from datasets import Audio, load_dataset


@beartype
def make_noise_dataset_for_dali(
    use_noise_audio_folder: bool,
    noise_dataset: str,
    noise_config: Optional[str],
    sample_rate: int,
) -> str:
    """Dali loads from disk faster than from RAM in tested cases,
    so noise data is saved to a tmpdir"""
    if use_noise_audio_folder:
        dataset = load_dataset("audiofolder", data_dir=noise_dataset, name=noise_config)
    else:
        dataset = load_dataset(noise_dataset, name=noise_config)

    resampled_dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=sample_rate, mono=True, decode=True)
    )

    array_list = [
        x["audio"]["array"].astype(np.float32) for x in resampled_dataset["train"]
    ]

    noise_dir = tempfile.mkdtemp()

    for i, noise in enumerate(array_list):
        np.save(f"{noise_dir}/{i}.npy", noise)

    return noise_dir
