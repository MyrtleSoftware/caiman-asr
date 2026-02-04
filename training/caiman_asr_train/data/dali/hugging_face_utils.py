#!/usr/bin/env python3
import os
from hashlib import sha256
from pathlib import Path

import numpy as np
from beartype import beartype
from beartype.typing import Optional
from datasets import Audio, load_dataset

from caiman_asr_train.train_utils.distributed import maybe_cached


@beartype
def make_noise_dataset_for_dali(
    use_noise_audio_folder: bool,
    noise_dataset: str,
    noise_config: Optional[str],
    sample_rate: int,
    noise_cache_dir: str | Path,
) -> str:
    """
    Process a hugging face noise dataset into a dali compatible format.
    The output is written to to noise_dir and the path is returned.
    This function is safe to call from multiple processes, only one will
    actually process the dataset. Furthermore, the noise is cached for future
    runs.
    """

    def populate(noise_dir):
        if use_noise_audio_folder:
            dataset = load_dataset(
                "audiofolder", data_dir=noise_dataset, name=noise_config
            )
        else:
            dataset = load_dataset(noise_dataset, name=noise_config)

        resampled_dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=sample_rate, mono=True, decode=True)
        )

        array_list = [
            x["audio"]["array"].astype(np.float32) for x in resampled_dataset["train"]
        ]

        os.makedirs(noise_dir)

        for i, noise in enumerate(array_list):
            np.save(f"{noise_dir}/{i}.npy", noise)

    data = f"{use_noise_audio_folder}{noise_dataset}{noise_config}{sample_rate}"
    hash = sha256(data.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(noise_cache_dir, hash)

    maybe_cached(populate, cache_dir)
    return cache_dir
