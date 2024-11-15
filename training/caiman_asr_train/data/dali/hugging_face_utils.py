#!/usr/bin/env python3
import os
from hashlib import sha256

import numpy as np
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Optional
from datasets import Audio, load_dataset

from caiman_asr_train.train_utils.distributed import print_once


@beartype
def make_noise_dataset_for_dali(
    use_noise_audio_folder: bool,
    noise_dataset: str,
    noise_config: Optional[str],
    sample_rate: int,
    noise_dir_parent: str = "/root/.cache/myrtle/noise_cache/",
) -> str:
    """
    Process a hugging face noise dataset into a dali compatible format.
    The output is written to to noise_dir and the path is returned.
    This function is safe to call from multiple processes, only one will
    actually process the dataset. Furthermore, the noise is cached for future
    runs.
    """
    data = f"{use_noise_audio_folder}{noise_dataset}{noise_config}{sample_rate}"
    hash = sha256(data.encode("utf-8")).hexdigest()

    noise_dir = os.path.join(noise_dir_parent, hash)
    done_file = os.path.join(noise_dir_parent, f"{hash}done.txt")

    if os.path.isdir(noise_dir) and os.path.isfile(done_file):
        print_once(f"Noise cache hit at: {noise_dir}")
        return noise_dir
    else:
        print_once(f"Noise cache miss at: {noise_dir}")

    multi_gpu = dist.is_initialized()

    if multi_gpu:
        # Let all the processes observe the cache miss.
        dist.barrier()

    if not multi_gpu or dist.get_rank() == 0:
        if not os.path.exists(noise_dir):
            os.makedirs(noise_dir)

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

        for i, noise in enumerate(array_list):
            np.save(f"{noise_dir}/{i}.npy", noise)

        with open(done_file, "w") as f:
            f.write("done")

    if multi_gpu:
        dist.barrier()

    return noise_dir
