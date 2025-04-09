# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import heapq
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, cycle, islice

import numpy as np
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, TypeAlias, Union
from more_itertools import chunked as batched
from tqdm import tqdm

from caiman_asr_train.data.dali.manifest_ratios import (
    ManifestRatios,
    build_json_fracs,
    duration,
)
from caiman_asr_train.train_utils.distributed import print_once, scoped_time_once
from caiman_asr_train.utils.iter import flat, lmap, lstarmap, starmap_zip
from caiman_asr_train.utils.math import ceil_div, round_down, round_up


def hash_list_of_strings(li):
    """hash will be different in different processes"""
    return str(abs(hash("".join(li))))


# Has two keys: "label" -> integer label
#               "duration" -> float duration (sometimes an int)
Utterance: TypeAlias = Dict[str, Union[int, float]]

# A Manifest contains many Utterances.
# Each key is a file name
Manifest: TypeAlias = Dict[str, Utterance]


@beartype
@dataclass
class SamplerUtt:
    """
    A dataclass to hold the utterance information needed for the sampler.
    """

    file_name: str
    label: int
    duration: float


@beartype
def _to_utt(path: str, utt: Dict[str, Union[int, float]]) -> SamplerUtt:
    return SamplerUtt(
        file_name=path, label=utt["label"], duration=float(utt["duration"])
    )


class Sampler(ABC):
    @beartype
    def __init__(
        self,
        *,
        total_batches: Optional[int],
        batch_size: int,
        global_batch_size: Optional[int],
        world_size: int,
        resume_step: int = 0,
        rng: Optional[np.random.Generator] = None,
        pessimistic_first_batch: bool = True,
        dump_shard_lists: bool = False,
        randomize_n_epochs: int = 0,
    ) -> None:
        """
        Abstract class for samplers.

        Args:
            total_batches: Total number of batches to sample.
            batch_size: Number of samples in a batch (AKA per-gpu accumulation batch).
            global_batch_size: Number of samples in a global batch.
            resume_step: Resume training from this batch step.
            rng: Random number generator for shuffling, etc.
            world_size: Number of processes in distributed training.
            pessimistic_first_batch: Re-order the first epoch such that the first
                batch is the longest.
            dump_shard_lists: Debugging option to dump the shard lists to disk,
                this will write to /tmp/shard_{i}.txt
            randomize_n_epochs: Randomize the first `n` epochs.

        If `total_batches` is None, the sampler generates an epoch that contains
        all the data, in this mode the sampler is not compatible with manifest_ratios.

        For a weighted sampler the definition of an epoch is: the minimum number
        of steps before a repeated utterance is seen. Hence, the
        `epoch_size != dataset_size`.

        For example, if manifest m1 has utterances a1, a2 and manifest m2 has
        utterance b1 then when weighted equally, the model could see:
            a1, b1, a2, b1, ...
        Hence, the epoch size is 2 and the dataset size is 3.
        """
        self._file_list_path: Optional[str] = None
        self._dataset_size: Optional[int] = None
        self._epoch_size: Optional[int] = None

        self.world_size = world_size

        if global_batch_size is None:
            global_batch_size = batch_size * world_size

        self.batch_size = batch_size
        self.dist_batch_size = batch_size * world_size
        self.global_batch_size = global_batch_size
        self.randomize_n_epochs = randomize_n_epochs

        if self.randomize_n_epochs > 0:
            assert rng is not None, "Randomize_n_epochs requires rng"

        assert global_batch_size % self.world_size == 0
        assert global_batch_size % self.batch_size == 0
        assert global_batch_size % self.dist_batch_size == 0

        self.total_utts = None if total_batches is None else total_batches * batch_size
        self.resume_step = resume_step
        self.rng = rng
        self.pessimistic_first_batch = pessimistic_first_batch
        self.dump_shard_lists = dump_shard_lists

        print_once(f"{self.__class__.__name__}: {batch_size=}")

    @property
    @beartype
    def file_list_path(self) -> str:
        """
        Safe getter for file list path.
        """
        assert self._file_list_path is not None, "File list not initialized!"
        return self._file_list_path

    @property
    @beartype
    def dataset_size(self) -> int:
        """
        Safe getter for dataset size, this is number of unique utterances.
        """
        assert self._dataset_size is not None, "DatasetFile not initialized"
        return self._dataset_size

    @property
    @beartype
    def epoch_size(self) -> int:
        """
        Safe getter for epoch size, this is the number of utterances in an epoch.
        """
        assert self._epoch_size is not None, "Epoch size not initialized"
        return self._epoch_size

    @beartype
    def read_file_list(self) -> List[str]:
        """
        Read file list from disk.
        """
        with open(self.file_list_path, "r") as f:
            return [line.split()[0] for line in f]

    @abstractmethod
    @beartype
    def is_sampler_random(self) -> bool:
        """
        Test if the sampler has deterministic output for a given input.
        """
        pass

    @beartype
    def process_output_files(
        self,
        output_files: List[Manifest],
        json_names: List[str],
        manifest_ratios: ManifestRatios,
    ) -> Tuple[List[SamplerUtt], int]:
        """
        Hook because we cannot call `make_file_list` in the tests.
        """

        epochs = self._build_epochs(output_files, json_names, manifest_ratios)

        epochs = self._order_all(epochs)

        if self.pessimistic_first_batch:
            epochs[0] = self._find_pessimistic_batch(epochs[0])

        if self.randomize_n_epochs > 0:
            if self.pessimistic_first_batch:
                # First epoch must preserve pessimization which makes
                # first global_batch_size of utterances special
                N = self.global_batch_size
                epochs[0][N:] = self._shuffle_all([epochs[0][N:]])[0]
                beg = 1
            else:
                beg = 0

            end = self.randomize_n_epochs

            if beg < end:
                epochs[beg:end] = self._shuffle_all([epochs[beg:end]])

        files = self._to_dali_order(epochs)

        return files, len(epochs[0]) if epochs else 0

    @scoped_time_once("Building file list")
    @beartype
    def make_file_list(
        self,
        output_files: List[Manifest],
        json_names: List[str],
        manifest_ratios: ManifestRatios,
    ) -> None:
        """
        Generate DALI's file list from the parsed manifest files.

        Args:
            output_files: List of manifests.
            json_names: List of manifest names.
            manifest_ratios: Target proportions of each manifest in each epoch.

        If `manifest_ratios` is None, the sampler will use the length of each manifest.
        """

        self._dataset_size = sum(len(x) for x in output_files)

        if not dist.is_initialized() or dist.get_rank() == 0:
            self._file_list_path = str(
                os.path.join(
                    "/tmp", "rnnt_dali.file_list." + hash_list_of_strings(json_names)
                )
            )

            files, self._epoch_size = self.process_output_files(
                output_files, json_names, manifest_ratios
            )

            self._write_file_list(files)

        objects = [self._file_list_path, self._epoch_size]

        if dist.is_initialized():
            dist.broadcast_object_list(objects, src=0)

        self._file_list_path, self._epoch_size = objects

    @beartype
    def _move_chunk_to_front(self, n: int, epoch: List[SamplerUtt]) -> List[SamplerUtt]:
        """
        Move the chunk (of width n) with the longest utterances to the front.
        """
        N = len(epoch)

        if N <= n:
            return epoch

        batches = zip(range(0, N, n), batched(epoch, n), strict=True)

        offset, _ = max(batches, key=lambda x: sum(utt.duration for utt in x[1]))

        print_once(f"Moving chunk {offset // n}/{N // n} to the front")

        for i in range(n):
            epoch[i], epoch[offset + i] = epoch[offset + i], epoch[i]

        return epoch

    @beartype
    def _find_pessimistic_batch(self, epoch: List[SamplerUtt]) -> List[SamplerUtt]:
        """
        Reorder the epoch such that the first few batches are the longest.
        """
        if len(epoch) <= self.global_batch_size:
            return epoch

        epoch = self._move_chunk_to_front(self.global_batch_size, epoch)
        epoch = self._move_chunk_to_front(self.dist_batch_size, epoch)
        epoch = self._move_chunk_to_front(self.batch_size, epoch)

        # Also put the longest samples in the first global batch

        n_big_batches = self.global_batch_size // self.batch_size

        top_k = heapq.nlargest(
            n_big_batches, range(len(epoch)), lambda i: epoch[i].duration
        )

        for i, k in enumerate(top_k):
            lo = (i + 0) * self.batch_size
            hi = (i + 1) * self.batch_size

            batch = epoch[lo:hi]

            j, _ = max(enumerate(batch, start=lo), key=lambda x: x[1].duration)

            assert epoch[k].duration >= epoch[j].duration

            epoch[j], epoch[k] = epoch[k], epoch[j]

        return epoch

    @scoped_time_once("Convert to dali order")
    @beartype
    def _to_dali_order(self, epochs: List[List[SamplerUtt]]) -> List[SamplerUtt]:
        """
        Convert the list of files to DALI's order.

        Dali will split the dataset into world_size shards, each gpu will read
        sequentially from each shard.
        """
        if not epochs:
            return []

        n_drop = self.resume_step * self.batch_size

        if len(epochs) == 1:
            # Dali's sharding is OK here (and handles partial batches nicely)

            if self.world_size > 1:
                assert n_drop == 0, "Cannot resume single batch with multiple GPUs"

            return epochs[0][n_drop:]

        # Must re-order such that epochs are spilt across GPUs
        if not all(n % self.dist_batch_size == 0 for n in map(len, epochs)):
            raise ValueError("Cannot shard the epochs evenly")

        shards = [[] for _ in range(self.world_size)]

        for epoch in tqdm(epochs, desc="Sharding epochs"):
            for shard, batch in zip(cycle(shards), batched(epoch, self.batch_size)):
                shard.extend(batch)

        # Drop seen utterances
        shards = [shard[n_drop:] for shard in shards]

        if self.dump_shard_lists:
            for i, shard in enumerate(shards):
                with open(f"/tmp/shard_{i}.txt", "w") as f:
                    f.writelines(f"{utt.file_name}\n" for utt in shard)

        return flat(shards)

    @scoped_time_once("Ordering epochs")
    @beartype
    def _order_all(self, epochs: List[List[SamplerUtt]]) -> List[List[SamplerUtt]]:
        """
        Order utterances within the epochs, this displays a progress bar.
        """
        return lmap(self._order_epoch, tqdm(epochs, desc="Ordering epochs"))

    @beartype
    def _shuffle_all(self, samples: List[List[SamplerUtt]]) -> List[List[SamplerUtt]]:
        """
        Shuffle the samples in each epoch or manifest.
        """

        out = []

        assert self.rng is not None

        for s in samples:
            out.append(s)
            self.rng.shuffle(out[-1])

        return out

    @beartype
    def _log(
        self,
        json_names: List[str],
        utts_per_epoch: List[int],
        lens: List[int],
        epochs_for_repetition: int,
        num_epochs: int,
    ):
        print_once("UTTs per epoch per manifest:")

        tot = sum(utts_per_epoch)

        for k, u, t in zip(json_names, utts_per_epoch, lens, strict=True):
            print_once(f"{u:>8} ({u / tot:.1%}) of {k} ({u / t:.1%} of manifest)")

        print_once(
            f"Hence {epochs_for_repetition} epoch(s) to see all data, {num_epochs} in run."
        )

    @beartype
    def _calc_utts_per_epoch_per_manifest(
        self, json_names: List[str], json_fracs: List[float], lens: List[int]
    ) -> tuple[int, List[int]]:
        """
        Given the target fractions of each manifest in each epoch, calculate
        the number of utterances per epoch for each manifest.
        """

        # Convert to fractions
        tot = sum(json_fracs)
        targets = [t / tot for t in json_fracs]

        epoch_len = min(l / frac for frac, l in zip(targets, lens, strict=True))

        utts_per_epoch = [int(frac * epoch_len) for frac in targets]

        # Rounding down to global_batch_size means that each epoch will contain
        # a whole number of batches and the number of batches will be divisible
        # by the number of GPUs. Additionally, it ensures that we don't offset
        # the optimizer step frequency from the epoch frequency.

        utts_per_epoch = [
            round_down(u, multiple_of=self.global_batch_size) for u in utts_per_epoch
        ]

        assert all(0 < u <= l for u, l in zip(utts_per_epoch, lens)), (
            "Number of utterances in a manifest is smaller than "
            f"global batch size={self.global_batch_size}"
        )

        epochs_for_repetition = max(
            ceil_div(n, by=u) for n, u in zip(lens, utts_per_epoch)
        )

        n_epochs = ceil_div(self.total_utts, by=sum(utts_per_epoch))

        self._log(json_names, utts_per_epoch, lens, epochs_for_repetition, n_epochs)

        return n_epochs, utts_per_epoch

    @beartype
    def _build_epochs(
        self,
        output_files: List[Manifest],
        json_names: List[str],
        manifest_ratios: ManifestRatios,
    ) -> List[List[SamplerUtt]]:
        """
        Goals:
            - Each epoch contains no repetitions of the same file
            - Each epoch has the requested fraction of files from each manifest
            - All the files are used over the course of the training

        Returns:
            A List of epochs
        """

        lens = [len(x) for x in output_files]

        if manifest_ratios is None and self.total_utts is None:
            # This is how user gets exactly one pass over all the data.

            # Union of all the data
            epoch = {k: v for d in output_files for k, v in d.items()}

            assert len(epoch) == sum(lens), f"Duplicates in {json_names}"

            num_epochs = 1

            self._log(json_names, lens, lens, 1, num_epochs)

            return [lstarmap(_to_utt, epoch.items())]

        elif self.total_utts is None:
            raise ValueError("Please provide total_batches or json_fracs")
        else:
            durs = lmap(duration, output_files)
            json_fracs = build_json_fracs(manifest_ratios, lens, durs)

        assert self.total_utts is not None

        n_epochs, utts_per_epoch = self._calc_utts_per_epoch_per_manifest(
            json_names, json_fracs, lens
        )

        data = lmap(lambda m: lstarmap(_to_utt, m.items()), output_files)

        if self.is_sampler_random():
            data = self._shuffle_all(data)

        inf_data = starmap_zip(batched, map(cycle, data), utts_per_epoch)

        out = []

        for _, *epoch in zip(range(n_epochs), *inf_data):
            epoch_list = list(chain(*epoch))
            epoch_dict = set(utt.file_name for utt in epoch_list)

            assert len(epoch_list) == len(epoch_dict), "Repeated file(s) in epoch"

            out.append(epoch_list)

        return out

    @abstractmethod
    @beartype
    def _order_epoch(self, epoch: List[SamplerUtt]) -> List[SamplerUtt]:
        """
        Order the utterances in the epoch.
        """
        pass

    @scoped_time_once("Writing file list to disk")
    @beartype
    def _write_file_list(self, files: List[SamplerUtt]) -> None:
        """
        Write list of files to disk, this is a private method.
        """
        with open(self._file_list_path, "w") as f:
            f.writelines(f"{utt.file_name} {utt.label}\n" for utt in files)


class SimpleSampler(Sampler):
    @beartype
    def __init__(
        self,
        *,
        total_batches: Optional[int],
        batch_size: int,
        global_batch_size: Optional[int],
        world_size: int,
        resume_step: int = 0,
        dump_shard_lists: bool = False,
    ):
        """
        The simplest sampler, performs no additional reordering of the data.
        """
        super().__init__(
            total_batches=total_batches,
            batch_size=batch_size,
            global_batch_size=global_batch_size,
            world_size=world_size,
            resume_step=resume_step,
            rng=None,
            pessimistic_first_batch=False,
            dump_shard_lists=dump_shard_lists,
        )

    @beartype
    def _order_epoch(self, epoch: List[SamplerUtt]) -> List[SamplerUtt]:
        return epoch

    @beartype
    def is_sampler_random(self) -> bool:
        return False


class SortedSampler(Sampler):
    @beartype
    def __init__(
        self,
        *,
        total_batches: Optional[int],
        batch_size: int,
        global_batch_size: Optional[int],
        world_size: int,
        resume_step: int = 0,
        dump_shard_lists: bool = False,
    ):
        """
        Sorts the samples in each epoch by duration.
        """
        super().__init__(
            total_batches=total_batches,
            batch_size=batch_size,
            global_batch_size=global_batch_size,
            world_size=world_size,
            resume_step=resume_step,
            rng=None,
            pessimistic_first_batch=False,
            dump_shard_lists=dump_shard_lists,
        )

    @beartype
    def _order_epoch(self, epoch: List[SamplerUtt]) -> List[SamplerUtt]:
        iter = sorted(epoch, key=lambda utt: utt.duration, reverse=True)

        if self.world_size > 1:
            iter = list(
                chain.from_iterable(
                    islice(iter, i, None, self.world_size)
                    for i in range(self.world_size)
                )
            )

        return list(iter)

    @beartype
    def is_sampler_random(self) -> bool:
        return False


class RandomSampler(Sampler):
    @beartype
    def __init__(
        self,
        *,
        total_batches: Optional[int],
        batch_size: int,
        global_batch_size: Optional[int],
        world_size: int,
        resume_step: int,
        rng: np.random.Generator,
        pessimistic_first_batch: bool = True,
        dump_shard_lists: bool = False,
    ):
        """
        Randomly shuffles the samples in each epoch.
        """
        super().__init__(
            total_batches=total_batches,
            batch_size=batch_size,
            global_batch_size=global_batch_size,
            world_size=world_size,
            resume_step=resume_step,
            rng=rng,
            pessimistic_first_batch=pessimistic_first_batch,
            dump_shard_lists=dump_shard_lists,
        )

    @beartype
    def _order_epoch(self, epoch: List[SamplerUtt]) -> List[SamplerUtt]:
        return self._shuffle_all([epoch])[0]

    @beartype
    def is_sampler_random(self) -> bool:
        return True


class BucketingSampler(Sampler):
    def __init__(
        self,
        *,
        total_batches: Optional[int],
        batch_size: int,
        global_batch_size: int,  # Not optional as only used by train loop
        world_size: int,
        resume_step: int,
        rng: np.random.Generator,
        num_buckets: int,
        pessimistic_first_batch: bool = True,
        dump_shard_lists: bool = False,
        randomize_n_epochs: int = 0,
    ):
        super().__init__(
            total_batches=total_batches,
            batch_size=batch_size,
            global_batch_size=global_batch_size,
            world_size=world_size,
            resume_step=resume_step,
            rng=rng,
            pessimistic_first_batch=pessimistic_first_batch,
            dump_shard_lists=dump_shard_lists,
            randomize_n_epochs=randomize_n_epochs,
        )

        self.num_buckets = num_buckets

    @beartype
    def _order_epoch(self, utts: List[SamplerUtt]) -> List[SamplerUtt]:
        N = self.dist_batch_size

        assert len(utts) > 0, "Empty epoch"
        assert len(utts) % self.batch_size == 0, "Epoch not divisible by batch size"
        assert len(utts) % N == 0, "Batches not divisible by number of GPUs"

        # Randomize then stable sort to break ties
        self.rng.shuffle(utts)
        utts.sort(key=lambda x: x.duration)

        # Split into buckets of lengths divisible by batch size
        bucket_size = ceil_div(len(utts), by=self.num_buckets)
        bucket_size = round_up(bucket_size, multiple_of=N)
        bucket_size = max(bucket_size, N)

        buckets = list(batched(utts, bucket_size))

        assert len(buckets) <= self.num_buckets
        assert sum(len(b) for b in buckets) == len(utts)
        assert all(len(b) % N == 0 for b in buckets)

        # Shuffle the buckets.
        for i in range(len(buckets)):
            self.rng.shuffle(buckets[i])

        # Spit buckets into batches
        batches = flat(batched(b, N) for b in buckets)
        assert all(len(b) == N for b in batches)

        # Shuffle the batch order
        self.rng.shuffle(batches)

        # Flatten the batches
        return flat(batches)

    @beartype
    def is_sampler_random(self) -> bool:
        return True
