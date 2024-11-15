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

import os
from itertools import chain, islice
from math import ceil

import numpy as np
import torch.distributed as dist

from caiman_asr_train.train_utils.distributed import time_print_once


def hash_list_of_strings(li):
    return str(abs(hash("".join(li))))


class SimpleSampler:
    def __init__(self, world_size, sort_by_duration=False):
        self.file_list_path = None
        self.dataset_size = None
        self.sort_by_duration = sort_by_duration
        self.world_size = world_size

    def write_file_list(self, files):
        time_print_once("Writing file list to disk")
        with open(self.file_list_path, "w") as f:
            f.writelines(f"{name} {label}\n" for name, label in files)
        time_print_once("Done writing file list to disk")

    def read_file_list(self):
        assert self.file_list_path
        files = []
        with open(self.file_list_path, "r") as f:
            for line in f:
                file, idx = line.split()
                files.append(file)
        return files

    def get_file_list_path(self):
        assert (
            self.file_list_path
        ), "File list not initialized. Run make_file_list first"
        return self.file_list_path

    def get_dataset_size(self):
        assert self.dataset_size, "Dataset size not known. Run make_file_list first"
        return self.dataset_size

    def is_sampler_random(self):
        return False

    def process_output_files(self, output_files):
        self.dataset_size = len(output_files)

        def dur(x):
            return x[1]["duration"]

        iter = (
            sorted(output_files.items(), key=dur, reverse=True)
            if self.sort_by_duration
            else output_files.items()
        )

        if self.sort_by_duration and self.world_size > 1:
            iter = list(
                chain.from_iterable(
                    islice(iter, i, None, self.world_size)
                    for i in range(self.world_size)
                )
            )

        return [(path, entry["label"]) for path, entry in iter]

    def make_file_list(self, output_files, json_names):
        objects = [self.file_list_path, self.dataset_size]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1 or dist.get_rank() == 0:
            self.file_list_path = str(
                os.path.join(
                    "/tmp", "rnnt_dali.file_list." + hash_list_of_strings(json_names)
                )
            )
            self.write_file_list(self.process_output_files(output_files))
            objects = [self.file_list_path, self.dataset_size]
        else:
            objects = [None, None]
        if dist.is_initialized():
            dist.broadcast_object_list(objects, src=0)
        self.file_list_path, self.dataset_size = objects


class BucketingSampler(SimpleSampler):
    def __init__(
        self,
        num_buckets,
        batch_size,
        num_workers,
        training_steps,
        global_batch_size,
        rng,
        resume_step,
    ):
        super(BucketingSampler, self).__init__(world_size=num_workers)
        # Shuffle the data in the same way across all processes:
        self.rng = rng
        self.num_buckets = num_buckets
        self.num_training_steps = training_steps
        self.global_batch_size = global_batch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resume_step = resume_step

    def process_output_files(self, output_files):
        time_print_once("Shuffling utterances")
        names = list(output_files)
        num_epochs = ceil(self.num_training_steps * self.global_batch_size / len(names))
        lengths = [output_files[name]["duration"] for name in names]
        labels = np.array([output_files[name]["label"] for name in names])
        len_ids = np.argsort(lengths)
        buckets = np.array_split(len_ids, self.num_buckets)

        gbs = self.batch_size * self.num_workers
        shuffled_buckets = np.array(
            [
                perm
                for _ in range(num_epochs)  # for every epoch
                for bucket in buckets  # from every bucket
                for perm in self.rng.permutation(bucket)  # pick samples in random order
            ]
        )

        # drop last batch
        epochs = np.reshape(shuffled_buckets, [num_epochs, -1])
        to_drop = epochs.shape[1] - (epochs.shape[1] // gbs * gbs)
        for epoch in epochs:
            dropped_idxs = self.rng.choice(epochs.shape[1], to_drop, replace=False)
            if dropped_idxs is not None:
                epoch[dropped_idxs] = -1
        epochs = epochs[epochs != -1].reshape(num_epochs, -1)
        self.dataset_size = epochs.shape[1]

        epochs_iters_batch = np.reshape(epochs, [num_epochs, -1, gbs])

        # shuffle iterations in epochs preserving batches
        for epoch in epochs_iters_batch:
            self.rng.shuffle(epoch, axis=0)

        # start 0th epoch with 10 batches of randomly shuffled longest utterances
        valid_len_ids = epochs_iters_batch[0, :, :].flatten()
        if len(valid_len_ids) > 10 * gbs:
            long_utt_ids = len_ids[-10 * gbs :]
            self.rng.shuffle(long_utt_ids)
            long_utt_ids = np.array([v for v in long_utt_ids if v in valid_len_ids])
            epochs_iters_batch = self.prepend_subset(
                epochs_iters_batch, long_utt_ids, 0
            )

        # reshape to final form
        epochs_iters_batch_worker = np.reshape(
            epochs_iters_batch, [num_epochs, -1, self.batch_size, self.num_workers]
        )
        workers_epochs_iters_batch = np.moveaxis(epochs_iters_batch_worker, -1, 0)
        flatten_labels = workers_epochs_iters_batch.flatten()
        flatten_labels = flatten_labels[self.resume_step * self.global_batch_size :]

        result = [(names[i], labels[i]) for i in flatten_labels]
        time_print_once("Done shuffling utterances")
        return result

    def is_sampler_random(self):
        return True

    def prepend_subset(self, array: np.ndarray, subset: np.ndarray, epoch: int = 0):
        """
        Prepend items defined in `subset` to the beginning of randomly shuffled `array`.
        The `array` is of shape (epoch, epoch_batches, batch_items) and prepending works
        for epoch specific by `epoch` argument. The use is to have longest utterances at
        the beginning of 0th epoch to scan for possible CUDA OOMs.
        """
        # Flatten the array and find indexes of subset elements
        epoch_array = array[epoch, :, :]
        flat_array = epoch_array.flatten()
        remainder = np.array([x for x in flat_array if x not in subset])
        out_array = np.concatenate((subset, remainder), axis=0)
        array[epoch, :, :] = out_array.reshape(epoch_array.shape)

        return array
