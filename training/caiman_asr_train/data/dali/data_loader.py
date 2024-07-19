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

import math
import os
from pathlib import Path

import torch.distributed as dist
from beartype.typing import List, Optional, Union

from caiman_asr_train.args.hugging_face import HuggingFaceArgs
from caiman_asr_train.args.noise_augmentation import NoiseAugmentationArgs
from caiman_asr_train.data.dali.iterator import DaliRnntIterator
from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer
from caiman_asr_train.data.dali.pipeline import DaliPipeline
from caiman_asr_train.data.dali.utils import (
    _filter_files,
    _parse_json,
    generate_json_names_from_dirs,
    set_predicate,
)
from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.hugging_face.core import HuggingFaceReader
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.data.webdataset import LengthUnknownError, WebDatasetReader
from caiman_asr_train.setup.dali import DaliYAMLConfig
from caiman_asr_train.train_utils.distributed import print_once


class DaliDataLoader:
    """
    DataLoader is the main entry point to the data preprocessing pipeline.
    To use, create an object and then just iterate over `data_iterator`.
    DataLoader will do the rest for you.
    Example:
        data_layer = DataLoader(DaliTrainPipeline, path, json, bs, ngpu)
        data_it = data_layer.data_iterator
        for data in data_it:
            print(data)  # Here's your preprocessed data

    Args:
        device_type: Which device to use for preprocessing. Choose: "cpu", "gpu"
        pipeline_type: Choose: "train", "val"
    """

    def __init__(
        self,
        gpu_id,
        dataset_path: str,
        dali_yaml_config: DaliYAMLConfig,
        json_names: list,
        tokenizer,
        batch_size: int,
        sampler,
        pipeline_type: str,
        num_cpu_threads: int,
        num_buckets: int,
        seed: int,
        turn_off_initial_padding: bool,
        inspect_audio: bool,
        prob_narrowband: float,
        output_dir: Path,
        n_utterances_only: Optional[int],
        noise_augmentation_args: NoiseAugmentationArgs,
        data_source: DataSource,
        hugging_face_args: Optional[HuggingFaceArgs],
        grad_accumulation_batches: int = 1,
        device_type: str = "gpu",
        tar_files: Union[str, List[str], None] = None,
        val_from_dir: bool = False,
        audio_dir: str = None,
        txt_dir: str = None,
        no_logging: bool = False,
        mel_feat_normalizer: Optional[MelFeatNormalizer] = None,
    ):
        self.batch_size = batch_size
        self.no_logging = no_logging
        self.grad_accumulation_batches = grad_accumulation_batches
        self.drop_last = pipeline_type == "train"
        self.device_type = device_type
        self.pipeline_type = self._parse_pipeline_type(pipeline_type)
        self.val_from_dir = val_from_dir
        self.audio_dir = audio_dir
        self.txt_dir = txt_dir
        self.sampler = sampler
        self.num_buckets = num_buckets
        self.data_source = data_source
        self._dali_data_iterator = self._init_iterator(
            gpu_id=gpu_id,
            dataset_path=dataset_path,
            dali_yaml_config=dali_yaml_config,
            json_names=json_names,
            tokenizer=tokenizer,
            pipeline_type=pipeline_type,
            num_cpu_threads=num_cpu_threads,
            noise_augmentation_args=noise_augmentation_args,
            tar_files=tar_files,
            mel_feat_normalizer=mel_feat_normalizer,
            seed=seed,
            turn_off_initial_padding=turn_off_initial_padding,
            inspect_audio=inspect_audio,
            prob_narrowband=prob_narrowband,
            output_dir=output_dir,
            n_utterances_only=n_utterances_only,
            hugging_face_args=hugging_face_args,
        )

    def _init_iterator(
        self,
        gpu_id,
        dataset_path,
        dali_yaml_config: DaliYAMLConfig,
        json_names: list,
        tokenizer: Tokenizer,
        pipeline_type,
        num_cpu_threads,
        noise_augmentation_args: NoiseAugmentationArgs,
        turn_off_initial_padding: bool,
        inspect_audio: bool,
        seed: int,
        prob_narrowband: float,
        output_dir: Path,
        n_utterances_only: Optional[int],
        hugging_face_args: Optional[HuggingFaceArgs],
        tar_files: Union[str, List[str], None] = None,
        mel_feat_normalizer: Optional[MelFeatNormalizer] = None,
    ):
        """
        Returns data iterator. Data underneath this operator is preprocessed within Dali
        """
        max_duration = dali_yaml_config.max_duration
        max_transcript_len = dali_yaml_config.max_transcript_len
        predicate = set_predicate(max_duration, max_transcript_len)
        if self.data_source is DataSource.JSON:
            output_files, transcripts = {}, {}
            # if reading from directories, generate json names from directories
            if self.val_from_dir and pipeline_type == "val":
                json_names = generate_json_names_from_dirs(
                    dataset_path, self.audio_dir, self.txt_dir
                )
            for jname in json_names:
                of, tr = _parse_json(
                    jname if jname[0] == "/" else os.path.join(dataset_path, jname),
                    len(output_files),
                    predicate=predicate,
                )
                output_files.update(of)
                transcripts.update(tr)
            output_files, transcripts = _filter_files(
                output_files, transcripts, n_utterances_only, seed
            )

            assert len(output_files) == len(transcripts)
            self.sampler.make_file_list(output_files, json_names)
            print_once(f"Dataset read by DALI. Number of samples: {self.dataset_size}")
            shard_size = self._shard_size()
            external_reader = None
        elif self.data_source is DataSource.TARFILE:
            transcripts = None
            shard_size = -1  # lengths & shard_sizes are unknown
            external_reader = WebDatasetReader(
                file_root=dataset_path,
                tar_files=tar_files,
                shuffle=pipeline_type == "train",
                batch_size=self.batch_size,
                tokenizer=tokenizer,
                normalize_config=dali_yaml_config.normalize_config,
                max_duration=max_duration,
                max_transcript_len=max_transcript_len,
                num_buckets=self.num_buckets,
                sample_rate=dali_yaml_config.sample_rate,
            )
        elif self.data_source is DataSource.HUGGINGFACE:
            transcripts = None
            shard_size = -1  # lengths & shard_sizes are unknown
            external_reader = HuggingFaceReader(
                hugging_face_args=hugging_face_args,
                num_shards=dist.get_world_size() if dist.is_initialized() else 1,
                shard_id=dist.get_rank() if dist.is_initialized() else 0,
                sample_rate=dali_yaml_config.sample_rate,
                tokenizer=tokenizer,
                normalize_config=dali_yaml_config.normalize_config,
                max_duration=max_duration,
                max_transcript_length=max_transcript_len,
            )
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        self.pipeline = DaliPipeline(
            dali_yaml_config=dali_yaml_config,
            num_threads=num_cpu_threads,
            device_id=gpu_id,
            file_root=dataset_path,
            sampler=self.sampler,
            preprocessing_device=self.device_type,
            batch_size=self.batch_size,
            pipeline_type=pipeline_type,
            noise_augmentation_args=noise_augmentation_args,
            external_reader=external_reader,
            data_source=self.data_source,
            no_logging=self.no_logging,
            seed=seed,
            turn_off_initial_padding=turn_off_initial_padding,
            inspect_audio=inspect_audio,
            prob_narrowband=prob_narrowband,
            output_dir=output_dir,
            mel_feat_normalizer=mel_feat_normalizer,
        )

        return DaliRnntIterator(
            [self.pipeline],
            transcripts=transcripts,
            tokenizer=tokenizer,
            shard_size=shard_size,
            pipeline_type=pipeline_type,
            device_type=self.device_type,
            normalize_config=dali_yaml_config.normalize_config,
            data_source=self.data_source,
        )

    @staticmethod
    def _parse_pipeline_type(pipeline_type):
        pipe = pipeline_type.lower()
        assert pipe in ("train", "val"), 'Invalid pipeline type ("train", "val").'
        return pipe

    def _shard_size(self):
        """
        Total number of samples handled by a single GPU in a single epoch.
        """
        if self.data_source is not DataSource.JSON:
            raise LengthUnknownError(
                "Shard size and dataset length is not defined "
                "for tar files/HuggingFace datasets"
            )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.drop_last:
            divisor = world_size * self.batch_size * self.grad_accumulation_batches
            return self.dataset_size // divisor * divisor // world_size
        else:
            return int(math.ceil(self.dataset_size / world_size))

    def __len__(self):
        """
        Number of batches handled by each GPU.
        """
        if self.drop_last:
            assert (
                self._shard_size() % self.batch_size == 0
            ), f"{self._shard_size()} {self.batch_size}"

        return int(math.ceil(self._shard_size() / self.batch_size))

    def data_iterator(self):
        return self._dali_data_iterator

    def __iter__(self):
        return self._dali_data_iterator

    @property
    def dataset_size(self) -> int:
        if self.sampler is not None:
            return self.sampler.get_dataset_size()
        raise LengthUnknownError(
            "Dataset size is unknown. Number of samples is unknown when reading from "
            "tar file/HuggingFace dataset"
        )
