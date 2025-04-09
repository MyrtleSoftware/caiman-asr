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
from itertools import islice
from pathlib import Path

import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union

from caiman_asr_train.args.hugging_face import HuggingFaceArgs
from caiman_asr_train.args.noise_augmentation import NoiseAugmentationArgs
from caiman_asr_train.data.dali.iterator import DaliRnntIterator
from caiman_asr_train.data.dali.manifest_ratios import (
    ManifestRatios,
    build_json_fracs,
    duration,
)
from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer
from caiman_asr_train.data.dali.pipeline import DaliPipeline
from caiman_asr_train.data.dali.sampler import Manifest
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
from caiman_asr_train.train_utils.distributed import (
    print_once,
    scoped_time_once,
    time_print_once,
)
from caiman_asr_train.utils.iter import lmap
from caiman_asr_train.utils.math import ceil_div


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

    @beartype
    def __init__(
        self,
        gpu_id,
        dataset_path: str,
        dali_yaml_config: DaliYAMLConfig,
        json_names: Optional[List],
        manifest_ratios: ManifestRatios,
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
        final_padding_secs: float,
        grad_accumulation_batches: int = 1,
        device_type: str = "gpu",
        tar_files: Union[str, List[str], None] = None,
        val_from_dir: bool = False,
        audio_dir: Optional[str] = None,
        txt_dir: Optional[str] = None,
        no_logging: bool = False,
        mel_feat_normalizer: Optional[MelFeatNormalizer] = None,
    ):
        self.batch_size = batch_size
        self.no_logging = no_logging
        self.grad_accumulation_batches = grad_accumulation_batches
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
            manifest_ratios=manifest_ratios,
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
            final_padding_secs=final_padding_secs,
        )
        self.iter_state = None

    @beartype
    def _init_iterator(
        self,
        gpu_id,
        dataset_path,
        dali_yaml_config: DaliYAMLConfig,
        json_names: Optional[List],
        manifest_ratios: ManifestRatios,
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
        final_padding_secs: float,
        tar_files: Union[str, List[str], None] = None,
        mel_feat_normalizer: Optional[MelFeatNormalizer] = None,
    ):
        """
        Returns data iterator. Data underneath this operator is preprocessed within Dali
        """
        max_duration = dali_yaml_config.max_duration
        min_duration = dali_yaml_config.min_duration
        max_transcript_len = dali_yaml_config.max_transcript_len
        predicate = set_predicate(
            max_duration, max_transcript_len, min_duration=min_duration
        )

        if self.data_source is DataSource.JSON:
            assert (
                json_names is not None
            ), "json_names must be provided for JSON data source"
            output_files, manifests, transcripts = self._parse_json(
                json_names=json_names,
                manifest_ratios=manifest_ratios,
                pipeline_type=pipeline_type,
                dataset_path=dataset_path,
                predicate=predicate,
                n_utterances_only=n_utterances_only,
                seed=seed,
            )

            self.sampler.make_file_list(manifests, json_names, manifest_ratios)

            print_once(f"Dataset read by DALI, {self.dataset_size} samples")

            external_reader = None
        elif self.data_source is DataSource.TARFILE:
            transcripts = None
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
                min_duration=min_duration,
            )
            output_files = None
        elif self.data_source is DataSource.HUGGINGFACE:
            transcripts = None
            external_reader = HuggingFaceReader(
                hugging_face_args=hugging_face_args,
                num_shards=dist.get_world_size() if dist.is_initialized() else 1,
                shard_id=dist.get_rank() if dist.is_initialized() else 0,
                sample_rate=dali_yaml_config.sample_rate,
                tokenizer=tokenizer,
                normalize_config=dali_yaml_config.normalize_config,
                max_duration=max_duration,
                max_transcript_length=max_transcript_len,
                min_duration=min_duration,
            )
            output_files = None
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        time_print_once("Initializing Dali pipeline")
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
            final_padding_secs=final_padding_secs,
            inspect_audio=inspect_audio,
            prob_narrowband=prob_narrowband,
            output_dir=output_dir,
            mel_feat_normalizer=mel_feat_normalizer,
        )
        time_print_once("Done initializing Dali pipeline")

        return DaliRnntIterator(
            [self.pipeline],
            transcripts=transcripts,
            tokenizer=tokenizer,
            pipeline_type=pipeline_type,
            device_type=self.device_type,
            normalize_config=dali_yaml_config.normalize_config,
            data_source=self.data_source,
            output_files=output_files,
        )

    @scoped_time_once("Parsing and filtering jsons")
    @beartype
    def _parse_json(
        self,
        json_names,
        manifest_ratios: ManifestRatios,
        pipeline_type: str,
        dataset_path,
        predicate,
        n_utterances_only: Optional[int],
        seed,
    ) -> Tuple[Manifest, List[Manifest], Dict[int, str]]:
        output_files: List[Dict] = []
        transcripts: List[Dict] = []

        if self.val_from_dir and pipeline_type == "val":
            json_names = generate_json_names_from_dirs(
                dataset_path, self.audio_dir, self.txt_dir
            )

        for name in json_names:
            of, tr = _parse_json(
                name if name[0] == "/" else os.path.join(dataset_path, name),
                0,  # Will do relabeling manually after filtering
                predicate=predicate,
            )

            output_files.append(of)
            transcripts.append(tr)

        if n_utterances_only is not None:
            # Take fraction of n_utterances_only from each json file

            manifest_lengths = lmap(len, output_files)
            manifest_durations = lmap(duration, output_files)
            json_fracs = build_json_fracs(
                manifest_ratios, manifest_lengths, manifest_durations
            )

            tot = sum(json_fracs)
            target = [int(n_utterances_only * frac / tot) for frac in json_fracs]

            for i, n_utt in enumerate(target):
                output_files[i], transcripts[i] = _filter_files(
                    output_files[i], transcripts[i], n_utt, seed
                )

                seed += 1

        # Make squashed transcripts

        s_transcripts = {}

        for of, tr in zip(output_files, transcripts):
            n = len(s_transcripts)

            for filename, (old_label, transcript) in zip(of, tr.items()):
                assert of[filename]["label"] == old_label
                s_transcripts[old_label + n] = transcript
                of[filename]["label"] += n

        for of in output_files:
            check_batch_size(self.batch_size, of)

        flat_output_files = {k: v for d in output_files for k, v in d.items()}

        assert len(flat_output_files) == len(s_transcripts)
        assert len(flat_output_files) == sum(len(of) for of in output_files)

        return flat_output_files, output_files, s_transcripts

    @staticmethod
    def _parse_pipeline_type(pipeline_type):
        pipe = pipeline_type.lower()
        assert pipe in ("train", "val"), 'Invalid pipeline type ("train", "val").'
        return pipe

    def __len__(self):
        """
        Number of batches handled by each GPU.
        If the epoch_size isn't a multiple of the world_size,
        then __len__ is (sometimes but not always) 1 too large on some GPUs.
        """
        n_gpus = dist.get_world_size() if dist.is_initialized() else 1

        utts_per_gpu = ceil_div(self._epoch_size, by=n_gpus)

        return ceil_div(utts_per_gpu, by=self.batch_size)

    def _islice(self):

        if self.iter_state is None:
            self.iter_state = iter(self._dali_data_iterator)

        # If we have reached the end of the dataset the cached iterator
        # will raise StopIteration. In that case we need a new iterator
        # that will repeat the dataset. We only allow this on the first
        # iteration as repeating halfway through an epoch is likely a bug.

        try:
            yield next(self.iter_state)
        except StopIteration:
            self.iter_state = iter(self._dali_data_iterator)
            yield next(self.iter_state)

        for batch in islice(self.iter_state, len(self) - 1):
            yield batch

    def __iter__(self):

        if self.sampler is None:
            return self._dali_data_iterator

        return self._islice()

    @property
    def dataset_size(self) -> int:
        if self.sampler is not None:
            return self.sampler.dataset_size
        raise LengthUnknownError(
            "Dataset size is unknown. Number of samples is unknown when reading from "
            "tar file/HuggingFace dataset"
        )

    @property
    def _epoch_size(self) -> int:
        """
        Number of sample in the epoch.
        """
        if self.sampler is not None:
            return self.sampler.epoch_size

        raise LengthUnknownError(
            "Epoch size is unknown. Number of samples is unknown when reading from "
            "tar file/HuggingFace dataset"
        )


@beartype
def check_batch_size(batch_size: int, output_files: dict[str, dict]):
    max_duration = max(
        label_and_duration["duration"] for label_and_duration in output_files.values()
    )
    if batch_size * max_duration > 1.5e5:
        raise ValueError(
            "Stopping because this validation will likely use all your RAM. "
            "Try reducing batch size"
        )
