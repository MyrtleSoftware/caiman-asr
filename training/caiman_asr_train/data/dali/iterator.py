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

import torch
from beartype.typing import List, Tuple
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from caiman_asr_train.data.dali.filename import FileNameExtractor
from caiman_asr_train.data.dali.pipeline import DaliPipeline
from caiman_asr_train.data.dali.token_cache import NormalizeCache
from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig
from caiman_asr_train.train_utils.distributed import time_print_once


class DaliRnntIterator:
    """
    Returns batches of data for RNN-T training:
        (preprocessed_signal, preprocessed_signal_length,
        transcript, transcript_length, raw_transcript)

    This iterator is not meant to be the entry point to Dali processing pipeline.
    Use DataLoader instead.
    """

    def __init__(
        self,
        dali_pipelines: List[DaliPipeline],
        transcripts: dict,
        tokenizer: Tokenizer,
        pipeline_type: str,
        device_type: str,
        data_source: DataSource,
        normalize_config: NormalizeConfig,
        output_files: dict[str, dict] | None,
    ):
        time_print_once(f"Making normalized text cache for {pipeline_type}")
        self.normalize_cache = NormalizeCache(
            normalize_config, tokenizer, device_type, data_source, transcripts
        )
        time_print_once(f"Done making token cache for {pipeline_type}")

        self.file_name_extractor = FileNameExtractor(output_files, data_source)

        # in train pipeline shard_size is set to divisible by batch_size, so
        # PARTIAL policy is safe
        out_arg_names = [
            "audio",
            "audio_shape",
            "label",
            "label_lens",
            "raw_transcript",
            "fname",
        ]

        assert len(dali_pipelines) == 1, "Only one pipeline is expected"

        self.dali_it = DALIGenericIterator(
            pipelines=dali_pipelines,
            output_map=out_arg_names,
            reader_name=dali_pipelines[0].reader_name,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def __next__(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]
    ]:
        """
        Return next sample as ('audio', 'audio_lens', 'transcript',
                               'transcript_lens', 'raw_transcript', 'fname')
        """
        data = self.dali_it.__next__()
        audio, audio_shape = data[0]["audio"], data[0]["audio_shape"][:, 1]
        if audio.shape[0] == 0:
            # empty tensor means, other GPUs got last samples from dataset
            # and this GPU has nothing to do; calling `__next__` raises StopIteration
            return self.dali_it.__next__()
        audio = audio[:, :, : audio_shape.max()]  # the last batch
        (
            transcripts,
            transcripts_lengths,
            raw_transcripts,
        ) = self.normalize_cache.get_transcripts(data[0])
        fnames = self.file_name_extractor.get_fnames(data[0])
        return (
            audio,
            audio_shape,
            transcripts,
            transcripts_lengths,
            raw_transcripts,
            fnames,
        )

    def next(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        return self.__next__()

    def __iter__(self):
        return self
