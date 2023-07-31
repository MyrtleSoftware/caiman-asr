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

# modified by rob@myrtle

import numpy as np
import torch
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from rnnt_train.common.text.preprocess import norm_and_tokenize


class DaliRnntIterator(object):
    """
    Returns batches of data for RNN-T training:
    preprocessed_signal, preprocessed_signal_length, transcript, transcript_length

    This iterator is not meant to be the entry point to Dali processing pipeline.
    Use DataLoader instead.
    """

    def __init__(
        self,
        dali_pipelines,
        transcripts,
        tokenizer,
        batch_size,
        shard_size,
        pipeline_type,
        device_type,
        normalize_transcripts=False,
        read_from_tar: bool = False,
    ):
        self.normalize_transcripts = normalize_transcripts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device_type = device_type
        self.read_from_tar = read_from_tar

        # in train pipeline shard_size is set to divisable by batch_size, so PARTIAL policy is safe
        out_arg_names = ["audio", "audio_shape", "label", "label_lens"]
        reader_name = "Reader" if not self.read_from_tar else None
        if pipeline_type == "val":
            self.dali_it = DALIGenericIterator(
                dali_pipelines,
                out_arg_names,
                reader_name=reader_name,
                dynamic_shape=True,
                auto_reset=True,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
        else:
            self.dali_it = DALIGenericIterator(
                dali_pipelines,
                out_arg_names,
                size=shard_size,
                dynamic_shape=True,
                auto_reset=True,
                last_batch_padded=True,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )

        if not self.read_from_tar:
            # then tokenize all transcripts once at start of training
            self.tokenize(transcripts)

    def tokenize(self, transcripts):
        transcripts = [transcripts[i] for i in range(len(transcripts))]
        transcripts = [
            norm_and_tokenize(t, self.tokenizer, self.normalize_transcripts)
            for t in transcripts
        ]

        transcripts = [torch.tensor(t) for t in transcripts]
        self.tr = np.array(transcripts, dtype=object)
        self.t_sizes = torch.tensor([len(t) for t in transcripts], dtype=torch.int32)

    def _gen_transcripts(self, data):
        """
        Generate transcripts in format expected by NN
        """
        labels = data["label"]
        if self.read_from_tar:
            # labels is the actual transcript
            transcripts = labels
            sizes = data["label_lens"].squeeze(1)
        else:
            # labels refers to an id that can be use to retrieve the transcript
            ids = labels.flatten().numpy()
            transcripts = self.tr[ids]
            # data['label_lens'] is populated with meaningless values and is not used
            sizes = self.t_sizes[ids]
        # Tensors are padded with 0. In `sentencepiece` we set it to <unk>,
        # because it cannot be disabled, and is absent in the data.
        # Note this is different from the RNN-T blank token (index 1023).
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True)
        # move to gpu only when requested - rob@myrtle
        if self.device_type == "gpu":
            transcripts = transcripts.cuda()
            sizes = sizes.cuda()

        return transcripts, sizes

    def __next__(self):
        data = self.dali_it.__next__()
        audio, audio_shape = data[0]["audio"], data[0]["audio_shape"][:, 1]
        if audio.shape[0] == 0:
            # empty tensor means, other GPUs got last samples from dataset
            # and this GPU has nothing to do; calling `__next__` raises StopIteration
            return self.dali_it.__next__()
        audio = audio[:, :, : audio_shape.max()]  # the last batch
        transcripts, transcripts_lengths = self._gen_transcripts(data[0])
        return audio, audio_shape, transcripts, transcripts_lengths

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
