# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.
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
import sentencepiece as spm
from beartype.typing import List, Optional
from numpy import random


class Tokenizer:
    def __init__(
        self,
        labels: List[str],
        sentpiece_model: Optional[str] = None,
        sampling: float = 0.0,
    ):
        """Converts transcript to a sequence of tokens.

        Args:
            labels (str): all possible output symbols
            sentpiece_model (str): name of tokenizer model
            sampling (float): probability of random sampling from
                             tokens when encoding text.
        """
        # For labels use vocab or load wordpieces
        self.charset = labels
        self.use_sentpiece = sentpiece_model is not None

        if self.use_sentpiece:
            self.sentpiece = spm.SentencePieceProcessor(model_file=sentpiece_model)
            self.num_labels = len(self.sentpiece)
            self.sampling = sampling
        else:
            self.num_labels = len(self.charset)
            self.label2ind = {lab: i for i, lab in enumerate(self.charset)}
            self.sampling = sampling

    def tokenize(self, transcript: str) -> List[int]:
        """
        Encodes the input transcript into tokens.
        There is the option of randomly sampling from the available tokens rather
        than using the most likely ones (default sampling probability is 0.0).
        The sampling option is only available when the tokenizer is a
        sentencePiece model.
        """
        # do sampling according to probability from uniform distribution
        do_sampling = False
        if self.sampling > 0.0:
            do_sampling = random.random_sample() < self.sampling

        if self.use_sentpiece:
            inds = self.sentpiece.encode(
                transcript, out_type=int, enable_sampling=do_sampling
            )
            assert (
                0 not in inds
            ), f"<unk> found during tokenization (OOV?)\n{transcript}"
        else:
            inds = [self.label2ind[x] for x in transcript if x in self.label2ind]
        return inds

    def detokenize(self, inds: List[int]) -> str:
        if self.use_sentpiece:
            return self.sentpiece.decode(inds)
        else:
            return "".join(self.charset[i] for i in inds)
