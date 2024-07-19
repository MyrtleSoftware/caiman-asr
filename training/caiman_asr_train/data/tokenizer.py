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
from beartype import beartype
from beartype.typing import List
from numpy import random


@beartype
class Tokenizer:
    def __init__(
        self,
        labels: List[str],
        sentpiece_model: str,
        sampling: float = 0.0,
    ):
        """Converts transcript to a sequence of tokens.

        Args:
            labels (str): all possible output symbols
            sentpiece_model (str): name of tokenizer model
            sampling (float): probability of random sampling from
                             tokens when encoding text.
        """
        # Other code reads this attribute:
        self.charset = labels

        self.sentpiece = spm.SentencePieceProcessor(model_file=sentpiece_model)
        self.num_labels = len(self.sentpiece)
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

        inds = self.sentpiece.encode(
            transcript, out_type=int, enable_sampling=do_sampling
        )
        assert 0 not in inds, f"<unk> found during tokenization (OOV?)\n{transcript}"
        return inds

    def detokenize(self, inds: int | List[int]) -> str:
        return self.sentpiece.decode(inds)
