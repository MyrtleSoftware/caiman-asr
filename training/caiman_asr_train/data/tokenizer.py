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

from caiman_asr_train.data.unk_handling import UnkHandling, check_tokenized_transcript
from caiman_asr_train.utils.iter import flat


@beartype
class Tokenizer:
    def __init__(
        self,
        labels: List[str],
        unk_handling: UnkHandling,
        sentpiece_model: str,
        sampling: float,
    ):
        """Converts transcript to a sequence of tokens.

        Args:
            labels (str): all possible output symbols
            sentpiece_model (str): name of tokenizer model
            sampling (float): probability of random sampling from
                             tokens when encoding text.
        """
        self.charset = labels
        self.sentpiece_model = sentpiece_model
        self.sampling = sampling
        self.unk_handling = unk_handling

        self.sentpiece = spm.SentencePieceProcessor(model_file=sentpiece_model)

        self.num_labels = len(self.sentpiece)

    def _tokenize_word(self, transcript: str) -> List[int]:
        # Do sampling according to probability from uniform distribution
        do_sampling = False
        if self.sampling > 0.0:
            do_sampling = random.random_sample() < self.sampling

        for _ in range(5):
            # This is a hack to work around a bug in SentencePiece where it
            # can split up a user defined symbol when sampling is enabled.

            inds = self.sentpiece.encode(
                transcript, out_type=int, enable_sampling=do_sampling
            )

            if 0 not in inds:
                break
        else:
            check_tokenized_transcript(inds, transcript, self.unk_handling)

        return inds

    def tokenize(self, transcript: str) -> List[int]:
        """
        Encodes the input transcript into tokens.
        There is the option of randomly sampling from the available tokens rather
        than using the most likely ones (default sampling probability is 0.0).
        The sampling option is only available when the tokenizer is a
        sentencePiece model.
        """
        return flat(self._tokenize_word(word) for word in transcript.split())

    def detokenize(self, inds: int | List[int]) -> str:
        if inds == 0:
            # sentpiece.decode([0]) == " ⁇ " but sentpiece.decode(0) inconsistently is "".
            # group_timestamps() requires consistency.
            return "⁇"
        return self.sentpiece.decode(inds)
