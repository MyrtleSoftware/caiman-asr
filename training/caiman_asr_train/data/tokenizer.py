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
from pathlib import Path

import sentencepiece as spm
from beartype import beartype
from beartype.typing import List, Optional
from numpy import random

from caiman_asr_train.data.cache import persistent_cache
from caiman_asr_train.data.label_smoother import LabelSmoother
from caiman_asr_train.data.unk_handling import UnkHandling, check_tokenized_transcript
from caiman_asr_train.utils.iter import flat


@beartype
def identity(x: str) -> str:
    """Identity function for label smoothing."""
    return x


@beartype
class Tokenizer:
    def __init__(
        self,
        labels: List[str],
        unk_handling: UnkHandling,
        sentpiece_model: str,
        sampling: float,
        label_smoothing_kwargs: Optional[dict[str, float | str | Path]] = None,
    ):
        """Converts transcript to a sequence of tokens.

        Args:
            labels (str): all possible output symbols
            sentpiece_model (str): name of tokenizer model
            sampling (float): probability of random sampling from
                             tokens when encoding text.
            label_smoothing_kwargs_kwargs: Arguments for label smoothing.
        """
        self.charset = labels
        self.sentpiece_model = sentpiece_model
        self.sampling = sampling
        self.unk_handling = unk_handling
        self.sentpiece = spm.SentencePieceProcessor(model_file=sentpiece_model)
        self.num_labels = len(self.sentpiece)

        if label_smoothing_kwargs is None:
            self.label_smoother = identity
        else:
            label_smoothing_kwargs["charset"] = set(self.charset)

            if "dict_cache_dir" not in label_smoothing_kwargs:
                label_smoothing_kwargs["dict_cache_dir"] = persistent_cache(
                    "word_cache"
                )

            self.label_smoother = LabelSmoother(
                **label_smoothing_kwargs,
            )

    def _tokenize_word(self, word: str) -> List[int]:
        # Do label smoothing
        word = self.label_smoother(word)

        # Do sampling according to probability from uniform distribution
        do_sampling = False
        if self.sampling > 0.0:
            do_sampling = random.random_sample() < self.sampling

        for _ in range(5):
            # This is a hack to work around a bug in SentencePiece where it
            # can split up a user defined symbol when sampling is enabled.

            inds = self.sentpiece.encode(
                word, out_type=int, enable_sampling=do_sampling
            )

            if 0 not in inds:
                break
        else:
            check_tokenized_transcript(inds, word, self.unk_handling)

        return inds

    def tokenize(self, transcript: str) -> List[int]:
        """
        Encodes the input transcript into tokens.
        There is the option of randomly sampling from the available tokens rather
        than using the most likely ones (default sampling probability is 0.0).
        The sampling option is only available when the tokenizer is a
        sentencePiece model.
        """

        try:
            return flat(self._tokenize_word(word) for word in transcript.split())
        except ValueError as e:
            raise ValueError(f"From: {transcript} got: {e}")

    def detokenize(self, inds: int | List[int]) -> str:
        if inds == 0:
            # sentpiece.decode([0]) == " ⁇ " but sentpiece.decode(0) inconsistently is "".
            # group_timestamps() requires consistency.
            return "⁇"
        return self.sentpiece.decode(inds)
