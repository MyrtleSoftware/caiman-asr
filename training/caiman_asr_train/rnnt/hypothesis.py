# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Myrtle Software Limited. All rights reserved.
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

# Being an amalgamation of MLCommons MLPerf and NVIDIA Nemo code,

import copy
from dataclasses import dataclass, field

import kenlm
import torch
from beartype import beartype
from beartype.typing import Tuple

from caiman_asr_train.keywords.trie import Keywords
from caiman_asr_train.lm.kenlm_ngram import KenLmModel

SPU_UNICODE = 0x2581  # sentencepiece underscore (space repr) unicode code point
CHR_SPU_UNICODE = chr(SPU_UNICODE)
MAX_UNICODE: int = 0x10FFFF  # max unicode code point
HASHSIZE: int = 1_000_000_000_039  # int64 prime


@beartype
@dataclass
class Hypothesis:
    """
    A beam search hypothesis instance.

    During beam search, tokens are defined as 'final' - these final tokens are then
    truncated from the front of the sequences in the Hypothesis instance so that they
    are not considered in the future. However, the score, hashval, y_length_tot and
    state are retained for the un-truncated sequence.

    NOTE: the first token in the sequence arrays is always ignored as it is either
    the SOS tag or it has already been shipped as a final.

    Args:
    score       : cumulative logprob
    p_seq       : sequence of probabilities for each token in y_seq
    y_seq       : sequence of non-blank token idxs
    y_len_t     : number of non-blank tokens added this timestep
    timesteps   : sequence of time idxs for when each token in y_seq was added
    s_seq       : sequence of str tokens (sentencepiece tokens)
    hashval     : int64 hash computed from the str sequence
    pred_state  : prediction network state
    ngram_lm_state: kenlm.State
    is_terminal : whether this hypothesis is terminal (i.e. EOS token)
    """

    score: float
    p_seq: list[float]
    y_seq: list[int]
    y_len_t: int
    timesteps: list[int]
    s_seq: list[str]
    hashval: int
    pred_state: Tuple[torch.Tensor, torch.Tensor] | None
    ngram_lm_state: kenlm.State | None = None
    is_terminal: bool = False
    kws_state: Keywords.State = field(default_factory=Keywords.init)

    _prev_length: int = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(score={self.score:6.2f}, '{self.transcript}')"
        )

    @property
    def y_last(self):
        return self.y_seq[-1]

    @property
    def y_length_tot(self):
        """
        Total length of sequence (excluding blanks) up to this point.
        """
        return len(self.y_seq) + self._prev_length

    def truncate(self, tkn_idx: int):
        """
        Truncate the hypothesis before the given token index.

        The final token is kept in the sequence but then ignored going forwards.
        """
        idx_keep = tkn_idx - 1
        self._prev_length += idx_keep

        self.p_seq = self.p_seq[idx_keep:]
        self.s_seq = self.s_seq[idx_keep:]
        self.y_seq = self.y_seq[idx_keep:]
        self.timesteps = self.timesteps[idx_keep:]

    @property
    def transcript(self) -> str:
        """
        Not required during beam search, but useful for debugging.
        """
        return token_strs_to_transcript(self.s_seq[1:])

    def update_hash(self, new_str):
        """
        Update hashval given new string.
        """
        h = self.hashval
        for c in new_str:
            h = (h * MAX_UNICODE) + ord(c)
            h = h % HASHSIZE
        self.hashval = h

    def check(self, tokenizer):
        """
        Check a hypothesis for consistency.
        """
        assert len(self.y_seq) > 0
        assert self.y_length_tot >= len(self.y_seq)
        assert (
            len(self.y_seq) == len(self.timesteps) == len(self.s_seq) == len(self.p_seq)
        )
        assert self.transcript == tokenizer.detokenize(self.y_seq[1:]).strip(), (
            self.transcript,
            tokenizer.detokenize(self.y_seq[1:]),
        )

    def clone(self):
        """
        Clone the hypothesis for extension.

        The model states (pred state, ngram state, ...) are shared between self and the
        cloned Hypothesis in order to save memory but everything else is copied.
        """

        return Hypothesis(
            score=self.score,
            p_seq=self.p_seq.copy(),
            y_seq=self.y_seq.copy(),
            y_len_t=self.y_len_t,
            timesteps=self.timesteps.copy(),
            s_seq=self.s_seq.copy(),
            hashval=self.hashval,
            pred_state=self.pred_state,
            ngram_lm_state=self.ngram_lm_state,
            is_terminal=self.is_terminal,
            kws_state=copy.deepcopy(self.kws_state),
            _prev_length=self._prev_length,
        )


def token_strs_to_transcript(tokens: list[str]) -> str:
    return "".join(tokens).replace(CHR_SPU_UNICODE, " ").strip()


@beartype
def init_sos_hyp(sos_tkn: int, ngram_lm: KenLmModel | None) -> Hypothesis:
    """
    Return a SOS hypothesis.
    """
    init_ngram_state = None
    if ngram_lm:
        init_ngram_state = kenlm.State()
        ngram_lm.model.BeginSentenceWrite(init_ngram_state)

    return Hypothesis(
        score=0.0,
        p_seq=[1.0],
        y_seq=[sos_tkn],
        y_len_t=1,
        timesteps=[-1],
        s_seq=[CHR_SPU_UNICODE],
        hashval=0,
        pred_state=None,
        ngram_lm_state=init_ngram_state,
        is_terminal=False,
    )
