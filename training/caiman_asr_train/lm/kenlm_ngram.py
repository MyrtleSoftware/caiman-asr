import os
from dataclasses import dataclass

import kenlm
import numpy as np
from beartype import beartype
from beartype.typing import Tuple


@beartype
class KenLmModel:
    """
    A wrapper class for the KenLM n-gram language model.

    Provides an interface to load a KenLM model and score n-grams with it.
    """

    lm_score_scale = 1.0 / np.log10(np.e)

    def __init__(self, model_path: str):
        self.model = kenlm.Model(model_path)

    def score_ngram(
        self, ngram: str, current_lm_state: kenlm.State
    ) -> Tuple[float, kenlm.State]:
        """Score the given n-gram and return the score and updated state."""
        next_state = kenlm.State()
        lm_score = self.model.BaseScore(current_lm_state, ngram, next_state)
        lm_score *= self.lm_score_scale
        return lm_score, next_state


@beartype
@dataclass
class NgramInfo:
    path: str
    scale_factor: float


def find_ngram_path(base_path: str) -> str | None:
    """Search for 'ngram.binary' and then 'ngram.arpa' in the given directory."""
    binary_file = os.path.join(base_path, "ngram.binary")
    arpa_file = os.path.join(base_path, "ngram.arpa")

    if os.path.exists(binary_file):
        return binary_file
    elif os.path.exists(arpa_file):
        return arpa_file
    return None
