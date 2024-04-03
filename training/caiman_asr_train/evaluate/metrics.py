# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


from beartype import beartype
from beartype.typing import List, Tuple
from levenshtein_rs import levenshtein_list as levenshtein

from caiman_asr_train.data.text.whisper_text_normalizer import EnglishTextNormalizer


@beartype
def word_error_rate(
    hypotheses: List[str], references: List[str], standardize: bool = True
) -> Tuple[float, int, int]:
    """Compute Word Error Rate (WER) between two text lists.

    This function calculates the WER between two lists. It is calculated as the sum of
    addition, deletions, and substitutions of words in the hypotheses list entries compared
    to the respective entries in the references list, divided by the total number of words
    in the the references list.

    Parameters
    ----------
    hypotheses
        list of strings predicted by the system
    references
        list of transcripts
    standardize
        whether to standardize hypotheses and references before the WER calculation

    Returns
    -------
    wer
        the wer metric
    scores
        the number of additions, deletions, substitutions in all the entries of the
        hypotheses list
    words
        the number of words in the references list

    Raises
    ------
    ValueError:
        if the number of references is greater than the number of hypotheses
    """

    scores = 0
    words = 0
    len_diff = len(references) - len(hypotheses)
    if len_diff > 0:
        raise ValueError(
            "Unequal number of hypotheses and references: "
            "{0} and {1}".format(len(hypotheses), len(references))
        )

    for hyp, ref in zip(hypotheses, references):
        if standardize:
            hyp, ref = standardize_wer(hyp), standardize_wer(ref)
        hyp_list = hyp.split()
        ref_list = ref.split()
        words += len(ref_list)
        scores += levenshtein(hyp_list, ref_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")
    return wer, scores, words


@beartype
def standardize_wer(text: str) -> str:
    """
    Apply Whisper normalization rules to text.

    This function applies the Whisper normalization rules on the text of both
    hypotheses and references in order to minimize
    penalization of non-semantic text differences.


    Parameters
    ----------
    text
        string containing un-standardized text

    Returns
    -------
    norm_text_list
        string containing standardized text
    """
    standardizer = EnglishTextNormalizer()
    standard_text = standardizer(text)

    return standard_text
