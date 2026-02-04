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


import string
from collections import Counter

import jiwer
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from levenshtein_rs import levenshtein_list as levenshtein

from caiman_asr_train.data.text.normalizers import (
    lowercase_normalize,
    punctuation_normalize,
)
from caiman_asr_train.data.text.whisper_text_normalizer import EnglishTextNormalizer
from caiman_asr_train.evaluate.error_rates import ErrorRate, decide_and_split
from caiman_asr_train.evaluate.punctuation_error_rate import (
    punctuation_error_rate_function,
)


@beartype
def punctuation_error_rate(
    hypotheses: List[str],
    references: List[str],
    punctuation_marks: List[str] = [",", ".", "?"],
    punctuation_normalization: bool = True,
) -> Tuple[float, int]:
    """Compute Punctuation Error Rate (PER) between two text lists.

    This function calculates the PER between two lists. It is calculated as the sum of
    addition, deletions, and substitutions of punctuation marks in the hypotheses list
    entries compared to the respective entries in the references list, divided by the
    total number of punctuation marks in the references list.

    Parameters
    ----------
    hypotheses
        list of strings predicted by the system
    references
        list of transcripts
    punctuation_marks
        list of punctuation marks to consider in the PER calculation
    punctuation_normalization
        whether to normalize punctuation marks before the PER calculation

    Returns
    -------
    per_score
        the per metric
    pun_symbols
        the number of punctuation marks in the references list
    """
    per_score, pun_symbols = 0, 0
    for hyp, ref in zip(hypotheses, references):
        if punctuation_normalization:
            hyp, ref = punctuation_normalize(hyp), punctuation_normalize(ref)
        _per, _pun_symbols = punctuation_error_rate_function(
            [ref], [hyp], punctuation_marks
        )
        per_score += round(_per * _pun_symbols)
        pun_symbols += _pun_symbols
    if pun_symbols != 0:
        return per_score / pun_symbols, pun_symbols
    else:
        return 0.0, pun_symbols


@beartype
def word_error_rate(
    hypotheses: List[str],
    references: List[str],
    error_rate: ErrorRate,
    standardize: bool = True,
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
    if len(references) != len(hypotheses):
        raise ValueError(
            "Unequal number of hypotheses and references: "
            "{0} and {1}".format(len(hypotheses), len(references))
        )

    for hyp, ref in zip(hypotheses, references):

        if standardize:
            hyp, ref = standardize_wer(hyp), standardize_wer(ref)
        hyp_list = decide_and_split(hyp, error_rate)
        ref_list = decide_and_split(ref, error_rate)
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
    Standardize text in preparation for WER calculation.
    This function applies the Whisper normalization rules
    and the train-time normalizer on the text of both
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
    # Hardcode charset, so WER doesn't depend on what charset the user trains with.
    # Keep <> so that the EnglishTextNormalizer recognizes and removes tags.
    charset = list(" '<>" + string.ascii_lowercase)
    norm_text = lowercase_normalize(text, charset=charset)
    standardizer = EnglishTextNormalizer()
    standard_text = standardizer(norm_text)

    return standard_text


@beartype
def count_keyword_stats(
    hypotheses: List[str],
    references: List[str],
    keywords: List[str],
    normalize: bool = False,
) -> Tuple[int, int, int]:
    """
    Counts true positives, false positives, and false negatives for keywords.
    """
    tp = fp = fn = 0
    charset = list(" '<>" + string.ascii_lowercase)
    if normalize:
        # Only normalize, don't standardize
        keywords = [lowercase_normalize(k, charset=charset) for k in keywords]
    keywords_set = {k for k in keywords}

    for hyp, ref in zip(hypotheses, references):
        if normalize:
            hyp = lowercase_normalize(hyp, charset=charset)
            ref = lowercase_normalize(ref, charset=charset)

        aligned = jiwer.process_words(ref, hyp)
        correct_hyps = []
        for ali in aligned.alignments[0]:
            if ali.type == "equal":
                correct_hyps += aligned.hypotheses[0][
                    ali.hyp_start_idx : ali.hyp_end_idx
                ]

        hyp_count = Counter(word for word in hyp.split() if word in keywords_set)
        ref_count = Counter(word for word in ref.split() if word in keywords_set)
        correct_count = Counter(word for word in correct_hyps if word in keywords_set)

        for k in keywords_set:
            tp += correct_count[k]
            fp += hyp_count[k] - correct_count[k]
            fn += ref_count[k] - correct_count[k]

    return tp, fp, fn


@beartype
def calculate_precision_recall_f1(
    hypotheses: List[str], references: List[str], keywords: List[str], normalize: bool
) -> Tuple[float, float, float]:
    """
    Calculates precision, recall, and F1 score for a list of hypotheses and references,
    given a list of keywords.

    Precision measures the accuracy of positive keyword predictions.
    Recall measures the model's ability to find all positive keywords.
    F1 score is the harmonic mean of precision and recall.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    tp, fp, fn = count_keyword_stats(hypotheses, references, keywords, normalize)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


@beartype
def calculate_keyword_wers(
    hypotheses: List[str],
    references: List[str],
    keywords: List[str],
    standardize: bool = True,
) -> Tuple[float | None, float | None]:
    """
    Calculates biased and unbiased WERs for a list of hypotheses and references,
    given a list of keywords.

    Biased WER (b-WER) is the WER of the hypotheses
    that contain at least one keyword in the references.

    Unbiased WER (u-WER) is the WER of the hypotheses
    that do not contain any keywords in the references.
    """
    if standardize:
        hypotheses = [standardize_wer(h) for h in hypotheses]
        references = [standardize_wer(r) for r in references]
        keywords = [standardize_wer(k) for k in keywords]

    biased_hypotheses, biased_references = [], []
    unbiased_hypotheses, unbiased_references = [], []

    for hyp, ref in zip(hypotheses, references):
        ref_words = set(ref.split())
        if not ref_words.isdisjoint(keywords):
            biased_hypotheses.append(hyp)
            biased_references.append(ref)
        else:
            unbiased_hypotheses.append(hyp)
            unbiased_references.append(ref)

    if biased_references:
        biased_wer, _, _ = word_error_rate(
            biased_hypotheses,
            biased_references,
            ErrorRate.WORD,
            standardize=standardize,
        )
    else:
        biased_wer = None
    if unbiased_references:
        unbiased_wer, _, _ = word_error_rate(
            unbiased_hypotheses,
            unbiased_references,
            ErrorRate.WORD,
            standardize=standardize,
        )
    else:
        unbiased_wer = None

    return biased_wer, unbiased_wer


@beartype
def compute_keyword_metrics(
    hypotheses: List[str],
    references: List[str],
    keywords: List[str],
    standardize: bool = True,
) -> Dict[str, float | None]:
    """
    Computes keyword-based metrics for a list of hypotheses and references,
    given a list of keywords.

    Returns a dictionary of the metrics: biased WER, unbiased WER,
    precision, recall, and F1 score.
    """
    b_wer, u_wer = calculate_keyword_wers(hypotheses, references, keywords, standardize)
    precision, recall, f1 = calculate_precision_recall_f1(
        hypotheses, references, keywords, standardize
    )
    return {
        "b_wer": b_wer,
        "u_wer": u_wer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
