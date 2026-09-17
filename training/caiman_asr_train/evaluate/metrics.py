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


from collections import Counter
from dataclasses import dataclass

import jiwer
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from rapidfuzz.distance import Levenshtein as RapidLevenshtein

from caiman_asr_train.data.text.allowed_charset import get_allowed_charset
from caiman_asr_train.data.text.normalizers import (
    MemoizedLowercase,
    lowercase_normalize,
    punctuation_normalize,
)
from caiman_asr_train.evaluate.error_rates import ErrorRate, decide_and_split
from caiman_asr_train.evaluate.punctuation_error_rate import (
    punctuation_error_rate_function,
)

# Shared module-level instance so the lru_cache inside it actually persists
# across calls, instead of being rebuilt (and thrown away) on every single
# word_error_rate()/calculate_keyword_wers() invocation.
_WER_NORMALIZER = MemoizedLowercase(user_symbols=[])


@beartype
@dataclass
class WerDetails:
    """
    Breakdown of WER results.


    Attributes
    ----------
    wer : float
        The WER metric.
    scores : int
        The number of additions, deletions, substitutions in all the entries of the
        hypotheses list.
    words : int
        Total number of words in the reference list.
    substitutions : int
        Number of substitutions.
    insertions : int
        Number of insertions.
    deletions : int
        Number of deletions.
    """

    wer: float
    scores: int
    words: int
    substitutions: int
    insertions: int
    deletions: int
    norm_ref: Optional[str] = None
    norm_hyp: Optional[str] = None

    @classmethod
    def aggregate(cls, items: list["WerDetails"]) -> "WerDetails":
        """Aggregates a list of WerDetails into a single summary instance."""
        tot_scores = sum(i.scores for i in items)
        tot_words = sum(i.words for i in items)
        tot_substitutions = sum(i.substitutions for i in items)
        tot_insertions = sum(i.insertions for i in items)
        tot_deletions = sum(i.deletions for i in items)

        return cls(
            wer=(tot_scores / tot_words) if tot_words > 0 else 0.0,
            scores=tot_scores,
            words=tot_words,
            substitutions=tot_substitutions,
            insertions=tot_insertions,
            deletions=tot_deletions,
        )


@beartype
@dataclass
class PerDetails:
    """
    Breakdown of PER results.

    Attributes
    ----------
    per : float
        The PER metric.
    scores : int
        The number of additions, deletions, substitutions of punctuation marks in all
        the entries of the hypotheses list.
    punc_symbols : int
        Total number of punctuation symbols in the reference list.
    """

    per: float
    scores: int
    punc_symbols: int

    @classmethod
    def aggregate(cls, items: list["PerDetails"]) -> "PerDetails":
        """Aggregates a list of PerDetails into a single summary instance."""
        tot_scores = sum(i.scores for i in items)
        tot_symbols = sum(i.punc_symbols for i in items)

        return cls(
            per=(tot_scores / tot_symbols) if tot_symbols > 0 else 0.0,
            scores=tot_scores,
            punc_symbols=tot_symbols,
        )


@beartype
def punctuation_error_rate(
    hypotheses: List[str],
    references: List[str],
    punctuation_marks: List[str] = [",", ".", "?"],
    punctuation_normalization: bool = True,
) -> PerDetails:
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
    PerDetails
        Breakdown of PER results including per, scores, and punc_symbols.
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
        per = per_score / pun_symbols
        return PerDetails(per=per, scores=per_score, punc_symbols=pun_symbols)
    else:
        return PerDetails(per=0.0, scores=0, punc_symbols=0)


@beartype
def word_error_rate(
    hypotheses: List[str],
    references: List[str],
    error_rate: ErrorRate,
    lang_code: str,
) -> WerDetails:
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
    error_rate
        type of error rate to compute (word, char, or mixture)
    lang_code
        language code for the language of the hypotheses and references

    Returns
    -------
    WerDetails
        Detailed WER breakdown results including wer, scores, words, substitutions,
        insertions, and deletions.

    Raises
    ------
    ValueError:
        if the number of references is greater than the number of hypotheses
    """

    if len(references) != len(hypotheses):
        raise ValueError(
            "Unequal number of hypotheses and references: "
            "{0} and {1}".format(len(hypotheses), len(references))
        )

    scores = 0
    words = 0
    insertions = 0
    deletions = 0
    substitutions = 0
    norm_hypotheses = []
    norm_references = []

    for hyp, ref in zip(hypotheses, references):
        hyp = _WER_NORMALIZER.normalize(hyp, lang_code)
        ref = _WER_NORMALIZER.normalize(ref, lang_code)
        norm_hypotheses.append(hyp)
        norm_references.append(ref)
        hyp_list = decide_and_split(hyp, error_rate)
        ref_list = decide_and_split(ref, error_rate)
        words += len(ref_list)

        editops = RapidLevenshtein.editops(ref_list, hyp_list)
        ins = sum(1 for op in editops if op[0] == "insert")
        dels = sum(1 for op in editops if op[0] == "delete")
        subs = sum(1 for op in editops if op[0] == "replace")
        insertions += ins
        deletions += dels
        substitutions += subs
        scores += ins + dels + subs

    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")

    norm_references = " ".join(norm_references)
    norm_hypotheses = " ".join(norm_hypotheses)

    return WerDetails(
        wer=wer,
        scores=scores,
        words=words,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        norm_hyp=norm_hypotheses,
        norm_ref=norm_references,
    )


@beartype
def count_keyword_stats(
    hypotheses: List[str],
    references: List[str],
    keywords: List[str],
    lang_code: str,
) -> Tuple[int, int, int]:
    """
    Counts true positives, false positives, and false negatives for keywords.
    """
    tp = fp = fn = 0

    charset = get_allowed_charset([lang_code], lowercase_only=True)

    # Only normalize, don't standardize
    keywords = [
        lowercase_normalize(k, charset=charset, lang_code=lang_code) for k in keywords
    ]
    keywords_set = {k for k in keywords}

    for hyp, ref in zip(hypotheses, references):
        hyp = lowercase_normalize(hyp, charset=charset, lang_code=lang_code)
        ref = lowercase_normalize(ref, charset=charset, lang_code=lang_code)

        correct_hyps = []

        # If ref is the empty string jiwer.process_words fails,
        # so handle this case separately.
        if ref.strip():
            aligned = jiwer.process_words(ref, hyp)
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
    hypotheses: List[str],
    references: List[str],
    keywords: List[str],
    lang_code: str,
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
    tp, fp, fn = count_keyword_stats(hypotheses, references, keywords, lang_code)
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
    lang_code: str,
) -> Tuple[float | None, float | None]:
    """
    Calculates biased and unbiased WERs for a list of hypotheses and references,
    given a list of keywords.

    Biased WER (b-WER) is the WER of the hypotheses
    that contain at least one keyword in the references.

    Unbiased WER (u-WER) is the WER of the hypotheses
    that do not contain any keywords in the references.
    """
    # Normalize only to decide the biased/unbiased split; word_error_rate()
    # below normalizes hyp/ref itself, so passing the raw text through (and
    # relying on _WER_NORMALIZER's shared cache) avoids normalizing twice.
    norm_references = [_WER_NORMALIZER.normalize(r, lang_code) for r in references]
    norm_keywords = {_WER_NORMALIZER.normalize(k, lang_code) for k in keywords}

    biased_hypotheses, biased_references = [], []
    unbiased_hypotheses, unbiased_references = [], []

    for hyp, ref, norm_ref in zip(hypotheses, references, norm_references):
        if not set(norm_ref.split()).isdisjoint(norm_keywords):
            biased_hypotheses.append(hyp)
            biased_references.append(ref)
        else:
            unbiased_hypotheses.append(hyp)
            unbiased_references.append(ref)

    if biased_references:
        biased_wer = word_error_rate(
            biased_hypotheses,
            biased_references,
            ErrorRate.WORD,
            lang_code,
        ).wer
    else:
        biased_wer = None
    if unbiased_references:
        unbiased_wer = word_error_rate(
            unbiased_hypotheses,
            unbiased_references,
            ErrorRate.WORD,
            lang_code,
        ).wer
    else:
        unbiased_wer = None

    return biased_wer, unbiased_wer


@beartype
def compute_keyword_metrics(
    hypotheses: List[str],
    references: List[str],
    keywords: List[str],
    lang_code: str,
) -> Dict[str, float | None]:
    """
    Computes keyword-based metrics for a list of hypotheses and references,
    given a list of keywords.

    Returns a dictionary of the metrics: biased WER, unbiased WER,
    precision, recall, and F1 score.
    """
    b_wer, u_wer = calculate_keyword_wers(hypotheses, references, keywords, lang_code)
    precision, recall, f1 = calculate_precision_recall_f1(
        hypotheses, references, keywords, lang_code
    )
    return {
        "b_wer": b_wer,
        "u_wer": u_wer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
