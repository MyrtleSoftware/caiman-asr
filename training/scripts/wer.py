#!/usr/bin/env python3

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

# prepared from other MLPerf code by Myrtle

# usage : ./wer.py transcripts.txt predictions.txt
# where both txt files contain one sentence per line

import sys

from editdistance import eval as levenshtein


def word_error_rate(hypotheses, references):
    """Computes average Word Error Rate (WER) between two text lists."""

    scores = 0
    words = 0
    len_diff = len(references) - len(hypotheses)
    if len_diff > 0:
        raise ValueError(
            "Uneqal number of hypthoses and references: "
            "{0} and {1}".format(len(hypotheses), len(references))
        )
    elif len_diff < 0:
        hypotheses = hypotheses[:len_diff]

    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += levenshtein(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")
    return wer, scores, words


references = []
hypotheses = []

with open(sys.argv[1], "r") as f:
    for line in f:
        references.append(line)

with open(sys.argv[2], "r") as f:
    for line in f:
        hypotheses.append(line)

wer, _scores, _words = word_error_rate(hypotheses, references)

print(f"{wer:.4f}")
