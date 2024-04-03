#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Originally from https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py
# Additions/amendments by Myrtle.ai

import json
import os
import re

from caiman_asr_train.data.text.whisper_basic_normalizer import (
    remove_symbols_and_diacritics,
)


class EnglishTextNormalizer:
    def __init__(self):
        # filler words to ignore
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\blemme\b": "let me",
            r"\bdunno\b": "do not know",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bkinda\b": "king of",
            r"\bgotta\b": "got to",
            r"\blotta\b": "lot of",
            r"\bsorta\b": "sort of",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            r"\balright\b": "all right",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"it's\b": "it is",
            r"he's\b": "he is",
            r"she's\b": "she is",
            r"that's\b": "that is",
            r"what's\b": "what is",
            r"there's\b": "there is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        # british to american english conversion
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str) -> str:
        s = s.lower()

        # remove words between brackets
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        # s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        # when there's a space before an apostrophe
        s = re.sub(r"\s+'", "'", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        # remove periods not followed by numbers
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)
        s = remove_symbols_and_diacritics(
            s, keep=".%$¢€£'"
        )  # keep numeric symbols and apostrophe

        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        # replace any successive whitespaces with a space
        s = re.sub(r"\s+", " ", s)

        return s


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings.

    This class will convert all the British English into American English.
    The source of the english.json file is not retrievable from the
    reference: https://www.tysto.com/uk-us-spelling-list.html
    but it can be found here:
    https://github.com/openai/whisper/blob/main/whisper/normalizers/english.json
    """

    def __init__(self):
        mapping_path = os.path.join(os.path.dirname(__file__), "english.json")
        self.mapping = json.load(open(mapping_path))

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())
