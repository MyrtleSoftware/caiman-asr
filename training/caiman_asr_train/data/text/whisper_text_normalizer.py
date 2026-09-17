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

import os
import re

from caiman_asr_train.data.text.whisper_basic_normalizer import (
    remove_symbols_and_diacritics,
)
from caiman_asr_train.utils.fast_json import fast_read_json


class EnglishTextNormalizer:
    def __init__(
        self,
        preserve_case: bool = False,
        preserve_punctuation: bool = False,
        expand_contractions: bool = True,
    ):
        self.preserve_case = preserve_case
        self.preserve_punctuation = preserve_punctuation
        # filler words to ignore
        fillers = r"(hm+|m{2,}|uh-?huh+|uh+|um+|m+h+m+|huh+)"
        # Match a whole-word filler and capture at most one punctuation mark
        # just before it (named group `pre`) and just after it (`post`).
        # Optional whitespace and single hyphen on either side are absorbed too
        self.filler_regex = re.compile(
            rf"(?P<pre>[.,;:!?-]?)\s*-?\b(?:{fillers})\b-?\s*(?P<post>[.,;:!?-]?)",
            flags=re.IGNORECASE,
        )
        # contractions to expand (only for WER calculation, not training)
        self.expand_contractions = expand_contractions
        self.contractions = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\blemme\b": "let me",
            r"\bdunno\b": "do not know",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bkinda\b": "kind of",
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
            r"\bcuz\b": "because",
            r"\B'cause\b": "because",
            # dr (without period) is optional
            # because 'dr' is also used in maths (derivatives)
            r"\bdr\b": "doctor ",
        }
        self.replacers = {
            r"\balright\b": "all right",
            # contractions in titles/prefixes
            r"\bmr\b\.?": "mister ",
            r"\bmrs\b\.?": "missus ",
            # dr. (with period) is nearly always used as a title
            r"\bdr\b\.": "doctor ",
            r"\bst\b\.?": "saint ",
            r"\bprof\b\.?": "professor ",
            r"\bcapt\b\.?": "captain ",
            r"\bgov\b\.?": "governor ",
            r"\bald\b\.?": "alderman ",
            r"\bgen\b\.?": "general ",
            r"\bsen\b\.?": "senator ",
            r"\brep\b\.?": "representative ",
            r"\bpres\b\.?": "president ",
            r"\brev\b\.?": "reverend ",
            r"\bhon\b\.?": "honorable ",
            r"\basst\b\.?": "assistant ",
            r"\bassoc\b\.?": "associate ",
            r"\blt\b\.?": "lieutenant ",
            r"\bcol\b\.?": "colonel ",
            r"\bjr\b\.?": "junior ",
            r"\bsr\b\.?": "senior ",
            r"\besq\b\.?": "esquire ",
            # also handled by the older ito/cleaners.py english_cleaners
            # (kept in sync so MemoizedNormalize and MemoizedLowercase agree)
            r"\bco\b\.?": "company ",
            r"\bmaj\b\.?": "major ",
            r"\bdrs\b\.?": "doctors ",
            r"\bsgt\b\.?": "sergeant ",
            r"\bltd\b\.?": "limited ",
            r"\bft\b\.?": "fort ",
        }
        # Pronounceable punctuation, unconditionally expanded to words.
        # Also handled by the older ito/cleaners.py expand_punctuation
        # (used by lowercase_normalize), kept in sync for the same reason
        # as self.replacers above.
        self.punctuation_replacers = {
            r"&": " and ",
            r"#": " number ",
            r"\+": " plus ",
            r"%": " percent ",
            r"@": " at ",
            r"=": " equal ",
            r":": " ",
        }
        # british to american english conversion
        self.standardize_spellings = EnglishSpellingNormalizer(
            preserve_case=preserve_case
        )

    def _strip_fillers(self, text: str) -> str:
        """
        Remove filler words from the text.
        """

        def _sub(m: re.Match) -> str:
            pre, post = m.group("pre"), m.group("post")

            before = m.string[: m.start()]
            after = m.string[m.end() :]

            has_word_before = bool(re.search(r"\w", before))
            has_word_after = bool(re.search(r"\w", after))

            # keep exactly one sentence terminator
            # keep a prefix terminator only if something follows
            if pre and pre in ".!?" and has_word_after:
                return pre
            # keep a suffix terminator only if something precedes
            if post and post in ".!?" and has_word_before:
                return post

            # otherwise just a spacer so words don’t stick together
            return " "

        out = self.filler_regex.sub(_sub, text)

        out = re.sub(r"([.!?])\s+\1", r"\1", out)  # ". ."  → "."
        out = re.sub(r"([.!?])\s+[.!?]", r"\1", out)  # ". ?" → "?"
        out = re.sub(r"(?<!\.)\.\.(?!\.)", ".", out)  # ".."  → "."
        return out

    def _collapse_mixed_terminators(self, text: str) -> str:
        """
        Keep deliberate repeats of the same mark (e.g. '...', '!!', '??').
        If the run contains different marks, collapse it to the first one.
        """

        def _repl(m: re.Match) -> str:
            run = m.group(0)
            return run if len(set(run)) == 1 else run[0]

        # two or more terminators, no other chars in between
        return re.sub(r"[.!?]{2,}", _repl, text)

    def _capitalise_after_terminator(self, text: str) -> str:
        """
        Capitalise the first letter after a terminator except for ellipses (...).
        """
        return re.sub(
            r"""
            (?<!\.)                          # not part of "..."
            (?<![A-Z])                       # not part of an acronym e.g. "U.S."
            ([.!?]                           # terminator
            ["’”)\]]*                        # optional closing quotes/bracket
            \s+)                             # one or more spaces
            ([a-z])                          # first letter
            """,
            lambda m: m.group(1) + m.group(2).upper(),
            text,
            flags=re.VERBOSE,
        )

    def _apply_replacers(
        self, s: str, replacers: dict[str, str], preserve_case: bool
    ) -> str:
        for pattern, repl in replacers.items():
            if preserve_case:
                # preserve original casing
                def _fix(m, repl=repl):
                    orig = m.group(0)
                    out = repl
                    if orig.isupper():
                        return out.upper()
                    elif orig[0].isupper() or (
                        orig[0] == "'" and len(orig) > 1 and orig[1].isupper()
                    ):
                        return out.capitalize()
                    return out

                s = re.sub(pattern, _fix, s, flags=re.IGNORECASE)
            else:
                s = re.sub(pattern, repl, s)
        return s

    def __call__(self, s: str) -> str:
        if not self.preserve_case:
            s = s.lower()

        # remove words between brackets
        s = re.sub(r"[«<\[][^»>\]]*[»>\]]", "", s)
        # remove angle quotes
        s = re.sub(r"[«»]", "", s)

        s = self._apply_replacers(s, self.replacers, self.preserve_case)
        s = self._apply_replacers(s, self.punctuation_replacers, self.preserve_case)

        if self.expand_contractions:
            s = self._apply_replacers(s, self.contractions, self.preserve_case)

        s = self._strip_fillers(s)

        # When there's a space before an apostrophe in a contraction
        contractions = ["m", "ve", "ll", "s", "d", "re", "t"]
        contractions = "|".join(contractions)
        s = re.sub(f"\\s+'({contractions})\\b", r"'\1", s, flags=re.IGNORECASE)

        # optional punctuation removal
        if not self.preserve_punctuation:
            # remove commas between digits
            s = re.sub(r"(\d),(\d)", r"\1\2", s)
            # remove periods not followed by a digit
            s = re.sub(r"\.([^0-9]|$)", r" \1", s)
            # keep numeric symbols and apostrophes
            s = remove_symbols_and_diacritics(s, keep=".%$¢€£'")

        s = self.standardize_spellings(s)

        # more symbol cleanup if not preserving punctuation
        # remove prefix/suffix symbols that are not preceded/followed by numbers
        if not self.preserve_punctuation:
            s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
            s = re.sub(r"([^0-9])%", r"\1 ", s)

        # collapse whitespace
        s = re.sub(r"\s+", " ", s)

        # strip space before any terminator
        s = re.sub(r"\s+([.!?])", r"\1", s)

        s = self._collapse_mixed_terminators(s)

        if self.preserve_case:
            s = self._capitalise_after_terminator(s)

        # if nothing but punctuation, return empty string
        if not re.search(r"\w", s):
            return ""

        return s.strip()


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings.

    This class will convert all the British English into American English.
    The source of the english.json file is not retrievable from the
    reference: https://www.tysto.com/uk-us-spelling-list.html
    but it can be found here:
    https://github.com/openai/whisper/blob/main/whisper/normalizers/english.json
    """

    def __init__(self, preserve_case: bool = False):
        mapping_path = os.path.join(os.path.dirname(__file__), "english.json")
        self.mapping = fast_read_json(mapping_path)
        self.preserve_case = preserve_case

        # Build one big regex: \b(word1|word2|...)\b  (ignore case)
        # Sort keys by length so longer spellings match first
        words = sorted(self.mapping.keys(), key=len, reverse=True)
        escaped = [re.escape(w) for w in words]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        self._regex = re.compile(pattern, flags=re.IGNORECASE)

    def __call__(self, s: str) -> str:
        def _repl(m):
            orig = m.group(1)
            amer = self.mapping[orig.lower()]
            if self.preserve_case:
                if orig.isupper():
                    amer = amer.upper()
                elif orig[0].isupper():
                    amer = amer.capitalize()
            return amer

        return self._regex.sub(_repl, s)
