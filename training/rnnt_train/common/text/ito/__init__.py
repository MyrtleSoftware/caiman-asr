# Copyright (c) 2017 Keith Ito
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.
""" from https://github.com/keithito/tacotron """
import string

from . import cleaners


def _clean_text(text, cleaner_names, *args):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text, *args)
    return text


def punctuation_map(labels):
    # Punctuation to remove
    punctuation = string.punctuation
    # The following punctuation won't be removed:
    punctuation = punctuation.replace("+", "")
    punctuation = punctuation.replace("&", "")
    punctuation = punctuation.replace("@", "")
    punctuation = punctuation.replace("%", "")
    # If a punctuation symbol is inside the vocab, do not remove from text
    for label in labels:
        punctuation = punctuation.replace(label, "")
    # Turn all punctuation to whitespace
    table = str.maketrans(punctuation, " " * len(punctuation))
    return table
