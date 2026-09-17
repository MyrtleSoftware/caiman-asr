# Copyright (c) 2017 Keith Ito
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.
"""from https://github.com/keithito/tacotron"""
import string


def punctuation_map(labels):
    # Punctuation to remove
    punctuation = string.punctuation
    # The following punctuation won't be removed:
    punctuation = punctuation.replace("+", "")
    punctuation = punctuation.replace("&", "")
    punctuation = punctuation.replace("@", "")
    punctuation = punctuation.replace("%", "")
    punctuation = punctuation.replace(":", "")
    # If a punctuation symbol is inside the vocab, do not remove from text
    for label in labels:
        punctuation = punctuation.replace(label, "")
    # Turn all punctuation to whitespace
    table = str.maketrans(punctuation, " " * len(punctuation))
    return table
