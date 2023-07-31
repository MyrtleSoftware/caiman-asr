#!/usr/bin/env python3

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023, Myrtle Software Limited. All rights reserved.
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

import argparse
import string

import inflect

from rnnt_train.common.text.ito import _clean_text, punctuation_map

CHARSTR = " '" + string.ascii_lowercase
DEFAULT_CHARSET = set(CHARSTR)
DEFAULT_PUNCT_MAP = punctuation_map(DEFAULT_CHARSET)


def normalize(s, quiet=False, charset=None, punct_map=None) -> str:
    """Normalizes string.

    Example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'
    """
    if charset is None:
        charset = DEFAULT_CHARSET
        punct_map = DEFAULT_PUNCT_MAP
    if punct_map is None:
        punct_map = punctuation_map(charset)

    try:
        text = _clean_text(s, ["english_cleaners"], punct_map).strip()
    except (ValueError, inflect.NumOutOfRangeError, IndexError) as err:
        if not quiet:
            print(f"Expected {err=}, {type(err)=}")
            print(f"WARNING: Normalizing this line failed: {s}")
            print("Omitting the line that couldn't be normalized")
            print()
        return ""
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        print(f"ERROR: Normalizing this line failed: {s}")
        raise
    else:
        return "".join([tok for tok in text if all(t in charset for t in tok)])


def normalize_list(transcripts, quiet=False):
    return [normalize(t, quiet) for t in transcripts]


def normalize_by_line(text, quiet=False):
    """Split text at \n, clean each line,
    and then glue together with spaces.
    Hence we can still normalize most of
    a document even if one line is unnormalizable
    and needs to be deleted"""
    return " ".join(normalize_list(text.split("\n"), quiet))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalizes a file to just a-z, space, and '. Handles numbers acceptably"
    )
    parser.add_argument(
        "file",
        type=str,
        help="The file to normalize",
        nargs="+",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Edit the file in place instead of making a new file",
    )
    args = parser.parse_args()
    for filename in args.file:
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        suffix = ".normalized"
        if args.in_place:
            suffix = ""
        with open(filename + suffix, "w") as f:
            f.write("\n".join(normalize_list(lines)))
