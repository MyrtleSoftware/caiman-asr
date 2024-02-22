#!/usr/bin/env python3

import argparse
import json
import re

from scipy.io import wavfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CommonVoice tsv files to RNNT json files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="CommonVoice language-specific input dir",
    )
    parser.add_argument(
        "--input_tsv", type=str, default="./", help="name of the input .tsv file"
    )
    parser.add_argument(
        "--output_json", type=str, default="./", help="name of the output json file"
    )
    args = parser.parse_args()

    args.input_dir = args.input_dir.rstrip("/")

    tsvfn = args.input_dir + "/" + args.input_tsv
    jsonfn = args.input_dir + "/" + args.output_json

    # There are some non Roman characters in the CommonVoice Version 10.0 dataset
    # (eg Japanese, Greek, etc).
    # Rather than making a list of these characters, which would not be future-proof,
    # make a list of permitted characters instead.
    # The complete list of puctuation in Version 10.0 is . ,'"-?!:’;”“‘—()–/`]_[=
    #
    # So what transcripts should be allowed?
    #
    # Reject &/=%~_ because they might be spoken.  There are not many of these.
    # Reject ’ because sometimes it is an apostrophe but sometimes a quote.
    # Reject ‘ because although it's always a quote it pairs with the ambiguous ’
    # Reject ` because there are only about 10 of them and while most are apostrophes
    # one is just wrong.
    # Allow and retain  a-z '
    # Allow and then drop (remove)  .,"?!:;”“()][  since these are silent punctuation.
    # Allow and then replace with space  —–-  since these are often inter-word hyphens.

    allow = re.compile(r"^[-a-z\. ,'\"?!:;”“—()–\]\[]+$")
    drop = re.compile(r'[\.,"?!:;”“()\]\[]')
    replace = re.compile(r"[-—–]")
    space = re.compile(r"[ ]+")

    lst = []
    with open(tsvfn, "r") as f:
        header = f.readline()
        for line in f:
            bits = line.split("\t")
            trans = bits[2].lower().strip()
            if not allow.match(trans):
                continue
            # substitute the empty string for any character in regex drop
            trans = drop.sub("", trans)
            # substitute a space for any characters in regex replace
            trans = replace.sub(" ", trans)
            # remove consecutive spaces and remove any leading or trailing spaces
            trans = space.sub(" ", trans).strip()
            # prepare wav filenames and load the speech data
            wavfn = "wav_clips/" + bits[1].replace(".mp3", ".wav")
            fullwavfn = args.input_dir + "/" + wavfn
            sr, data = wavfile.read(fullwavfn)
            # prepare the dictionary for this utterance
            d = dict()
            d["transcript"] = trans
            d["files"] = []
            f = dict()
            if len(data.shape) == 1:
                f["channels"] = 1
            else:
                f["channels"] = data.shape[1]
            f["sample_rate"] = float(sr)
            f["duration"] = data.shape[0] / sr
            f["num_samples"] = data.shape[0]
            f["fname"] = wavfn
            d["files"].append(f)
            d["original_duration"] = data.shape[0] / sr
            d["original_num_samples"] = data.shape[0]
            lst.append(d)

    with open(jsonfn, "w") as o:
        json.dump(lst, o, indent=2)
