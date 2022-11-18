#!/usr/bin/env python3

# Copyright (c) 2022 Myrtle.ai
# theodore@myrtle.ai, September 2022

import json
import time
import argparse
from scipy.io import wavfile


parser = argparse.ArgumentParser(
    description="Given one or more wav files, makes a json file that can be fed into the RNN-T to make predictions"
)
parser.add_argument(
    "file_path",
    type=str,
    help="wav file",
    nargs="+",
)
parser.add_argument(
    "--output_path",
    type=str,
    help="the path of the output json file",
    default="./" + str(time.time()) + ".json",
)
parser.add_argument(
    "--transcript_file",
    type=str,
    help="A file with one transcript per line, matching the order of the given files",
    default=None,
)


def get_json_dictionary_from_path(filename, transcript):
    # This is based on Rob's convert_commonvoice.py
    sr, data = wavfile.read(filename)
    d = dict()
    d["transcript"] = transcript
    d["files"] = []
    f = dict()
    if len(data.shape) == 1:
        f["channels"] = 1
    else:
        f["channels"] = data.shape[1]
    f["sample_rate"] = float(sr)
    f["duration"] = data.shape[0] / sr
    f["num_samples"] = data.shape[0]
    f["fname"] = filename
    d["files"].append(f)
    d["original_duration"] = data.shape[0] / sr
    d["original_num_samples"] = data.shape[0]
    return d


args = parser.parse_args()
if args.transcript_file != None:
    with open(args.transcript_file) as f:
        transcript_list = f.read().splitlines()
else:
    transcript_list = ["dummy" for path in args.file_path]

assert len(transcript_list) == len(args.file_path)

l = [
    get_json_dictionary_from_path(args.file_path[i], transcript_list[i])
    for i in range(len(args.file_path))
]

with open(args.output_path, "w") as o:
    json.dump(l, o, indent=2)
