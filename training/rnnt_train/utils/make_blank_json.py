#!/usr/bin/env python3

# Copyright (c) 2022 Myrtle.ai
# theodore@myrtle.ai, September 2022
import argparse
import json
import time

parser = argparse.ArgumentParser(
    description="Given one or more wav files, makes a blank json file that can be fed into the RNN-T to make predictions"
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


def get_json_dictionary_from_path(filename):
    # This is based on Rob's convert_commonvoice.py
    outer = {}
    outer["transcript"] = "dummy"
    outer["files"] = []
    inner = {}
    inner["fname"] = filename
    outer["files"].append(inner)
    outer["original_duration"] = 0.0
    return outer


args = parser.parse_args()
l = [get_json_dictionary_from_path(path) for path in args.file_path]

with open(args.output_path, "w") as o:
    json.dump(l, o, indent=2)
