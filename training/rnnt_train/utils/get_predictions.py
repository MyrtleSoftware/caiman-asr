#!/usr/bin/env python3

# Copyright (c) 2022 Myrtle.ai
# theodore@myrtle.ai, September 2022
import argparse
import os
import time

parser = argparse.ArgumentParser(
    description="Given one or more wav files, feeds them into the RNN-T and prints predictions. Default limit of 30000 tokens ~ 100 minutes. Will overwrite /results/preds.txt"
)
parser.add_argument(
    "file_path",
    type=str,
    help="wav file",
    nargs="+",
)
parser.add_argument(
    "--token-limit",
    help="the maximum number of tokens the RNN-T decodes",
    default=30000,
    type=int,
)
parser.add_argument(
    "--quiet",
    help="hide the model's logs",
    action="store_true",
)
parser.add_argument(
    "--data-dir",
    help="The DATA_DIR variable fed to valCPU.sh. It's best to leave it at the default of / unless the paths to the wav file are relative",
    default="/",
    type=str,
)
parser.add_argument(
    "--checkpoint",
    help="The path to the RNN-T checkpoint. valCPU.sh will by default choose /results/RNN-T_best_checkpoint.pt",
    default=None,
    type=str,
)
args = parser.parse_args()
os.system("mkdir -p /tmp/predicting")
os.system("rm -f /results/preds.txt")
json_name = "/tmp/predicting/" + str(time.time()) + ".json"


def add_quotes(a_string):
    return '"' + a_string + '"'


os.system(
    "python rnnt_train/utils/make_blank_json.py "
    + " ".join(list(map(add_quotes, args.file_path)))
    + " --output_path "
    + json_name
)
command = ""
if args.checkpoint != None:
    command += f"CHECKPOINT={args.checkpoint} "
command += (
    "DUMP_PREDS=true MAX_SYMBOL_PER_SAMPLE="
    + str(args.token_limit)
    + " DATA_DIR="
    + str(args.data_dir)
    + " VAL_MANIFESTS="
    + json_name
    + " ./scripts/valCPU.sh "
)
if args.quiet:
    command += " > /dev/null"
os.system(command)
os.system("cat /results/preds.txt")
