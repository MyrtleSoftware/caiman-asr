#!/usr/bin/env python3
import argparse
from pathlib import Path

import orjson

from caiman_asr_train.data.segment_manifest import add_eos_to_manifest_avoid_empty
from caiman_asr_train.data.text.is_tag import is_tag
from caiman_asr_train.utils.fast_json import fast_read_json


def make_parser():
    parser = argparse.ArgumentParser(
        description="Utility to segment transcripts with an EOS token using an SAT model."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the manifests.",
    )

    parser.add_argument(
        "--manifests",
        type=str,
        nargs="+",
        required=True,
        help="List of manifests to process.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Where to save the modified transcripts.",
        required=True,
    )

    parser.add_argument(
        "--out_manifests",
        type=str,
        nargs="+",
        help="Optional list of output manifests to save the modified transcripts.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    parser.add_argument(
        "--eos_token",
        type=str,
        default="<EOS>",
        help="The token to use as the end of segment marker.",
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Use the CPU to segment the transcripts"
    )

    return parser


def main(args):
    for manifest, out_manifest in zip(args.manifests, args.out_manifests):
        ifile = Path(args.data_dir) / manifest
        ofile = Path(args.output_dir) / out_manifest

        if ofile.exists() and not args.overwrite:
            print(f"Skipping {ofile}, use --overwrite to overwrite.")
            continue

        # Check if the output directory exists
        if not ofile.parent.exists():
            print(f"Skipping {ofile}, the output directory does not exist.")
            continue

        print(f"Reading {ifile}")

        manifest = fast_read_json(ifile)

        print(f"Processing {ifile}")

        out = add_eos_to_manifest_avoid_empty(
            manifest, args.eos_token, not args.no_cuda
        )
        print(f"Writing {ofile}")

        with open(ofile, "wb") as f:
            f.write(orjson.dumps(out, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    assert is_tag(args.eos_token), "EOS token must be in form: '<[a-zA-Z]+>'"

    args.manifests = [Path(m) for m in args.manifests]

    if args.out_manifests is not None:
        assert len(args.manifests) == len(args.out_manifests)
    else:
        args.out_manifests = [m.with_suffix(".eos.json") for m in args.manifests]

    main(args)
