import argparse
import os
from argparse import Namespace
from pathlib import Path

from beartype import beartype
from beartype.typing import Literal

from caiman_asr_train.lhotse.config import Manifest
from caiman_asr_train.lhotse.cuts2shar import cutset_to_shar
from caiman_asr_train.lhotse.cutset import load_jsonl_manifest
from caiman_asr_train.lhotse.json2cuts import json_to_cutset
from caiman_asr_train.lhotse.jsonl2cuts import jsonl_to_cutset


@beartype
def sample_rate_type(x: str) -> int | Literal["original"]:
    if x == "original":
        return "original"
    try:
        return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be an integer or 'original'.")


@beartype
def _parse() -> Namespace:
    # === Architecture of the CLI tool

    parent_parser = argparse.ArgumentParser(
        "CLI tool for working with Lhotse CutSets and Legacy JSONs"
    )

    subparsers = parent_parser.add_subparsers(
        title="subcommands", dest="command", required=True
    )

    # === For converting legacy json to cutset

    json2cuts_parser = subparsers.add_parser(
        "json2cuts",
        help="Convert legacy json to a Lhotse CutSet",
    )

    # Positional arguments

    json2cuts_parser.add_argument(
        "legacy_json",
        type=Path,
        help="Path to input legacy JSON file",
    )
    json2cuts_parser.add_argument(
        "cut_dir",
        type=Path,
        help="A dir to store output cutsets with the same base name as the input jsons",
    )

    # Flags

    json2cuts_parser.add_argument(
        "--text-keys",
        nargs="+",
        type=str,
        default=[],
        help="List of additional json keys to use as transcripts",
    )
    json2cuts_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional data directory if the cutset contains relative paths",
    )

    # === For converting granary jsonl to cutset

    jsonl2cuts_parser = subparsers.add_parser(
        "jsonl2cuts",
        help="Convert granary jsonl to a Lhotse CutSet",
    )

    # Positional arguments

    jsonl2cuts_parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to input granary JSONL file",
    )
    jsonl2cuts_parser.add_argument(
        "cut_dir",
        type=Path,
        help="A dir to store output cutsets with the same base name as the input jsonls",
    )

    # Flags

    jsonl2cuts_parser.add_argument(
        "--text-keys",
        nargs="+",
        type=str,
        default=[],
        help="List of additional json keys to use as transcripts",
    )

    jsonl2cuts_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional data directory if the cutset contains relative paths",
    )

    # === For saving cutset to shar

    cuts2shar_parser = subparsers.add_parser(
        "cuts2shar",
        help="Convert a CutSet to SHAR archive",
    )

    # Positional arguments

    cuts2shar_parser.add_argument(
        "cutset_json",
        type=Path,
        help="Path to input CutSet JSONL file",
    )
    cuts2shar_parser.add_argument(
        "shar_dir",
        type=Path,
        help="Path to output SHAR directory",
    )

    # Flags

    cuts2shar_parser.add_argument(
        "--compression",
        type=str,
        choices=["wav", "flac", "mp3", "opus", "original"],
        required=True,
        help="The audio compression format to use, 'original' keeps the input format",
    )
    cuts2shar_parser.add_argument(
        "--sample-rate",
        type=sample_rate_type,
        required=True,
        help="If specified, all audio will be resampled to this rate",
    )
    cuts2shar_parser.add_argument(
        "--shard-size",
        type=int,
        default=4096,
        help="Number of cuts per shard",
    )
    cuts2shar_parser.add_argument(
        "--num-jobs",
        type=int,
        default=4,
        help="Number of workers for parallel write",
    )
    cuts2shar_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional data directory if the cutset contains relative paths",
    )

    return parent_parser.parse_args()


@beartype
def _cuts2shar(args: Namespace):
    # Log straight to stdout
    print(f"Converting:\n\t{args.cutset_json}\nto shar:\n\t{args.shar_dir}")

    cutset = load_jsonl_manifest(Manifest.build(args.cutset_json, args.data_dir))

    cutset_to_shar(
        cutset=cutset,
        out_path=args.shar_dir,
        shard_size=args.shard_size,
        num_jobs=args.num_jobs,
        compression=args.compression,
        sample_rate=args.sample_rate,
        verbose=True,
    )


@beartype
def _json2cuts(args: Namespace):
    if not args.legacy_json.suffix == ".json":
        raise ValueError("Input legacy file must be json files")

    in_file = args.legacy_json
    out_file = args.cut_dir / in_file.with_suffix(".cuts.jsonl.gz").name

    if out_file.exists():
        raise FileExistsError(f"The output cutset file {out_file} already exists.")

    print(f"Converting:\n\t{in_file}\nto:\n\t{out_file}")

    cuts = json_to_cutset(
        Manifest.build(in_file, args.data_dir),
        text_keys=args.text_keys,
        verbose=True,
    )

    print("Shuffling and saving...")

    cuts = cuts.to_eager().shuffle()
    os.makedirs(out_file.parent, exist_ok=True)
    cuts.to_file(out_file)


@beartype
def _jsonl2cuts(args: Namespace):
    if not args.jsonl_path.suffix == ".jsonl":
        raise ValueError("Input manifest file must be jsonl file")

    in_file = args.jsonl_path
    out_file = args.cut_dir / in_file.with_suffix(".cuts.jsonl.gz").name

    if out_file.exists():
        raise FileExistsError(f"The output cutset file {out_file} already exists.")

    print(f"Converting:\n\t{in_file}\nto:\n\t{out_file}")

    cuts = jsonl_to_cutset(
        Manifest.build(in_file, args.data_dir),
        text_keys=args.text_keys,
        verbose=True,
    )

    print("Shuffling and saving...")

    cuts = cuts.to_eager().shuffle()
    os.makedirs(out_file.parent, exist_ok=True)
    cuts.to_file(out_file)


@beartype
def _main(args: Namespace):
    match args.command:
        case "json2cuts":
            _json2cuts(args)
        case "jsonl2cuts":
            _jsonl2cuts(args)
        case "cuts2shar":
            _cuts2shar(args)
        case unk:
            raise ValueError(f"Unknown command {unk}")


if __name__ == "__main__":
    _main(_parse())
