#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

import yaml
from beartype import beartype
from beartype.typing import Optional
from tqdm import tqdm

from caiman_asr_train.data.text.normalizers import standardize_text
from caiman_asr_train.data.text.preprocess import (
    generate_symbols_mask,
    mask_user_symbols,
    unmask_user_symbols,
)
from caiman_asr_train.utils.fast_json import fast_read_json, fast_write_json
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


def process_entry(
    entry: dict,
    transcript_fields: list[str],
    preserve_case: bool = False,
    preserve_punctuation: bool = False,
    expand_contractions: bool = True,
    symbols_mask: Optional[dict[str, str]] = None,
):
    """Process a single manifest entry by standardizing its transcripts."""
    for field in transcript_fields:
        if field in entry:
            transcript = mask_user_symbols(entry[field], symbols_mask)
            transcript = standardize_text(
                transcript,
                preserve_case=preserve_case,
                preserve_punctuation=preserve_punctuation,
                expand_contractions=expand_contractions,
            )
            transcript = unmask_user_symbols(transcript, symbols_mask)
            entry[f"{field}-standardized"] = transcript
    return entry


@beartype
def standardize_manifest_transcripts(
    manifest: list[dict],
    transcript_fields: list[str],
    num_workers: Optional[int] = None,
    preserve_case: bool = False,
    preserve_punctuation: bool = False,
    expand_contractions: bool = True,
    symbols_mask: Optional[dict[str, str]] = None,
    min_entries_per_process: int = 50,
) -> list[dict]:
    """
    Load a manifest, standardize specified transcript fields,
    and add standardized transcripts as new fields.

    Parameters
    ----------
    manifest
        List of dictionaries, each representing a manifest entry.
    transcript_fields
        List of transcript field names to standardize
    num_workers
        Number of worker processes to use for parallel processing.
        If None, uses CPU count.
    preserve_case
        Whether to preserve the original case of the text
    preserve_punctuation
        Whether to preserve the original punctuation of the text
    symbols_mask
        Dictionary mapping random strings to symbols that should be preserved
    min_entries_per_process
        Minimum number of entries per process to use multiprocessing

    Returns
    -------
    list[dict]
        The manifest with standardized transcripts added
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    entries_num = len(manifest)
    print(f"Standardizing {entries_num} entries")

    process_func = partial(
        process_entry,
        transcript_fields=transcript_fields,
        preserve_case=preserve_case,
        preserve_punctuation=preserve_punctuation,
        expand_contractions=expand_contractions,
        symbols_mask=symbols_mask,
    )

    # Skip multiprocessing for small datasets or single CPU
    if min_entries_per_process * num_workers >= entries_num or num_workers < 2:
        print("Using single-processing (dataset too small or only one CPU)")
        results = []
        for entry in tqdm(manifest, total=entries_num):
            results.append(process_func(entry))
    else:
        print(f"Using {num_workers} processes")
        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        process_func,
                        manifest,
                        chunksize=max(10, entries_num // (num_workers * 10)),
                    ),
                    total=entries_num,
                )
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standardize transcripts in manifest files"
    )
    parser.add_argument(
        "--manifests",
        type=str,
        nargs="+",
        required=True,
        help="Absolute paths to manifest JSON files",
    )
    parser.add_argument(
        "--transcript-fields",
        type=str,
        nargs="+",
        default=["transcript"],
        help="Names of transcript fields to standardize (default: ['transcript'])",
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Preserve the original case of the text",
    )
    parser.add_argument(
        "--preserve-punctuation",
        action="store_true",
        help="Preserve the original punctuation of the text",
    )
    parser.add_argument(
        "--no-expand-contractions",
        action="store_false",
        default=True,
        help="Do not expand contractions (e.g. 'don't' -> 'do not')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save standardized manifests "
        "(default: same directory as each input)",
    )
    parser.add_argument(
        "--output-filenames",
        type=str,
        nargs="+",
        default=None,
        help="Filenames to save standardized manifests "
        "(default: same as input plus '_standardized'). "
        "Must have same length as --manifests if provided.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="/workspace/training/configs/testing-1023sp_run.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: CPU count)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = parser.parse_args()

    manifest_paths = [Path(manifest) for manifest in args.manifests]

    missing_files = [path for path in manifest_paths if not path.exists()]
    if missing_files:
        print("The following manifest files do not exist:")
        for path in missing_files:
            print(f"  {path}")
        sys.exit(1)

    num_jsons = len(manifest_paths)
    if args.output_filenames:
        assert len(args.output_filenames) == num_jsons, (
            f"Number of output filenames ({len(args.output_filenames)}) "
            f"must match number of manifest files ({num_jsons})"
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = None

    output_paths = []
    for i, manifest_path in enumerate(manifest_paths):
        if output_dir is not None:
            current_output_dir = output_dir
        else:
            current_output_dir = manifest_path.parent

        if args.output_filenames:
            output_path = current_output_dir / args.output_filenames[i]
        else:
            output_path = (
                current_output_dir
                / f"{manifest_path.stem}_standardized{manifest_path.suffix}"
            )

        output_paths.append(output_path)

    if not args.overwrite:
        existing_files = [path for path in output_paths if path.exists()]
        if existing_files:
            print("The following output files already exist:")
            for path in existing_files:
                print(f"  {path}")
            print(
                "Use --overwrite to overwrite, or specify different --output-filenames."
            )
            sys.exit(1)

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    _, user_symbols = get_all_user_tokens(model_config)
    symbols_mask = generate_symbols_mask(user_symbols)

    print(f"Processing {num_jsons} manifest file(s)")

    for i, (manifest_path, output_path) in enumerate(zip(manifest_paths, output_paths)):
        print(
            (
                f"\n--- Processing manifest {i+1}/{num_jsons}: "
                f"{manifest_path.name} ---"
            )
        )

        manifest = fast_read_json(manifest_path)

        standardized_manifest = standardize_manifest_transcripts(
            manifest,
            transcript_fields=args.transcript_fields,
            num_workers=args.num_workers,
            preserve_case=args.preserve_case,
            preserve_punctuation=args.preserve_punctuation,
            expand_contractions=args.no_expand_contractions,
            symbols_mask=symbols_mask,
        )

        print(f"Saving standardized manifest to {output_path}")
        fast_write_json(standardized_manifest, output_path)

    print(f"\nCompleted processing all {num_jsons} manifest files!")


if __name__ == "__main__":
    main()
