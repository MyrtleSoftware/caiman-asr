#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from functools import partial
from glob import glob
from pathlib import Path

import yaml
from beartype import beartype
from beartype.typing import Generator, List, Optional
from lhotse import CutSet, SupervisionSegment
from tqdm import tqdm

from caiman_asr_train.data.text.normalizers import standardize_text
from caiman_asr_train.data.text.symbols_mask import (
    generate_symbols_mask,
    mask_user_symbols,
    unmask_user_symbols,
)
from caiman_asr_train.data.utils import validate_args
from caiman_asr_train.lhotse.warn import lhotse_warn
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens

MIN_SEGS_PER_PROCESS = 50


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize transcripts in lhotse shar manifest files"
    )

    parser.add_argument(
        "--manifests",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the lhotse Shar manifests",
    )
    parser.add_argument(
        "--output-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directory to save standardized manifests "
        "(default: <manifest_parent_dir>/standardized)",
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
        "--expand-contractions",
        action="store_true",
        help="Expand contractions (e.g. 'don't' -> 'do not')",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
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
    parser.add_argument(
        "--transcript-fields",
        type=str,
        nargs="+",
        help="Names of transcript fields to standardize (default: text)",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        required=True,
        help="ISO code of the language of the transcripts to be standardized",
    )
    args = parser.parse_args()

    return args


def get_user_symbols_mask(model_config: str) -> Optional[dict[str, str]]:
    """
    Generate mask for user symbols in model_config
    """
    with open(model_config, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    _, user_symbols = get_all_user_tokens(model_config)
    symbols_mask = generate_symbols_mask(user_symbols)

    return symbols_mask


@beartype
class SharManifestIter:
    """
    Iterator for all cuts of a Lhotse Shar Manifest
    """

    def __init__(self, manifest: str):
        self.manifest = sorted(glob(f"{manifest}/*.jsonl.gz"))

    def __iter__(self) -> Generator[str]:
        for cut_path in self.manifest:
            yield cut_path

    def __len__(self) -> int:
        return len(self.manifest)


@beartype
class LhotseManifestStandardizer:
    """
    Standardizer for Lhotse Shar Manifests
    """

    def __init__(
        self,
        model_config: str,
        num_workers: Optional[int] = None,
        preserve_case: bool = False,
        preserve_punctuation: bool = False,
        expand_contractions: bool = False,
        overwrite: bool = False,
        standardized_suffix: str = "standardized",
        lang_code: str = "en",
    ):
        self.preserve_case = preserve_case
        self.preserve_punctuation = preserve_punctuation
        self.expand_contractions = expand_contractions
        self.standardized_suffix = standardized_suffix
        self.overwrite = overwrite
        self.lang_code = lang_code

        self.symbols_mask = get_user_symbols_mask(model_config)

        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = mp.cpu_count()
        self.min_segs_per_process = MIN_SEGS_PER_PROCESS

    def _process_text(self, transcript: str) -> str:
        transcript = mask_user_symbols(transcript, self.symbols_mask)
        transcript = standardize_text(
            transcript,
            preserve_case=self.preserve_case,
            preserve_punctuation=self.preserve_punctuation,
            expand_contractions=self.expand_contractions,
            lang_code=self.lang_code,
        )
        transcript = unmask_user_symbols(transcript, self.symbols_mask)

        return transcript

    def process_supervision(
        self, sup: SupervisionSegment, transcript_fields: Optional[List[str]] = None
    ) -> SupervisionSegment:
        """
        Standardize each of the transcripts (either indicated by transcript_fields
        or default "text") in each supervision segment and save them to a custom
        {transcript_field}-standardized field. For example:

        sup = SupervisionSegment(
            id='sup-id', recording_id='rec-id', start=0, duration=30.0, channel=0,
            text='Dr. Smith visited the cafe.',
            language=None, speaker=None, gender=None, alignment=None)

        will be standardized (with default options) as:

        sup = SupervisionSegment(
            id='sup-id', recording_id='rec-id', start=0, duration=30.0, channel=0,
            text='Dr. Smith visited the cafe.',
            custom={'transcript-standardized': 'doctor smith visited the cafe'}
            language=None, speaker=None, gender=None, alignment=None)

        """
        if transcript_fields:
            for transcript_field in transcript_fields:
                if sup.has_custom(transcript_field):
                    transcript = self._process_text(sup.custom[transcript_field])
                    sup = sup.with_custom(
                        f"{transcript_field}-{self.standardized_suffix}", transcript
                    )
                else:
                    lhotse_warn(
                        f"{transcript_field} not found in SupervisionSegment. Skipping."
                    )
        else:
            transcript = self._process_text(sup.text)
            sup = sup.with_custom(f"transcript-{self.standardized_suffix}", transcript)

        return sup

    def process_cutset(
        self,
        cutset_path: str,
        output_dir: str,
        transcript_fields: Optional[List[str]] = None,
    ):
        """
        Standardize each supversion for each cutset and save to output directory
        """
        output_cutset_path = str(Path(output_dir) / Path(cutset_path).name)

        cutset = CutSet.from_jsonl(cutset_path)
        cutset = cutset.map_supervisions(
            partial(self.process_supervision, transcript_fields=transcript_fields)
        )
        cutset.to_file(output_cutset_path)

    def standardize(
        self,
        manifest: SharManifestIter,
        output_dir: str,
        transcript_fields: Optional[List[str]] = None,
    ):
        """
        Standardize cutsets for a shar manifest in parallel
        """

        print(f"Using {self.num_workers} processes")
        with mp.Pool(processes=self.num_workers) as pool, tqdm(
            total=len(manifest)
        ) as pbar:
            for _ in pool.imap(
                partial(
                    self.process_cutset,
                    output_dir=output_dir,
                    transcript_field=transcript_fields,
                ),
                manifest,
            ):
                pbar.update()
                pbar.refresh()


@beartype
def standardize_manifests(args: argparse.Namespace):
    """
    Standardize all input manifests
    """

    lhotse_standardizer = LhotseManifestStandardizer(
        preserve_case=args.preserve_case,
        preserve_punctuation=args.preserve_punctuation,
        expand_contractions=args.expand_contractions,
        overwrite=args.overwrite,
        model_config=args.model_config,
        num_workers=args.num_workers,
        lang_code=args.lang_code,
    )

    for manifest, output_dir in zip(args.manifests, args.output_dirs):
        if Path(output_dir).exists() and not args.overwrite:
            raise FileExistsError(
                f"Output_dir {output_dir} already exists. "
                "Use --overwrite to overwrite the existing files."
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Processing: {manifest}")
        lhotse_standardizer.standardize(
            SharManifestIter(manifest), output_dir, args.transcript_fields
        )


@beartype
def main(args: argparse.Namespace):
    validate_args(args)
    standardize_manifests(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
