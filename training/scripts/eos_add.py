#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from functools import partial
from glob import glob
from pathlib import Path

from beartype import beartype
from beartype.typing import Dict, Optional
from lhotse import CutSet, SupervisionSegment
from tqdm import tqdm

from caiman_asr_train.data.segment_manifest import ManifestSegmenter
from caiman_asr_train.data.text.is_tag import is_tag
from caiman_asr_train.data.utils import validate_args


def parse_args():
    parser = ArgumentParser(
        description="Utility to segment transcripts with an EOS token using an SAT model."
    )

    parser.add_argument(
        "--manifests",
        type=str,
        nargs="+",
        required=True,
        help="List of shar manifests to process.",
    )
    parser.add_argument(
        "--output_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of where to save the modified transcripts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    parser.add_argument(
        "--transcript_field",
        type=str,
        default=None,
        help="List of fields in the manifests to add EOS to (default: text)",
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Use the CPU to segment the transcripts"
    )

    parser.add_argument(
        "--eos_token",
        type=str,
        default="<EOS>",
        help="The token to use as the end of segment marker.",
    )

    args = parser.parse_args()
    return args


@beartype
def _transform_text(text: str, transforms: Dict[str, str]) -> str:
    return transforms.get(text, text)


@beartype
def _transform_supervision(
    sup: SupervisionSegment,
    transforms: Dict[str, str],
    transcript_field: Optional[str] = None,
) -> SupervisionSegment:
    if transcript_field:
        text = sup.custom[transcript_field]
        sup = sup.with_custom(transcript_field, transforms[text])
    else:
        sup = sup.transform_text(partial(_transform_text, transforms=transforms))
    return sup


@beartype
def _process_shar_manifest(
    manifest: str,
    output_dir: str,
    segmenter: ManifestSegmenter,
    transcript_field: Optional[str] = None,
):
    """Process all cuts within each shar manifest"""

    for cut_path in tqdm(sorted(glob(f"{manifest}/*.jsonl.gz"))):

        cutset_id = Path(cut_path).name
        cutset = CutSet.from_jsonl(cut_path)

        output_cutset_path = str(Path(output_dir) / cutset_id)

        transforms = segmenter.add_eos_to_cutset(
            cutset_id, cutset, transcript_field=transcript_field
        )
        cutset = cutset.map_supervisions(
            partial(
                _transform_supervision,
                transforms=transforms,
                transcript_field=transcript_field,
            )
        )
        cutset.to_file(output_cutset_path)


@beartype
def _eos_add(args: Namespace):
    """Add EOS for all input Shar manifests"""

    segmenter = ManifestSegmenter(eos_token=args.eos_token, use_cuda=not args.no_cuda)

    for manifest, output_dir in zip(args.manifests, args.output_dirs):
        if Path(output_dir).exists() and not args.overwrite:
            raise FileExistsError(
                f"Output_dir {output_dir} already exists. "
                "Use --overwrite to overwrite the existing files."
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Adding EOS for: {manifest}")
        _process_shar_manifest(manifest, output_dir, segmenter, args.transcript_field)


@beartype
def main(args: Namespace):

    # Validate eos_token and inputs
    assert is_tag(args.eos_token), "EOS token must be in form: '<[a-zA-Z]+>'"
    validate_args(args)

    _eos_add(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
