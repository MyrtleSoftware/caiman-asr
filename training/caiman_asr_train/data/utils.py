from argparse import Namespace
from pathlib import Path

from beartype import beartype
from beartype.typing import List


@beartype
def _validate_manifests(manifests: List[str]):
    manifest_paths = [Path(manifest) for manifest in manifests]
    missing_files = [path for path in manifest_paths if not path.exists()]
    if missing_files:
        err_string = "The following manifest files do not exist:"
        for path in missing_files:
            err_string += f"  {path}"
        raise FileNotFoundError(err_string)


@beartype
def validate_args(args: Namespace):
    _validate_manifests(args.manifests)
    num_manifests = len(args.manifests)

    assert num_manifests == len(args.output_dirs), (
        f"Number of output directories ({len(args.output_dirs)})"
        f" must match number of manifests({num_manifests})"
    )
