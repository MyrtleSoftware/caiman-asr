from pathlib import Path

from beartype import beartype
from beartype.typing import Literal
from lhotse import CutSet


@beartype
def cutset_to_shar(
    cutset: CutSet,
    out_path: Path,
    shard_size: int = 1000,
    num_jobs: int = 4,
    verbose: bool = True,
    compression: Literal["wav", "flac", "mp3", "opus", "original"] = "flac",
    sample_rate: int | Literal["original"] = "original",
) -> dict[str, list[str]]:
    """
    Write a CutSet to a SHAR archive.

    Arguments:
        cutset: The CutSet to write.
        out_path: The output directory for the SHAR files.
        shard_size: Number of cuts per shard.
        num_jobs: Number of parallel jobs for writing.
        verbose: Whether to display a progress bar during writing.
        compression: The audio compression format to use.
        sample_rate: Optional resampling rate for the audio written files.
    """

    # This raises an error if out_path is a file
    out_path.mkdir(parents=True, exist_ok=True)

    if any(out_path.iterdir()):
        raise ValueError(
            f"Output path {out_path} must be an empty directory to avoid overwriting files."
        )

    if sample_rate != "original":
        cutset = cutset.resample(sample_rate)

    return cutset.to_shar(
        out_path,
        fields={"recording": compression},
        shard_size=shard_size,
        num_jobs=num_jobs,
        verbose=verbose,
    )
