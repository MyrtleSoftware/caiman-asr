#!/usr/bin/env python3
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import Optional

from caiman_asr_train.utils.iter import lmap


@beartype
@dataclass
class AbsoluteManifestRatios:
    ratios: list[float]


@beartype
@dataclass
class RelativeManifestRatios:
    ratios: list[float]


@beartype
@dataclass
class CanaryManifestRatios:
    exponent: float


ManifestRatios = (
    AbsoluteManifestRatios | RelativeManifestRatios | CanaryManifestRatios | None
)


@beartype
def build_manifest_ratios(
    train_manifest_ratios: Optional[list[float]],
    relative_train_manifest_ratios: Optional[list[float]],
    canary_manifest_exponent: Optional[float],
) -> ManifestRatios:
    it = [
        train_manifest_ratios is not None,
        relative_train_manifest_ratios is not None,
        canary_manifest_exponent is not None,
    ]

    if it.count(True) > 1:
        raise ValueError("At most one kind of manifest mode should be set")

    if train_manifest_ratios is not None:
        return AbsoluteManifestRatios(ratios=train_manifest_ratios)
    elif relative_train_manifest_ratios is not None:
        return RelativeManifestRatios(ratios=relative_train_manifest_ratios)
    elif canary_manifest_exponent is not None:
        return CanaryManifestRatios(exponent=canary_manifest_exponent)
    else:
        return None


@beartype
def duration(manifest: dict) -> float:
    return sum(x["duration"] for x in manifest.values())


@beartype
def build_json_fracs(
    manifest_ratios: ManifestRatios,
    manifest_lengths: list[int],
    manifest_durations: Optional[list[float]] = None,
) -> list[float]:
    match manifest_ratios:
        case RelativeManifestRatios(ratios):
            return [x * y for x, y in zip(ratios, manifest_lengths, strict=True)]
        case AbsoluteManifestRatios(ratios):
            assert len(ratios) == len(manifest_lengths)
            return ratios
        case CanaryManifestRatios(exponent):
            assert manifest_durations is not None
            T = sum(manifest_durations)
            return [
                (u / n) * (n / T) ** exponent
                for u, n in zip(manifest_lengths, manifest_durations, strict=True)
            ]
        case None:
            return lmap(float, manifest_lengths)
        case _:
            raise ValueError(f"Invalid valid for {manifest_ratios=}")
