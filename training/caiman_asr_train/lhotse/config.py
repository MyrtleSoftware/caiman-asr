import json
from dataclasses import dataclass, field
from enum import Enum, auto
from math import log
from pathlib import Path

import torch
from beartype import beartype
from beartype.typing import (
    Literal,
    NamedTuple,
    Optional,
    TypeAlias,
    Union,
)
from lhotse import Seconds

from caiman_asr_train.lhotse.stats import FiltStats
from caiman_asr_train.setup.text_normalization import (
    NormalizeConfig,
)

# === Manifests


@beartype
@dataclass
class Manifest:
    """
    A class representing an audio dataset.

    Members:
        path: The absolute path to the manifest.
        manual_data_dir: If given use this as the root of the audio files.
        filt_duration: If given assume the filtered duration/utt-count is this.
    """

    path: Path
    manual_data_dir: Optional[Path] = None
    filt_stats: Optional[FiltStats] = None

    @classmethod
    def build(
        cls,
        path: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        filt_stats: Optional[FiltStats] = None,
    ) -> "Manifest":
        """
        Create a Manifest, try to infer if the path is relative to the CWD
        or the data_dir if given.

        This will try in order:
            If path is absolute, just use it.
            If path is relative and exists, resolve it and use it.
            If path is relative, and doesn't exist, prepend data_dir if given.
        """

        path = Path(path) if isinstance(path, str) else path
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

        if data_dir is not None:
            data_dir = data_dir.resolve()

        if path.is_absolute() or path.is_file() or data_dir is None:
            return cls(
                path=path.resolve(), manual_data_dir=data_dir, filt_stats=filt_stats
            )

        return cls(
            path=(data_dir / path).resolve(),
            manual_data_dir=data_dir,
            filt_stats=filt_stats,
        )

    @property
    def data_dir(self) -> Path:
        """
        If manual_data_dir is given use that, otherwise infer from the manifest path.
        """
        return (
            self.path.parent if self.manual_data_dir is None else self.manual_data_dir
        )


@beartype
class WeightedManifest(NamedTuple):
    manifest: Manifest
    weight: Union[int, float]


class CanaryMode(Enum):
    DUR = auto()  # Expected duration per batch follows canary ratio
    UTT = auto()  # Expected utterances per batch follows canary ratio


@beartype
class CanaryHyperParams(NamedTuple):
    alpha: float = 0.5
    mode: CanaryMode = CanaryMode.DUR


@beartype
class CanaryManifests(NamedTuple):
    manifests: list[Manifest]
    params: CanaryHyperParams


ManifestGroup: TypeAlias = Union[
    list[Manifest], list[WeightedManifest], CanaryManifests
]


@beartype
@dataclass
class MultilingualSharManifests:
    manifests: dict[str, ManifestGroup]  # lang-code -> collection of manifests
    canary_params: Optional[CanaryHyperParams] = None


@beartype
@dataclass
class MultilingualJsonlManifests:
    manifests: dict[str, ManifestGroup]  # lang-code -> collection of manifests
    canary_params: Optional[CanaryHyperParams] = None


MultiManifestGroup: TypeAlias = Union[
    MultilingualSharManifests,
    MultilingualJsonlManifests,
]


# === Bucketing/Sampling


@beartype
@dataclass
class Shuffle:
    buffer_size: int = 20_000


@beartype
@dataclass
class BucketingOpts:
    concurrent: bool = False  # This is off because of issues upstream in lhotse
    sync_buckets: bool = True
    buffer_size: int = 20_000


@beartype
@dataclass
class DurationBatch:
    """
    Specify the total duration of a batch, if quadratic_duration is given then
    each effective cut duration becomes dur <- dur + dur^2 / quadratic_duration
    """

    max_duration: Seconds
    quadratic_duration: Optional[float] = None


@beartype
@dataclass
class AreaBatch:
    """
    The total memory is approx (with batch size B) is:

        M = b_1 U_max B + b_2 T_max B + b_3 sum_i {u_i * t_i}

    With each utterances having u_i tokens and t_i seconds of audio. We use
    'sum_i {u_i * t_i}' instead of `T_max * U_max * B` as we use a packed joint
    network.

    This allows you to constrain the batch by the above weighted combination,
    with optional bounds on the maximum duration/tokens/area.

    """

    max_vram: float = float("inf")
    max_time: float = float("inf")  # Seconds
    max_toks: float = float("inf")
    max_area: float = float("inf")

    coeff_time: float = 1.0
    coeff_toks: float = 1.0
    coeff_area: float = 1.0

    def full(self, batch_size: int, max_U: int, max_T: float, sum_ut: float) -> bool:
        """
        Test if a batch is full given the current stats.
        """

        time = max_T * batch_size
        toks = max_U * batch_size
        area = sum_ut

        if time > self.max_time or toks > self.max_toks or area > self.max_area:
            return True

        ols_vram = (
            self.coeff_toks * toks + self.coeff_time * time + self.coeff_area * area
        )

        return ols_vram > self.max_vram


# ... Unbucketed


@beartype
@dataclass
class SimpleBatch:
    """
    No bucketing, batch is constrained by batch size only.
    """

    batch_size: int


@beartype
@dataclass
class SimpleDurationBatch:
    """
    No bucketing, batch is constrained by total duration only.
    """

    batch_size: DurationBatch


# ... 1D bucketing


@beartype
@dataclass
class DynamicBucketing1D:
    """
    The bucket boundaries are chosen dynamically by scanning the cutset.
    """

    batch_size: DurationBatch
    num_buckets: int
    num_cuts_for_bins_estimate: int = 10_000
    opts: BucketingOpts = field(default_factory=BucketingOpts)


@beartype
@dataclass
class Bucketing1D:
    """
    The bucket boundaries are supplied, no need to scan the cutset.
    """

    batch_size: DurationBatch
    duration_bins: Union[Path, list[Seconds]]  # max_durations
    opts: BucketingOpts = field(default_factory=BucketingOpts)

    @property
    def bins(self) -> list[Seconds]:
        match self.duration_bins:
            case Path():
                with self.duration_bins.open("r") as f:
                    return json.load(f)
            case list():
                return self.duration_bins


# ... 2D bucketing


@beartype
@dataclass
class Bucketing2D:
    """
    Use fixed 2D buckets, fill each batch to the area constraint.

    Instead of directly specifying the buckets a path to a json file can
    be provided.
    """

    batch_size: AreaBatch
    area_bins: Union[Path, list[tuple[Seconds, int]]]  # (max_duration, max_tokens)
    strict: bool = True
    opts: BucketingOpts = field(default_factory=BucketingOpts)

    @property
    def bins(self) -> list[tuple[Seconds, int]]:
        match self.area_bins:
            case Path():
                with self.area_bins.open("r") as f:
                    return [tuple(x) for x in json.load(f)]
            case list():
                return self.area_bins


@beartype
@dataclass
class SplitPaths:
    """
    To separate the bin shapes from bin batch sizes. This is useful because the
    bin sizes are a function of the dataset while the batch sizes are a function
    of the GPU.
    """

    bin_sizes: Path  # (max_dur, max_toks)
    bin_batch: Path


@beartype
@dataclass
class FixedBucketing2D:
    """
    Use fixed 2D buckets, fill each batch to a fixed number of cuts.

    Members:
        fixed_area_bins: (batch_size, max_dur, max_toks)
    """

    # (bin_batch_size, max_duration, max_tokens)
    fixed_area_bins: Union[Path, SplitPaths, list[tuple[int, Seconds, int]]]
    strict: bool = True
    opts: BucketingOpts = field(default_factory=BucketingOpts)

    @property
    def bins(self) -> list[tuple[int, Seconds, int]]:
        match self.fixed_area_bins:
            case Path():
                with self.fixed_area_bins.open("r") as f:
                    return [tuple(x) for x in json.load(f)]
            case list():
                return self.fixed_area_bins
            case SplitPaths(bin_sizes=sizes, bin_batch=batches):
                with sizes.open("r") as f:
                    dur, toks = zip(*json.load(f))

                with batches.open("r") as f:
                    bat = [x for x in json.load(f)]

                return list(zip(bat, dur, toks, strict=True))


@beartype
@dataclass
class SingleBucket2D:
    """
    Use fixed 2D buckets, only return batches from a single bucket.
    """

    batch_size: int
    bucket_idx: int
    area_bins: Union[Path, list[tuple[Seconds, int]]]  # (max_duration, max_tokens)
    strict: bool = True

    @property
    def bins(self) -> list[tuple[Seconds, int]]:
        match self.area_bins:
            case Path():
                with self.area_bins.open("r") as f:
                    return [tuple(x) for x in json.load(f)]
            case list():
                return self.area_bins


# ... Aggregate

Sampling: TypeAlias = Union[
    SimpleBatch,
    SimpleDurationBatch,
    Bucketing1D,
    Bucketing2D,
    DynamicBucketing1D,
    FixedBucketing2D,
    SingleBucket2D,
]


# === Mel normalization


@beartype
@dataclass
class UttNorm:
    pass


@beartype
@dataclass
class GlobalNorm:
    stats: Path


@beartype
@dataclass
class BlendedNorm:
    stats: Path
    #
    ramp_beg: Seconds
    ramp_end: Seconds


MelNorm: TypeAlias = Union[UttNorm, GlobalNorm, BlendedNorm]

# === Post pipeline


@beartype
@dataclass
class SpecAug:
    freq_masks: int = 2
    freq_width: int = 20
    time_masks: int = 10
    time_width: Union[int, float] = 0.03


@beartype
@dataclass
class AlwaysEOI:
    pass


@beartype
@dataclass
class ProbEOI:
    drop_fraction: float = 0.5


OptEOI: TypeAlias = Union[AlwaysEOI, ProbEOI]

# === Text normalization


@beartype
@dataclass
class Normalization:
    config: NormalizeConfig  # passed to norm_and_tokenize()
    charset: list[str]  # charset derived from the lang of the transcripts
    cache_size: Optional[int] = None  # None ~ unlimited


@beartype
@dataclass
class Conditional:
    user_symbols: dict[str, str]
    pnc_prob: float = 0.25
    cache_size: Optional[int] = None  # None ~ unlimited
    languages: Optional[dict[str, str]] = None  # lang-code -> lang-token
    lang_mask_prob: Optional[float] = None  # randomly mask lang-toks in train data


# === Misc


class DatasetKind(Enum):
    MAPPABLE = auto()
    ITERABLE = auto()


class PinOptions(Enum):
    ALWAYS = auto()
    NEVER = auto()
    AUTO = auto()  # Pin if using cuda


# Seed will be resolved at runtime in the main (GPU) process.
#   int        -> Just use as the seed
#   trng       -> True ranom number generator
#   randomized -> If in torch dataloader worker returns a unique
#                 but deterministic seed, otherwise returns
#                 Python's random module global seed.
Seed: TypeAlias = Union[int, Literal["trng", "randomized"]]


# === Augmentation


@beartype
@dataclass
class CutConcat:
    """
    gap: The duration of silence in seconds that is inserted between the
        cuts; its goal is to let the model "know" that there are separate
        utterances in a single example.
    duration_factor: Determines the maximum duration of the concatenated
        cuts; by default it's 1, setting the limit at the duration of the
        longest cut in the batch.
    max_duration: If a value is given (in seconds), the maximum duration
        of concatenated cuts is fixed to the value while duration_factor is
        ignored.
    """

    gap: Seconds = 0.1
    duration_factor: Seconds = 1.0
    max_duration: Optional[Seconds] = None


@beartype
@dataclass
class RoomImpulseResponseAug:
    path: Optional[Path]  # to RecordingSet
    prob: float = 0.1
    early_only: bool = False


@beartype
@dataclass
class BackgroundNoise:
    hf_dataset: str = "Myrtle/CAIMAN-ASR-BackgroundNoise"
    hf_split: Optional[str] = "train"
    snr: tuple[float, float] = (10.0, 20.0)  # dB
    mix_prob: float = 0.33


@beartype
@dataclass
class RequiredOverride:
    """
    If none match, raises an error. If drop is true, drop any custom fields
    matching the keys after the override.
    """

    keys: list[str]
    drop: bool = True


@beartype
@dataclass
class TryOverride:
    """
    If none match fall back to default text (no override). If drop is true,
    drop any custom fields matching the keys after the override.
    """

    keys: list[str]
    drop: bool = True


@beartype
@dataclass
class DataConfig:
    """
    Single source of truth for lhotse pipeline configuration.
    """

    # ... Manifests

    # Where the data manifests are located
    manifests: MultiManifestGroup
    # Override the transcript field in the supervision with this key, if a list
    # is given then take the first key that exists in the json.
    override_text_key: Optional[Union[RequiredOverride, TryOverride]] = None

    # Drop these custom fields from the supervision segments, this is applied
    # after override_text_key.
    drop_custom_fields: Optional[list[str]] = None

    # If true, the source manifests path will be added the the supervision's
    # custom field under the "source_manifest" key.
    append_source_manifest: bool = True

    # Enforce that the cutset is finite (no repetition), this will break
    # manifest balancing and probably cause DDP to hang on the final iteration.
    force_finite: bool = False

    # If set the cutset will be sorted by duration before sampling. This will
    # require a full iteration and make the cutset eager hence, this is
    # probably only useful for validation and debugging.
    initial_sort: bool = False

    # ... Global

    # Each GPU process will resolve the prime_seed and use it to
    # initialize it and its workers RNGs. Within the sampler all
    # lhotse classes receive the shard_seed.
    prime_seed: Seed = "trng"
    shard_seed: Seed = "trng"

    # See function with the same name, this should almost always be true.
    cuda_expandable_segments: bool = True

    # If true print warnings about possible misconfigurations.
    warn: bool = True

    # ... Sampler

    # Filtering on audio duration, (pre balancing)
    min_duration: Seconds = 0.25
    max_duration: Seconds = Seconds("inf")

    # Filtering on number of characters, (pre balancing/normalization)
    min_chars: int = 0
    max_chars: Optional[int] = None  # None ~ infinite

    # Filtering on characters per second, (pre balancing/normalization)
    min_cps: float = 0.0
    max_cps: float = float("inf")

    # Select a given channel from multi-channel audio recordings or down-mix
    # them to mono if "mono_downmix" is given.
    channel_selector: Optional[Union[int, Literal["mono_downmix"]]] = None

    # Resample all audio to this rate on-the-fly (recommended to do this offline).
    sample_rate: int = 16_000

    # Controls the text normalization/standardization of the tokenized
    # (training) stream. This is applied lazily but cached
    normalize: Optional[Normalization] = None

    # PnC mode
    conditional: Optional[Conditional] = None

    # Filter out transcripts containing this unk token, note that if the
    # tokenizer is set to fail on unk then this filter is redundant.
    filter_unk: Optional[int] = None

    background_noise: Optional[BackgroundNoise] = field(default_factory=BackgroundNoise)

    # Mix-in versions of the audio with the following speed perturbations.
    speed_perturbations: Optional[list[float]] = field(
        default_factory=lambda: [0.9, 1.1]
    )

    # Mix-in versions of the audio with the following volume perturbations.
    volume_perturbations: Optional[list[float]] = None

    # Mix-in versions of the audio with the following tempo perturbations.
    tempo_perturbations: Optional[list[float]] = None

    # Filtering on number of tokens, (post balancing/normalization)
    min_tokens: int = 1
    max_tokens: Optional[int] = None  # None ~ infinite

    # Filtering on tokens per second, (post balancing/normalization)
    min_tps: float = 0.0
    max_tps: float = float("inf")

    # ... Sampler.lhotse

    sampler: Sampling = field(default_factory=lambda: SimpleBatch(1))
    sampler_shuffle: Optional[Shuffle] = field(default_factory=Shuffle)
    sampler_drop_last: bool = True

    # ... Sampler.post_sample

    # On the fly cut concatenation
    cut_concat: Optional[CutConcat] = None

    # On the fly reverberation
    rir_augmentation: Optional[RoomImpulseResponseAug] = None

    # ... DataLoader options

    # Passed to torch.utils.data.DataLoader's __init__.
    loader_workers: int = 0
    loader_pin_memory: PinOptions = PinOptions.AUTO
    loader_prefetch_factor: Optional[int] = None

    @property
    def loader_pin_memory_bool(self) -> bool:
        """
        Convert `self.loader_pin_memory` to a bool given `self.use_cuda`.
        """
        match self.loader_pin_memory:
            case PinOptions.ALWAYS:
                return True
            case PinOptions.NEVER:
                return False
            case PinOptions.AUTO:
                return self.use_cuda

    # The following two options may be used to override auto-detection of
    # appropriate PyTorch dataset flavour for your data types. PyTorch
    # DataLoader uses two objects to yield data: dataset and sampler.
    #
    # Map-dataset flavour: there is one sampler per GPU that lives in the
    # training loop process; it selects the examples to be prepared by
    # map-dataset class. Each batch selection determined by the sampler is then
    # passed by the dataloader to one of its worker processes to be processed
    # by the dataset class. In lhotse the sampler will take responsibility for
    # dividing the data between multiple GPUs and workers.
    #
    # Iterable-dataset flavour: each dataloading worker has its own sampler
    # replica instead; the sampler must have the logic for either data
    # deduplication or unique order shuffling to avoid duplicated data across
    # workers and GPUs. Lhotse relies on unique order shuffling.
    #
    # The default settings are: Iterable dataset for tarred audio data,
    # Map dataset for non-tarred audio data.
    #
    # In lhotse the sampler does 100% of the work while the dataset just
    # handles the audio IO.
    #
    # As we disable pytorch's automatic batching the map-dataset looks like:
    #
    # ```python
    #   for index in sampler:
    #       yield dataset[index]
    # ```
    #
    # And for iterable-dataset:
    #
    # ```python
    #   for data in dataset:
    #       yield data
    # ```
    #
    # Where in lhotse the iterable dataset is just a wrapper around a sampler
    # and a mappable dataset.
    #
    # Hence, forcing an iterable dataset for non-tarred data will result in all
    # the workers iteration over the same data order, possibly with some stream
    # shuffling if shuffle=True and they are seeded differently.
    #
    # Conversely, forcing a map-dataset for tarred data will result in
    # performing IO in the training loop as the sampler (inside the dataset)
    # has to do IO to iterate over the tarred data shards.
    loader_force_dataset: Optional[DatasetKind] = None

    @property
    def use_iterable_dataset(self) -> bool:
        match self.loader_force_dataset, self.manifests:
            case None, MultilingualSharManifests():
                return True
            case None, MultilingualJsonlManifests():
                return False
            case DatasetKind.MAPPABLE, _:
                return False
            case DatasetKind.ITERABLE, _:
                return True

    # ... DataSet options

    dataset_workers: int = 0
    # Pre/post-pad with silence to mirror inference.
    head_stream_padding: bool = True
    tail_stream_padding: bool = True
    # Post-pad with this much silence.
    final_silence_padding: Seconds = 0.0

    @property
    def final_silence_padding_samples(self) -> int:
        return int(self.final_silence_padding * self.sample_rate)

    # ... Pipeline

    # Expecting a path to dump the debug audio (flac) to.
    audio_dump_dir: Optional[Path] = None

    # Expecting a path to dump the dataloader memory consumption stats to.
    memory_dump_path: Optional[Path] = None

    # Expecting a path to dump the features to.
    feature_dump_dir: Optional[Path] = None

    # Expecting a path to dump cut text (raw + normalized) to as JSONL.
    cuts_dump_path: Optional[Path] = None

    # Whether to the use the GPU for feature extraction (and all operations
    # on/after features), if this is true then the pipeline's output will be a
    # `torch.Tensor` on the GPU.
    use_cuda: bool = field(default_factory=torch.cuda.is_available)

    def maybe_cuda(self, x: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """
        Monadic version of `x.cuda()` if `self.use_cuda` is true.
        """
        if self.use_cuda:
            return x.cuda(non_blocking=non_blocking)

        return x

    # ... Pipeline.mel_spec

    # Pre-frequency domain
    dither_coeff: float = 0.00001
    preemph_coeff: float = 0.97

    # Spectrogram
    n_fft: int = 512
    win_length: Seconds = 0.02
    hop_length: Seconds = 0.01
    center: bool = False

    # For spectrogram mel binning
    n_mels: int = 80

    # For linear -> decibel conversion
    multiplier: float = log(10)
    reference: float = 1.0
    cutoff_db: float = log(1e-20)

    @property
    def win_sample_length(self) -> int:
        return int(self.sample_rate * self.win_length)

    @property
    def hop_sample_length(self) -> int:
        return int(self.sample_rate * self.hop_length)

    # ... Pipeline.mel_norm

    mel_norm: Optional[MelNorm] = None

    # ... Pipeline.spec_aug

    spec_aug: Optional[SpecAug] = field(default_factory=SpecAug)

    # ... Pipeline.frame_stacking

    feature_frame_stacking: int = 3
    encoder_frame_stacking: int = 2

    @property
    def total_frame_stacking(self) -> int:
        return self.feature_frame_stacking * self.encoder_frame_stacking

    # ... Pipeline.eoi

    # EOI frame options:
    #   none          -> no eoi appended
    #   always        -> always append eoi frame
    #   probabilistic -> append eoi frame with given probability
    eoi_append: Optional[OptEOI] = field(default_factory=ProbEOI)
