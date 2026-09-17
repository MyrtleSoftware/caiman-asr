from collections import deque
from dataclasses import asdict, dataclass
from functools import partial
from itertools import accumulate, starmap
from math import sqrt

from beartype import beartype
from beartype.typing import Iterable, NamedTuple, Optional, Union
from lhotse import CutSet, Seconds, SupervisionSegment
from lhotse.cut import Cut
from lhotse.utils import fastcopy as lhotse_fastcopy
from tqdm import tqdm

from caiman_asr_train.lhotse.config import (
    CanaryHyperParams,
    CanaryManifests,
    CanaryMode,
    DataConfig,
    Manifest,
    ManifestGroup,
    MultilingualSharManifests,
    RequiredOverride,
    Seed,
    TryOverride,
    WeightedManifest,
)
from caiman_asr_train.lhotse.stats import (
    DurationStats,
    EpochStats,
    FiniteStats,
    InfiniteStats,
)
from caiman_asr_train.lhotse.text_filters import get_banned_transcripts
from caiman_asr_train.train_utils.distributed import exec_and_broadcast, is_rank_zero
from caiman_asr_train.utils.iter import lstarmap_zip


@beartype
@dataclass
class _StatsAcc:
    """
    Accumulates statistics for a set of cuts.
    """

    min_duration: float = float("inf")
    max_duration: float = 0.0
    sum_duration: float = 0.0
    squ_duration: float = 0.0


@beartype
def read_cutset_from_config(config: DataConfig) -> tuple[CutSet, EpochStats]:
    """
    Read paths to Lhotse manifest files and create a CutSet.

    If multiple manifests are provided, they will be combined with a weighted
    multiplexer. The returned CutSet is always infinitely iterable unless
    config.force_finite is specified.

    Filtering is applied according to the config such that the
    multiplexing/weighting is done on the post-filtering statistics.

    All seeding is done using config.shard_seed.

    Returns:
        cutset, statistics
    """

    cutsets, names, langs, probs, filt_utt, filt_stats = _load_all(config)

    filt_dur = [s.sum_duration for s in filt_stats]

    tot_dur = sum(filt_dur)
    tot_utt = sum(filt_utt)

    # Compute global stats

    w_stats = _StatsAcc()

    for p, s in zip(probs, filt_stats, strict=True):
        w_stats = _StatsAcc(
            min_duration=min(w_stats.min_duration, s.min_duration),
            max_duration=max(w_stats.max_duration, s.max_duration),
            sum_duration=w_stats.sum_duration + p * s.sum_duration,
            squ_duration=w_stats.squ_duration + p * s.squ_duration,
        )

    eff_n = sum(p * n for p, n in zip(probs, filt_utt, strict=True))

    var, mean = _var_mean(w_stats.sum_duration, w_stats.squ_duration, eff_n)

    stats = DurationStats(
        max_duration=w_stats.max_duration,
        min_duration=w_stats.min_duration,
        avg_duration=mean,
        var_duration=var,
    )

    if is_rank_zero():
        _pretty_print(
            micro=filt_stats,
            count=filt_utt,
            names=names,
            langs=langs,
            probs=probs,
        )

    if config.force_finite:
        stats = FiniteStats(
            **asdict(stats),
            total_utt=tot_utt,
            total_dur=tot_dur,
        )
    else:
        orbit_dur = [dur / p for dur, p in zip(filt_dur, probs, strict=True)]
        orbit_utt = [utt / p for utt, p in zip(filt_utt, probs, strict=True)]

        stats = InfiniteStats(
            **asdict(stats),
            min_orbit_utt=min(orbit_utt),
            max_orbit_utt=max(orbit_utt),
            min_orbit_dur=min(orbit_dur),
            max_orbit_dur=max(orbit_dur),
        )

    if is_rank_zero():
        print("Global stats:")
        for k, v in asdict(stats).items():
            if "orbit_dur" in k or "total_dur" in k:
                print(f"\t{k} {v / 3600:.0f}h")
            elif "utt" in k:
                if v > 1000:
                    print(f"\t{k} {v / 1000:.0f}k")
                else:
                    print(f"\t{k} {v:.0f}")
            else:
                print(f"\t{k} {v:.2f}s")

    if len(cutsets) == 1:
        return cutsets[0], stats

    return CutSet.mux(*cutsets, weights=probs, seed=config.shard_seed), stats


@beartype
@dataclass
class _LoadResult:
    """
    - List of cutsets
    - Names of the manifests the cutsets were loaded from
    - Weights (or None) for each cutset
    - Number of utterances after filtering for each cutset (possibly user-supplied)
    - Stats after filtering for each cutset (possibly user-supplied)
    - Canary exponent and mode (or none if not canary manifest balanced)
    """

    cutsets: list[CutSet]
    names: list[str]
    weights: list[Union[None, int, float]]
    filt_utt: list[int]
    filt_stats: list[_StatsAcc]
    canary_params: Optional[CanaryHyperParams]

    @property
    def filt_dur(self) -> list[float]:
        return [s.sum_duration for s in self.filt_stats]


@beartype
class _WeightedCutsets(NamedTuple):
    cutsets: list[CutSet]
    names: list[str]
    langs: list[str]
    probs: list[float]
    filt_utt: list[int]
    filt_stats: list[_StatsAcc]


@beartype
def _load_all(config: DataConfig) -> _WeightedCutsets:
    if isinstance(config.manifests, MultilingualSharManifests):
        is_tarred = True
    else:
        is_tarred = False

    lang_manifests: dict[str, ManifestGroup] = config.manifests.manifests

    loaded_langs = {
        k: _load_each(is_tarred, v, config, k) for k, v in lang_manifests.items()
    }

    within_lang_probs = [
        _reify_weights(
            canary_params=v.canary_params,
            weights=v.weights,
            filt_utt=v.filt_utt,
            filt_dur=v.filt_dur,
        )
        for v in loaded_langs.values()
    ]

    if config.manifests.canary_params is not None:
        across_lang_probs = _reify_weights(
            canary_params=config.manifests.canary_params,
            weights=[None] * len(loaded_langs),
            filt_utt=[sum(v.filt_utt) for v in loaded_langs.values()],
            filt_dur=[sum(v.filt_dur) for v in loaded_langs.values()],
        )
    else:
        # Default to a uniform distribution across languages when no canary parameters
        # are specified. This ensures that across_lang_probs sums to 1.0.
        num_langs = len(loaded_langs)
        across_lang_probs = [1.0 / num_langs] * num_langs if num_langs > 0 else []

    # Aggregate the languages together

    cutsets: list[CutSet] = []
    names: list[str] = []
    langs: list[str] = []
    probs: list[float] = []
    filt_utt: list[int] = []
    filt_stats: list[_StatsAcc] = []

    for across_prob, within_probs, (lang, data) in zip(
        across_lang_probs,
        within_lang_probs,
        loaded_langs.items(),
        strict=True,
    ):
        cutsets.extend(data.cutsets)
        names.extend(data.names)
        langs.extend(lang for _ in data.cutsets)
        probs.extend(across_prob * within_prob for within_prob in within_probs)
        filt_utt.extend(data.filt_utt)
        filt_stats.extend(data.filt_stats)

    return _WeightedCutsets(
        cutsets=cutsets,
        names=names,
        langs=langs,
        probs=probs,
        filt_utt=filt_utt,
        filt_stats=filt_stats,
    )


@beartype
def _load_each(
    is_tarred: bool, manifests: ManifestGroup, config: DataConfig, lang: str
) -> _LoadResult:
    """
    Load all the manifests in the config into a CutSet and:
        - Patch the paths (relative -> absolute)
        - Select the correct text field
        - Filter on duration, characters, cps
        - Convert them to infinite iterables unless force_finite is set
    """

    if isinstance(manifests, CanaryManifests):
        manifests, canary_params = manifests
    else:
        canary_params = None

    assert len(manifests) > 0, "At least one manifest must be provided"

    cutsets: list[CutSet] = []
    weights: list[Union[None, int, float]] = []

    filt_utt: list[int] = []
    filt_stats: list[_StatsAcc] = []

    names: list[str] = []

    for manifest, weight in map(_unwrap, manifests):
        cutset = _load(manifest, is_tarred=is_tarred, shard_seed=config.shard_seed)
        init_utt = len(cutset)

        match config.override_text_key:
            case RequiredOverride(keys, drop):
                cutset = cutset.map_supervisions(
                    partial(_select_text, keys=keys, optional=False, drop=drop)
                )
            case TryOverride(keys, drop):
                cutset = cutset.map_supervisions(
                    partial(_select_text, keys=keys, optional=True, drop=drop)
                )
            case None:
                pass

        if config.drop_custom_fields:
            cutset = cutset.map_supervisions(
                partial(_prune_custom, keys=config.drop_custom_fields)
            )

        # We want to balance/mux on the accepted audio, we don't apply token
        # based filters here or the normalization because we need to iterate
        # the entire cutset to get the stats in the next step and that would
        # take very long if normalization/tokenisation is happening here.
        cutset = cutset.filter(
            partial(_dur_pred, min_dur=config.min_duration, max_dur=config.max_duration)
        )
        cutset = cutset.filter(
            partial(_char_pred, min_chars=config.min_chars, max_chars=config.max_chars)
        )
        cutset = cutset.filter(
            partial(_cps_pred, min_cps=config.min_cps, max_cps=config.max_cps)
        )

        # append language code to all supervisions
        cutset = cutset.map_supervisions(partial(_add_lang, lang=lang))

        # Quick pre-normalization text based filtering per language
        cutset = cutset.filter(_heuristics_pred)

        names.append(str(manifest.path))

        if manifest.filt_stats is None:
            utts, stats = exec_and_broadcast(
                _iter_for_stats, cutset.data, name=names[-1], length=init_utt
            )

        else:
            utts = manifest.filt_stats.tot_utterances

            stats = _StatsAcc(
                min_duration=manifest.filt_stats.min_duration,
                max_duration=manifest.filt_stats.max_duration,
                sum_duration=manifest.filt_stats.sum_duration,
                squ_duration=manifest.filt_stats.squ_duration,
            )

        if config.initial_sort:
            cutset = cutset.sort_by_duration(ascending=False)

        if config.append_source_manifest:
            # Add "source_manifest" custom field to all supervisions
            cutset = cutset.map_supervisions(
                partial(_add_source, source_manifest=str(manifest.path))
            )

        cutsets.append(cutset if config.force_finite else cutset.repeat())
        weights.append(weight)

        filt_utt.append(utts)
        filt_stats.append(stats)

    return _LoadResult(
        cutsets,
        names,
        weights,
        filt_utt,
        filt_stats,
        canary_params,
    )


@beartype
def _reify_weights(
    canary_params: Optional[CanaryHyperParams],
    weights: list[Union[None, int, float]],
    filt_utt: list[int],
    filt_dur: list[float],
) -> list[float]:
    """
    Calculate the final weights to use for multiplexing, these are probabilities.
    """
    tot_dur = sum(filt_dur)

    if canary_params is not None:
        # If Canary requested then compute biased weights.
        #
        # Let c(x) the _canary_ ratio be (tot_dur(x) / tot_dur) ** exp_alpha.
        # When exp_alpha = 1 this is uniform and when exp_alpha is < 1 this
        # over samples from shorter manifests.
        #
        # In utterance mode the probabiltiy of sampling from manifest x is c(x)
        # hence, in a batch of N samples:
        #
        # - The expected number of samples from manifest x is N * c(x)
        # - The expected duration from manifest x is N * avg_dur(x) * c(x)
        #
        # In duration mode the probability of sampling from manifest x is
        # c(x) / avg_dur(x) hence, in a batch of N samples:
        #
        # - The expected number of samples from manifest x is N * c(x) / avg_dur(x)
        # - The expected duration from manifest x is N * c(x)

        @beartype
        def _canary(utts: int, dur: float) -> float:
            match canary_params.mode:
                case CanaryMode.UTT:
                    return (dur / tot_dur) ** canary_params.alpha
                case CanaryMode.DUR:
                    return (utts / dur) * (dur / tot_dur) ** canary_params.alpha

        weights = lstarmap_zip(_canary, filt_utt, filt_dur, strict=True)

    elif all(w is None for w in weights):
        # Otherwise, if no weights then weight according to total duration of manifest
        weights = [dur for dur in filt_dur]
    else:
        # Otherwise, the user has specified weights
        pass

    assert all(isinstance(x, int | float) for x in weights)

    # Help the type checker
    resolved_weights: list[int | float] = weights  # type: ignore
    denom = sum(resolved_weights)

    return [w / denom for w in resolved_weights]


@beartype
def _pretty_print(
    micro: list[_StatsAcc],
    count: list[int],
    names: list[str],
    langs: list[str],
    probs: list[float],
) -> None:
    """
    Pretty print dataset statistics.
    """

    def proj(p, n, stats):
        return p * _var_mean(stats.sum_duration, stats.squ_duration, n)[1]

    pt = sum(starmap(proj, zip(probs, count, micro, strict=True)))

    print("Dataset stats:")

    for p, n, name, lang, stats in zip(probs, count, names, langs, micro, strict=True):
        # Header
        var, mean = _var_mean(stats.sum_duration, stats.squ_duration, n)
        # utt-fraction, dur-fraction, name
        print(f"  utt={p:6.2%} (dur={p * mean / pt:6.2%}) {lang=} {name}")

        print(f"\tutt={n // 1000}k", end="")
        print(f" avg={mean:.1f}±{sqrt(var):.1f}s", end="")
        print(f" dur={stats.sum_duration / 3600:.0f}h", end="")
        print(f" squ={stats.squ_duration / 3600:.0f}h", end="")
        print(f" min={stats.min_duration:.1f}s", end="")
        print(f" max={stats.max_duration:.1f}s", end="\n")


@beartype
def _var_mean(sum: float, sum_sq: float, n: int | float) -> tuple[float, float]:
    mean = sum / n
    var = sum_sq / n - mean**2
    return var, mean


@beartype
def _load(man: Manifest, is_tarred: bool, shard_seed: Seed) -> CutSet:
    """
    Load dataset from manifest, handling tarred and non-tarred cases. In the
    case of jsonl manifest the audio file paths are prefixed with the data_dir.
    """
    if is_tarred:
        # The combination of shuffle_shards, and later repeat() causes this to
        # be an infinite manifest that is internally reshuffled on each epoch.
        # The parameter ``config.shard_seed`` is used to determine
        # shard shuffling order.
        #
        # Options:
        #
        # - "trng" means we'll defer setting the seed until the iteration is
        # triggered, and we'll use system TRNG to get a completely random seed
        # for each worker. This results in every dataloading worker using full
        # data but in a completely different order.
        #
        # - "randomized" means we'll defer setting the seed until the iteration
        # is triggered, and we'll use config.seed to get a pseudo-random seed
        # for each worker. This results in every dataloading worker using full
        # data but in a completely different order. Unlike "trng", this is
        # deterministic, and if you resume training, you should change the seed
        # to observe different data examples than in the previous run.
        #
        # - integer means we'll set a specific seed in every worker, and data
        # would be duplicated across them. This is mostly useful for unit
        # testing or debugging.
        #
        # This all happens inside Lhotse's CutSet.from_shar().

        if not man.path.is_dir():
            raise ValueError(f"Tar manifest directory does not exist: {man.path}")

        return CutSet.from_shar(
            in_dir=man.path,
            seed=shard_seed,
            shuffle_shards=True,
            stateful_shuffle=True,
        )
    else:
        return load_jsonl_manifest(man)


@beartype
def _select_text(
    seg: SupervisionSegment,
    keys: list[str],
    optional: bool,
    drop: bool,
) -> SupervisionSegment:
    """
    Override the text of a SupervisionSegment with the first available custom
    data. This is expected to be a variant of the text, e.g., normalized text.

    Any keys present are dropped from the custom fields.
    """

    def maybe_drop(seg):
        return _prune_custom(seg, keys) if drop else seg

    for key in keys:
        if seg.has_custom(key):
            return maybe_drop(
                seg.transform_text(partial(_just, text=getattr(seg, key)))
            )

    if optional:
        return seg

    raise ValueError(f"None of the keys {keys} found in seg: {seg}")


@beartype
def _prune_custom(seg: SupervisionSegment, keys: list[str]) -> SupervisionSegment:
    """
    Remove custom fields from a SupervisionSegment.
    """
    for key in keys:
        if (maybe := seg.drop_custom(key)) is not None:
            seg = maybe

    return seg


@beartype
def _just(_, text: str) -> str:
    return text


@beartype
def load_jsonl_manifest(man: Manifest) -> CutSet:
    """
    Open a jsonl manifest and patch the paths from relative to absolute with
    the data_dir.
    """
    if not man.path.is_file():
        raise ValueError(f"Manifest file does not exist: {man.path}")

    cuts = CutSet.from_file(man.path)
    cuts = cuts.with_recording_path_prefix(man.data_dir)
    cuts = cuts.with_features_path_prefix(man.data_dir)
    return cuts


@beartype
def _add_source(seg: SupervisionSegment, source_manifest: str) -> SupervisionSegment:
    """
    Add a custom field "source_manifest" to a SupervisionSegment.
    """
    return seg.with_custom("source_manifest", source_manifest)


@beartype
def _add_lang(seg: SupervisionSegment, lang: str) -> SupervisionSegment:
    """
    Overwrite the language field of a SupervisionSegment.
    """
    # fastcopy is what Lhotse uses internally for transformations
    return lhotse_fastcopy(seg, language=lang)


@beartype
def _iter_for_stats(
    iter: Iterable[Cut],
    name: str,
    length: int,
) -> tuple[int, _StatsAcc]:
    """
    Iterate over a CutSet and compute the (#utts, total_duration, etc).
    """
    # Wrap for a status bar
    iter = tqdm(iter, desc=f"Scan {name}", total=length)

    @beartype
    def _bop(acc: _StatsAcc, dur: Seconds) -> _StatsAcc:
        return _StatsAcc(
            min_duration=min(acc.min_duration, dur),
            max_duration=max(acc.max_duration, dur),
            sum_duration=acc.sum_duration + dur,
            squ_duration=acc.squ_duration + dur * dur,
        )

    acc_iter = accumulate((x.duration for x in iter), _bop, initial=_StatsAcc())

    acc = deque(enumerate(acc_iter, 0), maxlen=1)
    assert len(acc) == 1, "Unreachable"

    print(f"Filtered {name}:")
    print(f"\ttot_utterances: {acc[0][0]}")
    print(f"\tmin_duration: {acc[0][1].min_duration:.2f}")
    print(f"\tmax_duration: {acc[0][1].max_duration:.2f}")
    print(f"\tsum_duration: {acc[0][1].sum_duration:.1f}")
    print(f"\tsqu_duration: {acc[0][1].squ_duration:.1f}")

    return acc[0]


@beartype
def _unwrap(
    item: Manifest | WeightedManifest,
) -> tuple[Manifest, int | float | None]:
    """
    Returns the manifest and its weight (or None).
    """
    match item:
        case Manifest():
            return item, None
        case WeightedManifest(manifest=manifest, weight=weight):
            return manifest, weight


@beartype
def _dur_pred(cut: Cut, min_dur: Seconds, max_dur: Seconds) -> bool:
    """
    Test if cut duration is within [min_dur, max_dur].
    """
    return min_dur <= cut.duration <= max_dur


@beartype
def _char_pred(cut: Cut, min_chars: int, max_chars: Optional[int]) -> bool:
    """
    Test if cut text length is within [min_chars, max_chars].
    """
    assert len(cut.supervisions) == 1, ""
    assert cut.supervisions[0].text is not None, "Cut has no text"
    chars = len(cut.supervisions[0].text)

    if max_chars is None:
        return min_chars <= chars
    else:
        return min_chars <= chars <= max_chars


@beartype
def _cps_pred(cut: Cut, min_cps: float, max_cps: float) -> bool:
    """
    Test if cut characters-per-second is within [min_cps, max_cps].
    """
    assert len(cut.supervisions) == 1, "Cut has multiple supervisions"
    assert cut.supervisions[0].text is not None, "Cut has no text"
    chars = len(cut.supervisions[0].text)
    cps = chars / cut.duration
    return min_cps <= cps <= max_cps


@beartype
def _heuristics_pred(cut: Cut) -> bool:
    """
    Test if cut transcript is one of the banned transcripts
    """
    transcript = cut.supervisions[0].text
    lang_code = cut.supervisions[0].language
    return transcript not in get_banned_transcripts(lang_code)
