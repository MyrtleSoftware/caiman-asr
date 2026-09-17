import random
from functools import partial

from beartype import beartype
from beartype.typing import Literal, Optional, Union
from datasets import load_dataset
from lhotse import CutSet, RecordingSet, SupervisionSegment
from lhotse.cut import Cut, MonoCut, MultiCut
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    DynamicCutSampler,
    ReverbWithImpulseResponse,
    SamplingConstraint,
    SimpleCutSampler,
    TimeConstraint,
)
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import CutSampler

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.lhotse.config import (
    AreaBatch,
    Bucketing1D,
    Bucketing2D,
    DataConfig,
    DynamicBucketing1D,
    FixedBucketing2D,
    SimpleBatch,
    SimpleDurationBatch,
    SingleBucket2D,
)
from caiman_asr_train.lhotse.constraints import (
    AreaConstraint,
    BucketingFilter,
    SingleBucketFilter,
)
from caiman_asr_train.lhotse.cutset import read_cutset_from_config
from caiman_asr_train.lhotse.stats import EpochStats
from caiman_asr_train.lhotse.text import (
    MemoizedNormalize,
    PrefixTokenize,
    Tokenize,
)
from caiman_asr_train.lhotse.warn import lhotse_warn


@beartype
def get_lhotse_sampler_from_config(
    config: DataConfig,
    global_rank: int,
    world_size: int,
    tokenizer: Optional[Tokenizer],
) -> tuple[CutSampler, EpochStats]:
    """
    Build a Lhotse CutSampler from the provided DataConfig.
    this will tokenize and apply augmentations on-the-fly
    """
    # This will have mux'ed datasets specified in the config
    cuts, stats = read_cutset_from_config(config)

    # Apply audio channel selection
    if config.channel_selector is not None:
        cuts = cuts.map(
            partial(_select_channel, channel_selector=config.channel_selector)
        )

    # Noop if correct sample rate
    cuts = cuts.resample(config.sample_rate)

    # If normalization is requested a custom attribute "normalized_text"
    # will be added to the supervision segments containing the on-the-fly
    # normalized text.

    if config.normalize is not None:
        if config.warn and not config.use_iterable_dataset:
            lhotse_warn(
                """\
                You are using a mappable dataset and requested normalization
                during sampling. This will cause the normalization to happen in
                the main (GPU) process, probably impacting the training speed
                until the normalization cache is filled.\
                """
            )

        if tokenizer is None:
            charset = config.normalize.charset
        else:
            # this is added by hand and not user-defined in the config
            charset = tokenizer.charset

            if config.warn and charset != config.normalize.charset:
                lhotse_warn(
                    """\
                    You have provided a tokenizer with a different charset to the
                    normalization charset. The tokenizer charset will be used for
                    normalization.\
                    """
                )

        # Because map/transform is lazy and the CutSet is infinite and
        # immutable the caching must be done by the callable.
        norm = MemoizedNormalize(
            config.normalize.config,
            charset=charset,
            max_cache_size=config.normalize.cache_size,
        )

        cuts = cuts.map_supervisions(partial(_normalize, norm=norm))

    # Tokenization will add a custom attribute "tokens" to the supervision
    # segments containing the token ids as a numpy array.

    if config.warn and tokenizer is not None and not config.use_iterable_dataset:
        lhotse_warn(
            """\
            You are using mappable dataset and requested tokenization
            during sampling. This will cause the tokenization to happen in
            the main (GPU) process, probably impacting the training speed.\
            """
        )

    if config.conditional is not None:
        if tokenizer is None:
            raise ValueError("PNC/NoPNC conditional training requires a tokenizer")

        cuts = cuts.map_supervisions(
            PrefixTokenize(
                tokenizer=tokenizer,
                user_symbols=config.conditional.user_symbols,
                pnc_prob=config.conditional.pnc_prob,
                languages=config.conditional.languages,
                lang_mask_prob=config.conditional.lang_mask_prob,
            )
        )

        if config.filter_unk is not None:
            cuts = cuts.filter_supervisions(partial(_unk_pred, unk=config.filter_unk))

    elif tokenizer is not None:
        cuts = cuts.map_supervisions(Tokenize(tokenizer))

        if config.filter_unk is not None:
            cuts = cuts.filter_supervisions(partial(_unk_pred, unk=config.filter_unk))
    else:
        # No tokenization will occur
        pass

    # Token based filtering
    if tokenizer is not None:
        cuts = cuts.filter(
            partial(_toks_pred, min_toks=config.min_tokens, max_toks=config.max_tokens)
        )
        cuts = cuts.filter(
            partial(_tps_pred, min_cps=config.min_tps, max_cps=config.max_tps)
        )

    # Apply augmentations to the CutSet

    if config.background_noise is not None:
        # This will be cached in the HF_CACHE
        noise = CutSet.from_huggingface_dataset(
            load_dataset(
                config.background_noise.hf_dataset,
                split=config.background_noise.hf_split,
            )
        )
        noise = noise.resample(config.sample_rate)
        noise = noise.to_eager()

        cuts = cuts.mix(
            cuts=noise,
            snr=config.background_noise.snr,
            mix_prob=config.background_noise.mix_prob,
            seed=config.shard_seed,
            random_mix_offset=True,
        )

    if config.speed_perturbations:
        cuts = CutSet.mux(
            cuts, *[cuts.perturb_speed(x) for x in config.speed_perturbations]
        )

    if config.tempo_perturbations:
        cuts = CutSet.mux(
            cuts, *[cuts.perturb_tempo(x) for x in config.tempo_perturbations]
        )

    if config.volume_perturbations:
        cuts = CutSet.mux(
            cuts, *[cuts.perturb_volume(x) for x in config.volume_perturbations]
        )

    # ... Sampling

    # If we are using an iterable dataset assume that the cutset is handling
    # the GPU data sharding (via randomization) and set rank=0, world_size=1.
    # Otherwise, inform the sampler about the distributed setup and let it
    # handle it.

    shuffle = config.sampler_shuffle

    cuts, constraint = _get_constraint(cuts, config)

    if isinstance(config.sampler, (SimpleBatch, SimpleDurationBatch, SingleBucket2D)):
        if cuts.is_lazy:
            sampler = DynamicCutSampler(
                cuts,
                constraint=constraint,
                # Common
                drop_last=config.sampler_drop_last,
                shuffle=shuffle is not None,
                shuffle_buffer_size=0 if shuffle is None else shuffle.buffer_size,
                seed=config.shard_seed,
                rank=0 if config.use_iterable_dataset else global_rank,
                world_size=1 if config.use_iterable_dataset else world_size,
            )
        else:
            # This can happened during validation, we don't want to use the
            # dynamic cut sampler as it raises a warning and can't detect the
            # drop_last validation stall.

            max_cuts, max_duration = None, None

            if isinstance(config.sampler, SimpleBatch):
                max_cuts = config.sampler.batch_size
            elif isinstance(config.sampler, SimpleDurationBatch):
                max_duration = config.sampler.batch_size.max_duration

                if config.sampler.batch_size.quadratic_duration is not None:
                    # Could use former branch but I think this is not required
                    # to support
                    raise NotImplementedError("Quadratic duration not supported")

            else:
                assert False, "Unreachable"

            sampler = SimpleCutSampler(
                cuts,
                max_duration=max_duration,
                max_cuts=max_cuts,
                # Common
                drop_last=config.sampler_drop_last,
                shuffle=shuffle is not None,
                # no shuffle buffer for eager
                seed=config.shard_seed,
                rank=0 if config.use_iterable_dataset else global_rank,
                world_size=1 if config.use_iterable_dataset else world_size,
            )
    else:
        match config.sampler:
            case FixedBucketing2D() | Bucketing2D():
                # We need to provide a mock bin list for 2D bucketing,
                # we subtract one because lhotse's impl adds one as it
                # assumes these are bin edges.
                duration_bins = [0.0 for _ in range(len(config.sampler.bins) - 1)]
            case Bucketing1D():
                duration_bins = config.sampler.bins
            case _:
                duration_bins = None

        # These arguments are ignored by lhotse if dur_bins is not None
        if isinstance(config.sampler, DynamicBucketing1D):
            num_buckets = config.sampler.num_buckets
            est_cuts = config.sampler.num_cuts_for_bins_estimate

            min_buffer = (
                num_buckets
                * config.sampler.batch_size.max_duration
                / stats.avg_duration
            )

            if min_buffer > config.sampler.opts.buffer_size:
                lhotse_warn(
                    f"""\
                    The minimum recommended bucketing buffer size for the
                    dataset is {min_buffer:.0f}. However, you have configured
                    a buffer size of {config.sampler.opts.buffer_size}, be
                    aware that increasing the buffer size may consume an
                    undesirable quantity of CPU ram.\
                    """
                )
        else:
            num_buckets = None
            est_cuts = 0

        sampler = DynamicBucketingSampler(
            cuts,
            constraint=constraint,
            num_buckets=num_buckets,
            num_cuts_for_bins_estimate=est_cuts,
            duration_bins=duration_bins,
            # Common for all bucketing
            sync_buckets=config.sampler.opts.sync_buckets,
            concurrent=config.sampler.opts.concurrent,
            buffer_size=config.sampler.opts.buffer_size,
            # Common with non-bucketing
            drop_last=config.sampler_drop_last,
            shuffle=shuffle is not None,
            seed=config.shard_seed,
            rank=0 if config.use_iterable_dataset else global_rank,
            world_size=1 if config.use_iterable_dataset else world_size,
        )

    if config.cut_concat is not None:
        # Cut concatenation will produce longer samples out of shorter samples
        # by gluing them together from the shortest to longest not to exceed a
        # duration of longest_cut * duration_factor (greedy knapsack algorithm
        # for minimizing padding). Useful e.g. for simulated code-switching in
        # multilingual setups. We follow concatenation by
        # ``merge_supervisions`` which creates a single supervision object with
        # texts joined by a whitespace so that "regular" dataset classes don't
        # have to add a special support for multi-supervision cuts.
        sampler = sampler.map(
            CutConcatenate(
                gap=config.cut_concat.gap,
                duration_factor=config.cut_concat.duration_factor,
                max_duration=config.cut_concat.max_duration,
            )
        )
        sampler = sampler.map(_merge_supervisions)

    if config.rir_augmentation is not None:
        if config.rir_augmentation.path is not None:
            rec = RecordingSet.from_file(config.rir_augmentation.path)
        else:
            rec = None

        sampler = sampler.map(
            ReverbWithImpulseResponse(
                rir_recordings=rec,
                p=config.rir_augmentation.prob,
                # This is created in the main process and passed
                # to workers so we use the primary seed.
                randgen=random.Random(resolve_seed(config.prime_seed)),
                early_only=config.rir_augmentation.early_only,
            )
        )

    return sampler, stats


@beartype
def _select_channel(
    cut: Cut, channel_selector: Union[int, Literal["mono_downmix"]]
) -> MonoCut:
    """
    Select a channel from a multi-channel cut recording.

    Note: if down-mixing is selected, a custom field "original_source" will be
    added to the returned MonoCut containing the list of original sources.
    """
    match cut, channel_selector:
        case MultiCut(), "mono_downmix":
            mono_cut = cut.to_mono(mono_downmix=True)
            assert isinstance(mono_cut, MonoCut)

            if cut.recording is not None:
                # Down-mixing will create a new recording, attach a custom
                # field to indicate its original source.
                return mono_cut.with_custom(
                    "original_source", [x.source for x in cut.recording.sources]
                )

            return mono_cut

        case MultiCut(), int(channel):
            if channel >= cut.num_channels:
                raise ValueError(
                    f"Channel {channel} not in cut with {cut.num_channels} channels"
                )

            mono_cut = cut.with_channels(channel)
            assert isinstance(mono_cut, MonoCut)
            return mono_cut

        case MonoCut(), "mono_downmix":
            return cut  # already mono

        case MonoCut(), int(channel):
            if channel != 0:
                raise ValueError("MonoCut can only select channel 'mix' or channel 0")

            return cut

        case _, _:
            raise ValueError(f"Unexpected Cut type: {type(cut)}")


@beartype
def _get_constraint(
    cuts: CutSet, config: DataConfig
) -> tuple[CutSet, SamplingConstraint]:
    """
    Build a constraint from the config, this may filter the cuts if
    required by the constraint.
    """
    match config.sampler:
        case SimpleBatch(batch_size):
            return cuts, TimeConstraint(
                max_cuts=batch_size,
            )
        case SingleBucket2D(batch_size=batch_size, bucket_idx=bucket_idx):
            bucket_cons = AreaConstraint(
                area_bins=config.sampler.bins,
                strict=config.sampler.strict,
                # `area` is not used by _select_bucket which is
                # the only method called by SingleBucketFilter
                area=AreaBatch(),
            )

            time_cons = TimeConstraint(max_cuts=batch_size)

            return cuts.filter(SingleBucketFilter(bucket_cons, bucket_idx)), time_cons

        case SimpleDurationBatch() | DynamicBucketing1D() | Bucketing1D():
            return cuts, TimeConstraint(
                max_duration=config.sampler.batch_size.max_duration,
                quadratic_duration=config.sampler.batch_size.quadratic_duration,
            )
        case Bucketing2D():
            constraint = AreaConstraint(
                area_bins=config.sampler.bins,
                area=config.sampler.batch_size,
                strict=config.sampler.strict,
            )

            return cuts.filter(BucketingFilter(constraint)), constraint
        case _:
            # TODO: implement
            raise NotImplementedError()


@beartype
def _normalize(seg: SupervisionSegment, norm: MemoizedNormalize) -> SupervisionSegment:
    """
    This function will normalize the text field of a supervision segment and
    store the result in a custom field "normalized_text".
    """
    # Sanity
    assert seg.text is not None, "Cut has no text"
    assert seg.language is not None, "Cut has no language"

    # the language is included in the SupervisionSegment
    return seg.with_custom("normalized_text", norm(seg.text, seg.language))


@beartype
def _merge_supervisions(cuts: CutSet) -> CutSet:
    return cuts.merge_supervisions()


@beartype
def _toks_pred(cut: Cut, min_toks: int, max_toks: Optional[int]) -> bool:
    """
    Test if the number of tokens in the cut is within the specified range.
    """
    assert len(cut.supervisions) == 1, cut.supervisions
    assert cut.supervisions[0].has_custom("tokens"), "Cut has no tokens"
    tokens = len(cut.supervisions[0].tokens)

    if max_toks is None:
        return min_toks <= tokens
    else:
        return min_toks <= tokens <= max_toks


@beartype
def _tps_pred(cut: Cut, min_cps: float, max_cps: float) -> bool:
    """
    Test if the number of tokens per second is within the specified range.
    """
    # Don't repeat checks in _toks_pred
    tokens = len(cut.supervisions[0].tokens)
    cps = tokens / cut.duration
    return min_cps <= cps <= max_cps


@beartype
def _unk_pred(seg: SupervisionSegment, unk: int) -> bool:
    return unk not in seg.tokens
