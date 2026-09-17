from argparse import Namespace
from pathlib import Path

import yaml
from beartype import beartype
from beartype.typing import Any, Generator, List, Literal, Optional
from psutil import cpu_count

import caiman_asr_train.lhotse.config as lhotse_conf
from caiman_asr_train.lhotse.warn import lhotse_warn
from caiman_asr_train.rnnt import config as rnnt_config
from caiman_asr_train.setup.text_normalization import (
    normalize_config_from_full_yaml,
)
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens

_MAX_DEDUCED_LOADER_WORKERS_PER_GPU = 16


@beartype
def validate_dataset_yaml(path: str) -> None:
    """
    Validate the dataset yaml file
    """
    _ = _parse_dataset_yaml(Path(path))


@beartype
def build_from_legacy_sources(
    args: Namespace,
    cfg: dict,
    pipeline: Literal["train", "val"],
    use_cuda: bool,
    global_rank: int = 0,
    world_size: int = 1,
) -> tuple[lhotse_conf.DataConfig, dict, dict]:
    """
    Build the lhotse config from the legacy yaml config and CLI args.

    This will create configs with different seeds for each GPU.
    """

    # Parse the manifests
    match pipeline:
        case "train":
            manifests = _parse_dataset_yaml(
                Path(args.train_dataset_yaml), langs=_select_train_langs(cfg)
            )
        case "val":
            manifests = lhotse_conf.MultilingualJsonlManifests(
                {
                    args.val_standardizer_lang: [
                        lhotse_conf.Manifest.build(x, args.val_dataset_dir)
                        for x in args.val_manifests
                    ]
                }
            )

    # Build a default config

    conf = lhotse_conf.DataConfig(manifests=manifests)

    (dataset_kw, features_kw, splicing_kw, specaug_kw) = rnnt_config.input(
        cfg, split=pipeline
    )

    # Override default transcript in the manifests if needed
    use_transcript = cfg.get("use_transcripts", {})

    if keys := use_transcript.get("to_use", []):
        match pipeline, use_transcript.get("on_missing", "raise_error"):
            case "val", _:
                # For validation we allow fallback to default transcript
                conf.override_text_key = lhotse_conf.TryOverride(keys)
            case "train", "use_default":
                conf.override_text_key = lhotse_conf.TryOverride(keys)
            case "train", "raise_error":
                conf.override_text_key = lhotse_conf.RequiredOverride(keys)
            case "train", on_missing:
                raise ValueError(f"Invalid {on_missing=} in config")

    conf.force_finite = pipeline == "val"
    conf.initial_sort = pipeline == "val"

    match pipeline:
        case "val":
            conf.prime_seed = 42
            conf.shard_seed = 42

            # Validation requires torch distributed tensor sharing, for the
            # beam decoder. this is not supported for expandable segments so we
            # disable it.
            conf.cuda_expandable_segments = False
        case "train":
            # If multi GPU we want each rank to have a different seed
            conf.prime_seed = args.seed + global_rank
            conf.shard_seed = "randomized"

    # Suppress warning for validation
    conf.warn = pipeline == "train"

    conf.min_duration = dataset_kw.get("min_duration", 0.0)
    conf.max_duration = dataset_kw.get("max_duration", float("inf"))

    conf.min_chars = dataset_kw.get("min_transcript_len", 0)
    conf.max_chars = _inf_as_none(dataset_kw.get("max_transcript_len", None))

    conf.min_cps = dataset_kw.get("min_cps", 0.0)
    conf.max_cps = dataset_kw.get("max_cps", float("inf"))

    if pipeline == "val":
        conf.channel_selector = "mono_downmix"

    conf.sample_rate = dataset_kw["sample_rate"]

    conf.normalize = lhotse_conf.Normalization(
        config=normalize_config_from_full_yaml(cfg, mode=pipeline),
        charset=cfg["tokenizer"]["labels"],
    )

    match pipeline:
        case "val":
            # During validation prefix is managed by the decoder
            conf.conditional = None
            conf.background_noise = None
            conf.speed_perturbations = None
            conf.volume_perturbations = None
            conf.tempo_perturbations = None
        case "train":
            if not args.conditional:
                conf.conditional = None
            else:
                symbols, _ = get_all_user_tokens(cfg)

                if args.conditional_language:
                    # Drop `lang_` prefix from lang-codes, this allows shorter codes
                    # in the dataset yaml.
                    langs = {
                        k[5:]: v for k, v in symbols.items() if k.startswith("lang_")
                    }

                    if not langs:
                        raise ValueError(
                            "Conditional language requested but no `lang_XX` tokens found"
                        )
                else:
                    langs = None

                conf.conditional = lhotse_conf.Conditional(
                    user_symbols=symbols,
                    pnc_prob=args.pnc_prob,
                    languages=langs,
                    lang_mask_prob=args.lang_mask_prob,
                )

            if args.prob_background_noise <= 0.0:
                conf.background_noise = None
            else:
                conf.background_noise = lhotse_conf.BackgroundNoise(
                    hf_dataset=args.noise_dataset,
                    hf_split=args.noise_config,
                    snr=(args.noise_snr_low, args.noise_snr_high),
                    mix_prob=args.prob_background_noise,
                )

            conf.speed_perturbations = dataset_kw.get("speed_perturbation", None)
            conf.volume_perturbations = None
            conf.tempo_perturbations = None

    conf.min_tokens = dataset_kw.get("min_tokens", 1)
    conf.max_tokens = _inf_as_none(dataset_kw.get("max_tokens", None))

    conf.min_tps = dataset_kw.get("min_tps", 0.0)
    conf.max_tps = float(dataset_kw.get("max_tps", float("inf")))

    @beartype
    def or_inf(x: Optional[float]) -> float:
        return float("inf") if x is None else x

    @beartype
    def check(xargs: list[str]) -> None:
        for x in xargs:
            if getattr(args, x) is None:
                raise ValueError(
                    f"For --sampling_mode={args.sampling_mode}, please specify flag --{x}"
                )

    match pipeline:
        case "val":
            match (
                getattr(args, "val_batch_size", None),
                getattr(args, "val_batch_duration", None),
            ):
                case None, None:
                    raise ValueError("Must specify --val_batch_[size|duration]")
                case None, vbd:
                    conf.sampler = lhotse_conf.SimpleDurationBatch(
                        batch_size=lhotse_conf.DurationBatch(vbd)
                    )
                case vbs, None:
                    conf.sampler = lhotse_conf.SimpleBatch(batch_size=vbs)
                case vbs, vbd:
                    conf.sampler = lhotse_conf.SimpleDurationBatch(
                        batch_size=lhotse_conf.DurationBatch(vbd)
                    )

                    lhotse_warn(
                        "Ignoring --val_batch_size in favor of --val_batch_duration"
                    )

            conf.sampler_shuffle = None
            conf.sampler_drop_last = False

            conf.cut_concat = None
            conf.rir_augmentation = None

        case "train":
            match args.sampling_mode:
                case "duration":
                    check(["batch_duration"])
                    conf.sampler = lhotse_conf.SimpleDurationBatch(
                        batch_size=lhotse_conf.DurationBatch(args.batch_duration)
                    )
                case "fixed":
                    check(["batch_size"])
                    conf.sampler = lhotse_conf.SimpleBatch(
                        batch_size=args.batch_size,
                    )
                case "1D-bucketing":
                    check(["batch_duration", "buckets"])

                    conf.sampler = lhotse_conf.Bucketing1D(
                        batch_size=lhotse_conf.DurationBatch(args.batch_duration),
                        duration_bins=Path(args.buckets),
                    )
                case "2D-bucketing":
                    check(["buckets", "ols_vram_coeff"])

                    xargs = [
                        args.batch_vram,
                        args.batch_duration,
                        args.batch_tokens,
                        args.batch_area,
                    ]

                    if all(x is None or x == float("inf") for x in xargs):
                        raise ValueError(
                            "At least one of --batch_[vram|duration|tokens|area] "
                            "must be non-None and finite for 2D bucketing"
                        )

                    batch_size = lhotse_conf.AreaBatch(
                        max_vram=or_inf(args.batch_vram),
                        max_time=or_inf(args.batch_duration),
                        max_toks=or_inf(args.batch_tokens),
                        max_area=or_inf(args.batch_area),
                        coeff_time=args.ols_vram_coeff[0],
                        coeff_toks=args.ols_vram_coeff[1],
                        coeff_area=args.ols_vram_coeff[2],
                    )

                    conf.sampler = lhotse_conf.Bucketing2D(
                        batch_size=batch_size, area_bins=Path(args.buckets)
                    )

                case "1D-bucketing-dyn":
                    check(["batch_duration", "num_dur_buckets"])

                    conf.sampler = lhotse_conf.DynamicBucketing1D(
                        batch_size=lhotse_conf.DurationBatch(args.batch_duration),
                        num_buckets=args.num_dur_buckets,
                    )
                case "2D-bucketing-fixed":
                    check(["buckets"])

                    conf.sampler = lhotse_conf.FixedBucketing2D(
                        fixed_area_bins=Path(args.buckets)
                    )
                case "2D-bucketing-singlet":
                    check(["batch_size", "singlet_bucket", "buckets"])

                    conf.sampler = lhotse_conf.SingleBucket2D(
                        batch_size=args.batch_size,
                        bucket_idx=args.singlet_bucket,
                        area_bins=Path(args.buckets),
                    )

                case unk:
                    raise ValueError(f"Unknown sampler mode: {unk}")

            if args.prob_room_aug > 0.0:
                conf.rir_augmentation = lhotse_conf.RoomImpulseResponseAug(
                    path=Path(args.room_dataset) if args.room_dataset else None,
                    prob=args.prob_room_aug,
                    early_only=args.rir_early_only,
                )

    if args.loader_workers_per_gpu is None:
        if (n := cpu_count(logical=False)) is None:
            raise RuntimeError(
                "Please specify --loader_workers_per_gpu, detection failed"
            )
        else:
            deduced = int(n / world_size)

            if deduced > (lim := _MAX_DEDUCED_LOADER_WORKERS_PER_GPU):
                lhotse_warn(
                    f"Truncating deduced loader-workers-per-GPU from {deduced} -> {lim}"
                )

            conf.loader_workers = min(deduced, _MAX_DEDUCED_LOADER_WORKERS_PER_GPU)
    else:
        conf.loader_workers = args.loader_workers_per_gpu

    if pipeline == "val":
        conf.loader_force_dataset = lhotse_conf.DatasetKind.MAPPABLE

    if args.turn_off_initial_padding:
        conf.head_stream_padding = False

    if pipeline == "val":
        # Let train use default (no padding)
        conf.final_silence_padding = args.val_final_padding_secs

    if args.inspect_audio:
        conf.audio_dump_dir = Path("/results/augmented_audios")

    if pipeline == "train" and getattr(args, "inspect_memory_usage", None) is not None:
        conf.memory_dump_path = Path(args.inspect_memory_usage)

    if pipeline == "train" and getattr(args, "cuts_dump_path", None) is not None:
        conf.cuts_dump_path = Path(args.cuts_dump_path)

    if pipeline == "val" and getattr(args, "feature_dump_dir", None) is not None:
        conf.feature_dump_dir = Path(args.feature_dump_dir)

    conf.use_cuda = use_cuda

    conf.dither_coeff = features_kw["dither"]
    conf.n_fft = features_kw["n_fft"]
    conf.win_length = features_kw["window_size"]
    conf.hop_length = features_kw["window_stride"]
    conf.n_mels = features_kw["n_filt"]

    match pipeline:
        case "val":
            conf.mel_norm = lhotse_conf.GlobalNorm(Path(features_kw["stats_path"]))
            conf.spec_aug = None
        case "train":
            if args.norm_over_utterance:
                conf.mel_norm = lhotse_conf.UttNorm()
            elif args.norm_use_global_stats:
                conf.mel_norm = lhotse_conf.GlobalNorm(Path(features_kw["stats_path"]))
            elif args.norm_ramp_start is None or args.norm_ramp_end is None:
                lhotse_warn(
                    f"""\
                    One of --norm_ramp_start={args.norm_ramp_start} or
                    --norm_ramp_end={args.norm_ramp_end} is not set, falling
                    back to utterance-level normalization.\
                    """
                )
                conf.mel_norm = lhotse_conf.UttNorm()
            else:
                # Divide by world size as each GPU will see 1/world_size of the data.
                conf.mel_norm = lhotse_conf.BlendedNorm(
                    stats=Path(features_kw["stats_path"]),
                    ramp_beg=args.norm_ramp_start / world_size,
                    ramp_end=args.norm_ramp_end / world_size,
                )

            # The yaml config use the old names max_* instead of *_width
            conf.spec_aug = lhotse_conf.SpecAug(
                freq_masks=specaug_kw["freq_masks"],
                freq_width=specaug_kw["max_freq"],
                time_masks=specaug_kw["time_masks"],
                time_width=specaug_kw["max_time"],
            )

    if isinstance(conf.mel_norm, lhotse_conf.UttNorm):
        lhotse_warn("Utterance-level normalization is not supported on hardware")

    conf.feature_frame_stacking = splicing_kw["frame_stacking"]
    conf.encoder_frame_stacking = cfg["rnnt"]["enc_stack_time_factor"]

    if conf.feature_frame_stacking != splicing_kw["frame_subsampling"]:
        raise ValueError("Lhotse pipeline doesn't support frame subsampling")

    match pipeline, dataset_kw.get("append_EOI", False):
        case _, False:
            conf.eoi_append = None
        case "val", True:
            conf.eoi_append = lhotse_conf.AlwaysEOI()
        case "train", True:
            conf.eoi_append = lhotse_conf.ProbEOI(drop_fraction=args.eoi_drop_fraction)

    if (
        pipeline == "val"
        and conf.eoi_append is None
        and conf.final_silence_padding == 0.0
    ):
        lhotse_warn(
            """\
            This model was trained without end-of-input tokens but validation
            is running with --val_final_padding_secs=0.0. Legacy models
            typically require ~0.24s of silence padding to flush out all
            predictions. Please set --val_final_padding_secs accordingly to
            avoid degradation. \
            """
        )

    return conf, dataset_kw, features_kw


@beartype
def _parse_canary(data: dict) -> Optional[lhotse_conf.CanaryHyperParams]:
    """
    Parse a block of canary hyperparameters from a dataset yaml file
    """
    if "canary" not in data:
        return None

    match data["canary"].get("mode", "duration"):
        case "duration":
            mode = lhotse_conf.CanaryMode.DUR
        case "utterance":
            mode = lhotse_conf.CanaryMode.UTT
        case _:
            raise ValueError(f"Unknown canary mode: {data['canary']['mode']}")

    return lhotse_conf.CanaryHyperParams(
        alpha=data["canary"].get("alpha", 0.5), mode=mode
    )


@beartype
def _parse_filter_stats(opts: dict) -> Optional[lhotse_conf.FiltStats]:
    """
    Parse filtering statistics from a dataset yaml file
    """
    if "filtered" not in opts:
        return None

    return lhotse_conf.FiltStats(**opts["filtered"])


@beartype
def _iter_datasets(data: dict) -> Generator[tuple[str, dict], None, None]:
    """
    Iterate over dataset (name, opts) pairs from a dataset yaml file:
        name: str - the manifest name/path (a key in the yaml)
        opts: dict - a dict of optional manifest options
    """
    for manifest in data["datasets"]:
        # Each list item should be a dict with a single key-value pair
        assert len(manifest) == 1, manifest

        name, opts = list(manifest.items())[0]

        if opts is None:
            yield name, {}
        else:
            yield name, opts


@beartype
def _parse_datasets(data: dict) -> lhotse_conf.ManifestGroup:
    """
    Parse a dataset block from a dataset yaml file
    """
    manifests = []

    canary_params = _parse_canary(data)

    for name, opts in _iter_datasets(data):
        man = lhotse_conf.Manifest.build(
            path=name,
            data_dir=opts.get("data_dir", None),
            filt_stats=_parse_filter_stats(opts),
        )

        if "weight" in opts:
            manifests.append(lhotse_conf.WeightedManifest(man, weight=opts["weight"]))
        else:
            manifests.append(man)

    if canary_params is not None:
        return lhotse_conf.CanaryManifests(manifests=manifests, params=canary_params)

    return manifests


@beartype
def _select_train_langs(cfg: dict) -> Optional[List[str]]:
    """
    Pick which languages' training data to load.

    `ft_langs` is an optional config key used to resume a
    multilingual model on a subset of its original `langs`. When present, it
    selects which languages' training data to load, without touching
    `langs`, which continues to determine the tokenizer/ngram/mel-stats
    artifacts so they stay identical to the original (pre-resume) model.
    If `ft_langs` is present but empty then fall back to using `langs` for training.
    """
    if "ft_langs" in cfg and cfg["ft_langs"] != "":
        ft_langs = [lang for lang in (cfg["ft_langs"] or []) if lang]
        if not ft_langs:
            lhotse_warn(
                "Empty `ft_langs` in config, falling back to using `langs` for training"
            )
            return cfg.get("langs", None)
        else:
            return cfg["ft_langs"]
    return cfg.get("langs", None)


@beartype
def _parse_dataset_yaml(
    path: Path, langs: Optional[List[str]] = None
) -> lhotse_conf.MultiManifestGroup:
    """
    Parse the dataset yaml file
    """
    with path.open("r") as f:
        data = yaml.safe_load(f)

    if "languages" not in data:
        raise ValueError(
            "Manifests without language codes are no longer supported."
            "Please use the multilingual manifest format"
        )

    match data.get("format", "jsonl"):
        case "multilingual-shar":
            cls = lhotse_conf.MultilingualSharManifests
        case "multilingual-jsonl":
            cls = lhotse_conf.MultilingualJsonlManifests
        case _:
            raise ValueError(f"Unknown dataset format: {data.get('format')}")

    # Fallthrough for multilingual formats

    @beartype
    def drop_prefix(s: str) -> str:
        # Allow but don't require 'lang_' prefix in language codes
        # to match the user symbols convention.
        return s[5:] if s.startswith("lang_") else s

    manifest_langs = [drop_prefix(lang) for lang in data["languages"].keys()]
    if langs is None:
        langs = manifest_langs

    assert all(
        lang in manifest_langs for lang in langs
    ), "Requested languages not found in manifest"

    return cls(
        manifests={
            drop_prefix(k): _parse_datasets(v)
            for k, v in data["languages"].items()
            if drop_prefix(k) in langs
        },
        canary_params=_parse_canary(data),
    )


@beartype
def _inf_as_none(x: Any) -> Optional[int]:
    """
    In the legacy config, 'inf' is used to denote no limit for integers.
    """
    match x:
        case None:
            return None
        case int():
            return x
        case float() if x == float("inf"):
            return None
        case _:
            raise ValueError(f"Invalid config type, {x}, expected int or inf")
