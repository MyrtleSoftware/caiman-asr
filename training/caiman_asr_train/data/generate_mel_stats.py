import os
from argparse import Namespace

import torch
import torch.distributed as dist
from beartype import beartype
from tqdm import tqdm

from caiman_asr_train.args.norm_stats_generation import stats_generation_parse_args
from caiman_asr_train.lhotse.config import (
    DataConfig,
    DurationBatch,
    SimpleDurationBatch,
)
from caiman_asr_train.lhotse.config_build import build_from_legacy_sources
from caiman_asr_train.lhotse.pipeline import LhotsePipeline
from caiman_asr_train.lhotse.stats import FiniteStats, InfiniteStats
from caiman_asr_train.rnnt import config
from caiman_asr_train.train_utils.distributed import (
    get_rank_or_zero,
    get_world_size_or_one,
    start_ddp_with_gpu,
)
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun


@beartype
def generate_stats(args: Namespace):
    """
    Record training data log mel stats and save them to disk.
    """

    world_size = get_world_size_or_one()
    global_rank = get_rank_or_zero()

    model_config = config.load(args.model_config)

    # ft_langs (if present) restricts training data to a subset of `langs`
    # for fine-tuning a resumed model. The mel stats must instead always be
    # computed over all of `langs`, so that they match the original model.
    model_config.pop("ft_langs", None)

    conf, *_ = build_from_legacy_sources(
        args,
        model_config,
        "train",
        use_cuda=args.loader_train_device == "gpu",
        global_rank=global_rank,
        world_size=world_size,
    )

    # Batch duration is configurable via mel_batch_duration (default: 1 hour)
    batch_duration: float = getattr(args, "mel_batch_duration", 3600.0)

    # 10% safety margin for stream padding etc
    max_nfeats = batch_duration / conf.hop_length / conf.feature_frame_stacking
    max_nfeats = int(1.1 * max_nfeats) + 1

    conf = _overwrite_for_mel_stats_gen(conf, batch_duration)

    loader = LhotsePipeline(config=conf, global_rank=global_rank, world_size=world_size)

    match loader.get_stats():
        case FiniteStats(total_dur=total_dur):
            # The iterator will become exhausted
            est_dur = total_dur
        case InfiniteStats(max_orbit_dur=mod):
            est_dur = mod

    est_dur = min(est_dur, args.mel_duration_sample) / world_size

    # ===================================================

    match args.loader_train_device:
        case "gpu":
            device = "cuda"
        case "cpu":
            device = "cpu"
        case other:
            raise ValueError(f"Unknown loader_train_device: {other}")

    # We cannot use the naive reduction to compute mel stats due to catastrophic
    # numerical cancellation between sum and sum of squares. Instead we use a
    # combination of Welford's algorithm and Chen's parallel reduction, see:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    meldim = conf.n_mels * conf.feature_frame_stacking

    # Welford's aggregates (we track `meldim * max_nfeats` in parallel):
    n1 = torch.zeros(max_nfeats, meldim, dtype=torch.int64, device=device)
    m1 = torch.zeros(max_nfeats, meldim, dtype=torch.float64, device=device)
    m2 = torch.zeros(max_nfeats, meldim, dtype=torch.float64, device=device)

    with tqdm(
        total=int(est_dur / 3600 + 0.5),
        desc="Iterating audio",
        disable=(global_rank != 0),
    ) as bar:
        for meta, _, feats_batch, *_ in loader:
            # feats are (T, B, H), pack to (*, H) and remove the padding
            feats = _pack(*feats_batch.as_tuple()).to(dtype=torch.float64)

            assert (N := feats.shape[0]) <= max_nfeats

            # Welford's update:
            n1[:N] += 1
            d1 = feats - m1[:N]
            m1[:N] += d1 / n1[:N].to(dtype=torch.float64)
            d2 = feats - m1[:N]
            m2[:N] += d1 * d2

            bar.update(meta.batch_dur / 3600)

            if meta.total_dur > est_dur:
                break

    # ===================================================

    # Reductions require that no zeros are present.
    n1, m1, m2 = _drop_zero_sentinels(n1, m1, m2)
    # Use Chen's reduction to combine independent local Welford aggregates.
    n1, m1, m2 = _chen_tree_reduce(n1, m1, m2)

    def unstack(x):
        return x[0].contiguous().view(conf.feature_frame_stacking, -1)

    # Undo time stacking and reduce across stacked frames.
    n1, m1, m2 = _chen_tree_reduce(*map(unstack, (n1, m1, m2)))

    # Merge per-rank Welford aggregates using Chen's reduction.
    n1, m1, m2 = _merge_welford_across_ranks(n1, m1, m2)

    # Finalize Welford's aggregates into means and variances
    meln = n1[0]
    melmeans = m1[0]
    melvars = m2[0] / (n1[0] - 1).to(dtype=torch.float64)

    if global_rank == 0:
        print(f"Means:\n{melmeans}")
        print(f"Vars:\n{melvars}")

    # ===================================================

    # Calculated as doubles for precision; convert to float32s for use
    meln = meln.to(device="cpu", dtype=torch.float32)
    melmeans = melmeans.to(device="cpu", dtype=torch.float32)
    melvars = melvars.to(device="cpu", dtype=torch.float32)

    assert not torch.isnan(melmeans).any(), "NaN in mel means"
    assert not torch.isnan(melvars).any(), "NaN in mel vars"

    output_dir = args.output_dir

    if global_rank == 0:
        print(f"\nSaving generated mel stats to {output_dir}")
        # create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save as PyTorch tensors
        torch.save(melmeans, f"{output_dir}/melmeans.pt", pickle_protocol=5)
        torch.save(melvars, f"{output_dir}/melvars.pt", pickle_protocol=5)
        torch.save(meln, f"{output_dir}/meln.pt", pickle_protocol=5)

    if dist.is_initialized():
        dist.barrier()
        if not getattr(args, "skip_init", False):
            dist.destroy_process_group()


@beartype
def _drop_zero_sentinels(
    n1: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Drop region where no data was accumulated.
    """
    assert not torch.all(n1 == 0), "No data accumulated for mel stats"

    # Since n1 is updated identically across all columns (via broadcast on line 83),
    # verify that all columns are indeed identical
    assert torch.all(n1 == n1[:, :1]), "All columns should be identical"

    # Use the first column as row mask
    row_mask = n1[:, 0] != 0

    n1 = n1[row_mask]
    m1 = m1[row_mask]
    m2 = m2[row_mask]

    return n1, m1, m2


@beartype
def _chen_tree_reduce(
    n1: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inputs are all shape [N, *], this function reduces them to [1, *] using
    Chen's parallel reduction over a balanced binary tree.
    """

    assert not torch.any(n1 == 0), "zeros could lead to NaNs"

    def f64(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.float64)

    while n1.shape[0] > 1:
        n1a, n1b = torch.chunk(n1, 2, dim=0)
        m1a, m1b = torch.chunk(m1, 2, dim=0)
        m2a, m2b = torch.chunk(m2, 2, dim=0)

        odd = n1a.shape[0] != n1b.shape[0]

        if n1a.shape[0] < n1b.shape[0]:
            # Swap to ensure a is the larger chunk
            n1a, n1b = n1b, n1a
            m1a, m1b = m1b, m1a
            m2a, m2b = m2b, m2a

        M = min(n1a.shape[0], n1b.shape[0])

        n1a, n1_rem = n1a[:M], n1a[M:]
        m1a, m1_rem = m1a[:M], m1a[M:]
        m2a, m2_rem = m2a[:M], m2a[M:]

        # The actual step:
        n1 = n1a + n1b
        n1a, n1b = f64(n1a), f64(n1b)
        m1 = (n1a * m1a + n1b * m1b) / f64(n1)
        m2 = m2a + m2b + n1a * n1b / f64(n1) * (m1a - m1b) ** 2

        if odd:
            # Move remainder to front so it gets included in next iteration
            n1 = torch.cat([n1_rem, n1], dim=0)
            m1 = torch.cat([m1_rem, m1], dim=0)
            m2 = torch.cat([m2_rem, m2], dim=0)

    return n1, m1, m2


@beartype
def _merge_welford_across_ranks(
    n1: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not dist.is_initialized():
        return n1, m1, m2

    n_gather = [torch.zeros_like(n1) for _ in range(dist.get_world_size())]
    m1_gather = [torch.zeros_like(m1) for _ in range(dist.get_world_size())]
    m2_gather = [torch.zeros_like(m2) for _ in range(dist.get_world_size())]

    dist.all_gather(n_gather, n1)
    dist.all_gather(m1_gather, m1)
    dist.all_gather(m2_gather, m2)

    n1 = torch.cat(n_gather, dim=0)
    m1 = torch.cat(m1_gather, dim=0)
    m2 = torch.cat(m2_gather, dim=0)

    return _chen_tree_reduce(n1, m1, m2)


@beartype
def _overwrite_for_mel_stats_gen(conf: DataConfig, batch_duration: float) -> DataConfig:
    """
    Overwrite config values not relevant for mel stats generation.
    """

    # Transcripts are irrelevant
    conf.normalize = None
    conf.conditional = None

    # Disable augmentations
    conf.background_noise = None
    conf.speed_perturbations = None
    conf.volume_perturbations = None
    conf.tempo_perturbations = None

    # Simple sampler
    conf.sampler = SimpleDurationBatch(
        batch_size=DurationBatch(max_duration=batch_duration),
    )

    # Post batch augmentations
    conf.cut_concat = None
    conf.rir_augmentation = None

    # Deterministic
    conf.dither_coeff = 0.0

    # We want plain-mel features
    conf.mel_norm = None
    conf.spec_aug = None

    # Don't want EOI in our stats
    conf.eoi_append = None

    return conf


@beartype
def _pack(x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
    """
    Pack a padded tensor by removing padding.
    Args:
        x: Tensor of shape (T, B, H)
        x_lens: Tensor of shape (B,)
    Returns:
        Packed tensor of shape (*, H)
    """

    # Need C-memory order for the output view to have the correct order
    x = x.contiguous()

    T, _, H = x.shape

    # Create mask values using vectorized indexing
    idxs = torch.arange(T, device=x.device)[:, None]
    mask = idxs < x_lens[None, :]
    mask = mask[:, :, None]

    return torch.masked_select(x, mask).view(-1, H)


if __name__ == "__main__":
    args = stats_generation_parse_args()
    if args.loader_train_device == "gpu":
        maybe_restart_with_torchrun(
            args.num_gpus,
            args.called_by_torchrun,
            "/workspace/training/caiman_asr_train/data/generate_mel_stats.py",
        )
    if args.loader_train_device == "gpu":
        multi_gpu = args.num_gpus > 1
        _ = start_ddp_with_gpu(
            multi_gpu,
            args.local_rank,
            skip_init=False,
            timeout={"weeks": 52},
        )
    generate_stats(args)
