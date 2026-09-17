import copy
from dataclasses import dataclass
from functools import partial

from beartype import beartype
from beartype.typing import Callable, Iterable, TypeVar
from lhotse import Seconds
from tqdm import tqdm

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.lhotse.config import DataConfig, SimpleBatch
from caiman_asr_train.lhotse.dataloader import get_lhotse_dataloader
from caiman_asr_train.lhotse.dataset import IdentityDataset
from caiman_asr_train.lhotse.stats import InfiniteStats


@beartype
def gather_utt_stats(
    config: DataConfig,
    tokenizer: Tokenizer,
    batch_size: int = 128,
    factor: int = 1,
) -> list[tuple[Seconds, int]]:
    """
    Sample (duration, tokens) from a data loader.
    """
    #
    config = copy.deepcopy(config)

    # Disable bucketing to force a simple loader
    config.sampler = SimpleBatch(batch_size=batch_size)

    loader, stats = get_lhotse_dataloader(
        config=config,
        global_rank=0,
        world_size=1,
        dataset=IdentityDataset(),
        tokenizer=tokenizer,
    )

    # Estimate number of batches so we can have a nice loader progress bar
    bound = stats.max_orbit_utt if isinstance(stats, InfiniteStats) else stats.total_utt
    bound *= factor
    est = 1 + bound // batch_size

    points: list[tuple[Seconds, int]] = []  # dur, toks
    count = 0

    for cuts in tqdm(loader, desc="Gathering sample statistics", total=est):
        # Generator
        tokens = (
            len(seg.tokens)
            for cut in cuts
            for seg in cut.supervisions
            if seg.has_custom("tokens")
        )

        durs = (cut.duration for cut in cuts)

        points.extend(zip(durs, tokens, strict=True))

        count += len(cuts)

        if count >= bound:
            # Because of length perturbation we may need more than one orbit
            break

    return points


@beartype
@dataclass
class OptimalBuckets:
    bins_1D: list[Seconds]
    bins_2D: list[tuple[Seconds, int]]


T = TypeVar("T")
V = TypeVar("V", float, int)


@beartype
def partition_into_buckets(
    stats: list[tuple[Seconds, int]], dur_bins: int, tok_bins: int = 1
) -> OptimalBuckets:
    """
    Partition utterance statistics into optimal 2D buckets and print a summary.
    """

    assert dur_bins > 1, "At least two duration bins are required"
    assert tok_bins > 0, "At least one token bin is required"

    # Partition into equal duration bins
    dur_parts, dur_bounds = _chunk(stats, dur_bins, key=_dur)

    tot_utt = len(stats)
    tot_dur = sum(map(_dur, stats))
    tot_tok = sum(map(_tok, stats))
    tot_vol = sum(map(_vol, stats))

    fmt_dbs = _fmt("{:>5.2f}")(dur_bounds)
    fmt_utt = _fmt("{:.1%}")(len(x) / tot_utt for x in dur_parts)
    fmt_dur = _fmt("{:.1%}")(sum(map(_dur, x)) / tot_dur for x in dur_parts)

    print(f"Duration boundary: {fmt_dbs}")
    print(f"Utt per 1D-bucket: {fmt_utt}")
    print(f"Dur per 1D-bucket: {fmt_dur}")

    bins_2D: list[tuple[Seconds, int]] = []

    def box(key: str) -> str:
        return f"[{key:<19}]"

    keys = ["token_bound...", "utts...", "duration...", "tokens...", "area..."]

    print("Key:")
    print(f"  Dur: {' '.join(map(box, keys))}")

    for part, dur in zip(dur_parts, dur_bounds):
        # Partition each time bin into equal token bins
        tok_parts, tok_bounds = _chunk(part, tok_bins, key=_tok)

        # Boundaries
        fmt_tbs = _fmt("{:>4}")(tok_bounds)
        fmt_utt = _fmt("{:.1%}")(len(x) / tot_utt for x in tok_parts)
        fmt_dur = _fmt("{:.1%}")(sum(map(_dur, x)) / tot_dur for x in tok_parts)
        fmt_tok = _fmt("{:.1%}")(sum(map(_tok, x)) / tot_tok for x in tok_parts)
        fmt_vol = _fmt("{:.1%}")(sum(map(_vol, x)) / tot_vol for x in tok_parts)

        print(f"{dur:5.2f}: {fmt_tbs} {fmt_utt} {fmt_dur} {fmt_tok} {fmt_vol}")

        bins_2D.extend((dur, tb) for tb in tok_bounds)

    return OptimalBuckets(bins_1D=dur_bounds, bins_2D=bins_2D)


@beartype
def _chunk(
    data: list[T], n: int, key: Callable[[T], V]
) -> tuple[list[list[T]], list[V]]:
    """
    Sort the input data by key and then greedily partition into `n` chunks such
    that the sum of the `key` values in each chunk is approximately equal.

    Returns:
        - The partitioned data as a list of lists.
        - The maximum `key` value in each chunk.
    """
    #
    bins: list[list[T]] = [[]]
    data = sorted(data, key=key)

    assert data, "No data to chunk"

    if n == 1:
        return [data], [key(data[-1])]

    acc = 0.0
    tot = sum(map(key, data))
    lim = tot / n

    for proj, elem in zip(map(key, data), data):
        acc += proj
        bins[-1].append(elem)

        if acc > lim:
            acc = 0.0
            bins.append([])

    bins = [b for b in bins if b]
    assert len(bins) == n, f"{len(bins)} != {n}"
    return bins, [key(b[-1]) for b in bins]


@beartype
def _lfmt(iter: Iterable, format_string: str):
    return f"[{' '.join(format_string.format(x) for x in iter)}]"


@beartype
def _fmt(format_string: str) -> Callable[[Iterable], str]:
    return partial(_lfmt, format_string=format_string)


def _dur(stat: tuple[Seconds, int]) -> Seconds:
    return stat[0]


def _tok(stat: tuple[Seconds, int]) -> int:
    return stat[1]


def _vol(stat: tuple[Seconds, int]) -> float:
    return stat[0] * stat[1]
