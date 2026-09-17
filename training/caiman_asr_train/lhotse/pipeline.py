import json
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from statistics import mean

import numpy as np
import soundfile as sf
import torch
from beartype import beartype
from beartype.typing import Generator, NamedTuple, Optional
from lhotse import CutSet

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.lhotse.config import (
    AlwaysEOI,
    Bucketing1D,
    DataConfig,
    DurationBatch,
    DynamicBucketing1D,
    FixedBucketing2D,
    ProbEOI,
    SimpleBatch,
    SimpleDurationBatch,
)
from caiman_asr_train.lhotse.dataloader import get_lhotse_dataloader
from caiman_asr_train.lhotse.dataset import StreamingBatch, StreamingDataset
from caiman_asr_train.lhotse.features import MelNormalizer, MelSpectrogram
from caiman_asr_train.lhotse.spec_augment import SpecAugment
from caiman_asr_train.lhotse.stats import EpochStats, InfiniteStats
from caiman_asr_train.utils.math import ceil_div


@beartype
@dataclass
class Metadata:
    """
    Reporting of things used for misc purposes i.e. logging of the pipeline stats.
    """

    total_dur: float
    total_its: int
    batch_dur: float

    mel_alpha: Optional[float]


@beartype
class PipelineBatch(NamedTuple):
    """
    A Batch of data produced by the pipeline.
    """

    meta: Metadata
    cuts: CutSet  # Meta-data only (no audio/feats)

    # Spectrograms
    feats: TimeFirst

    # Tokenized
    tokens: Optional[TimeFirst] = None
    prefix: Optional[TimeFirst] = None


class UnknownLengthError(Exception):
    """
    Cannot estimate the length of the pipeline.
    """

    pass


class InfiniteLengthError(Exception):
    """
    The length is not really defined, you can still observe some stats though.
    """

    pass


@beartype
@dataclass
class ResumeInfo:
    """
    A lhotse pipeline is deterministic given the same input config with fixed
    seeds. However, it is not possible to resume it from an arbitrary iteration
    without iteration back to that point. Hence, when a train is resumed from a
    checkpoint the data order will not be the same as the initial run however,
    any subsequent resumptions from the same point will have the same data
    order as each other.

    This structure holds the information to restore the state of any augmentation
    schedules.

    Members:
        total_dur: The total duration of audio processed so far.
        total_its: The total number of iterations processed so far.
    """

    total_dur: float = 0.0
    total_its: int = 0


_NO_RESUME = ResumeInfo()


@beartype
class LhotsePipeline:
    def __init__(
        self,
        config: DataConfig,
        global_rank: int = 0,
        world_size: int = 1,
        tokenizer: Optional[Tokenizer] = None,
        resume_info: ResumeInfo = _NO_RESUME,
    ):
        """
        Build the data pipeline, this is a mostly drop-in replacement of the
        dali pipeline, it produces an iterable that returns batches of
        network inputs (features) and expected outputs (tokens). Various
        augmentations and regularizations are applied on-the-fly.

        Args:
            config: The data configuration.
            global_rank: The global rank of this process.
            world_size: The total number of processes.
            tokenizer: The tokenizer to use for tokenizing the text (if None, no tokens).
            resume_info: Information for resuming from a checkpoint.

        """
        # Sanity
        if config.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but config.use_cuda=True")

        self.config = deepcopy(config)
        self.global_rank = global_rank
        self.world_size = world_size
        self.resume_info = resume_info

        # Offset the seeds by the total_its to prevent repeating the order
        if isinstance(self.config.prime_seed, int):
            self.config.prime_seed += resume_info.total_its

        if isinstance(self.config.shard_seed, int):
            self.config.shard_seed += resume_info.total_its

        # This just loads/pads the audio/tokens from the cutset on-the-fly
        dataset = StreamingDataset(self.config)

        self.dloader, self.stats = get_lhotse_dataloader(
            config=self.config,
            global_rank=global_rank,
            world_size=world_size,
            dataset=dataset,
            tokenizer=tokenizer,
        )

        self.mel_spectrogram = MelSpectrogram(config=self.config)
        self.mel_normalizer = MelNormalizer(config=self.config)

        if (sc := self.config.spec_aug) is not None:
            self.spec_augment = SpecAugment(
                freq_masks=sc.freq_masks,
                freq_width=sc.freq_width,
                time_masks=sc.time_masks,
                time_width=sc.time_width,
            )
        else:
            self.spec_augment = None

        if self.config.use_cuda:
            self.mel_spectrogram = self.mel_spectrogram.cuda()
            self.mel_normalizer = self.mel_normalizer.cuda()

            if self.spec_augment is not None:
                self.spec_augment = self.spec_augment.cuda()

        if self.config.eoi_append is not None:
            eoi = torch.arange(end=self.config.n_mels, dtype=torch.int32)
            eoi = 2 * (eoi % 2) - 1
            eoi = eoi.to(dtype=torch.float32)
            eoi = torch.cat([eoi] * self.config.total_frame_stacking)
            eoi = eoi[None, None, :]
            eoi = self.config.maybe_cuda(eoi)

            # Help the type checker
            assert isinstance(eoi, torch.Tensor)

            self.eoi = eoi

    def get_stats(self) -> EpochStats:
        """
        Fetch the global dataset statistics.
        """
        return self.stats

    def _maybe_dump_audio(
        self, n: int, audio: torch.Tensor, audio_lens: torch.Tensor
    ) -> None:
        if self.config.audio_dump_dir is None:
            return

        self.config.audio_dump_dir.mkdir(parents=True, exist_ok=True)

        for i in range(audio.shape[0]):
            length = audio_lens[i].item()
            wav = audio[i, :length].numpy(force=True)
            name = f"debug_{self.global_rank}_{n}.flac"

            with open(self.config.audio_dump_dir / name, "wb") as f:
                sf.write(f, wav, 16000)

    def _maybe_dump_memory(self, cuts: CutSet, token_lens: Optional[torch.Tensor]):
        if self.config.memory_dump_path is None:
            return

        self.config.memory_dump_path.parent.mkdir(parents=True, exist_ok=True)

        B = len(cuts)

        durs = [cut.duration for cut in cuts.data]

        sum_durs = sum(durs)
        squ_durs = B * max(durs)

        free, total = torch.cuda.mem_get_info()
        mem = (total - free) / 1024**2

        def write(line: str):
            with self.config.memory_dump_path.open("a") as f:
                print(line, file=f)

        if token_lens is None:
            write(f"{B},{mem},{sum_durs},{squ_durs}")
        else:
            toks = [int(x) for x in token_lens]
            sum_toks = sum(toks)
            squ_toks = B * max(toks)

            area = sum(d * t for d, t in zip(durs, toks))

            write(
                f"{B},{mem},{sum_durs},{squ_durs},{sum_toks},{squ_toks},{area}",
            )

    def _maybe_dump_features(
        self, cuts: CutSet, feats: torch.Tensor, feat_lens: torch.Tensor
    ):
        if self.config.feature_dump_dir is None:
            return

        self.config.feature_dump_dir.mkdir(parents=True, exist_ok=True)

        _dump_counter = getattr(self, "_dump_counter", count())

        for b, _ in enumerate(cuts.data):
            length = feat_lens[b].item()
            feat = feats[:length, b, :].detach().cpu().numpy()
            idx = next(_dump_counter)
            np.savez_compressed(
                self.config.feature_dump_dir / f"feats_{idx:06d}.npz", logmels=feat
            )
        self._dump_counter = _dump_counter

    def _maybe_dump_cuts(self, n: int, cuts: CutSet) -> None:
        """
        Dump the cuts to a JSON file for debugging purposes.

        This function dumps the cuts to a JSON file. Each cut's ID, text,
        and normalized text are recorded along with the batch number and
        rank of the process.
        If the cuts have identical text and normalized text, or
        have no normalized text, they are skipped.

        Each rank writes to its own file (suffixed with its rank) rather
        than a shared path, since concurrent appends from multiple DDP
        processes to the same file are not guaranteed atomic and can
        interleave/corrupt lines.

        Args:
            n: The batch number.
            cuts: The CutSet containing the cuts to be dumped.
        """
        if self.config.cuts_dump_path is None:
            return

        path = self.config.cuts_dump_path
        rank_path = path.with_name(f"{path.stem}_rank{self.global_rank}{path.suffix}")
        rank_path.parent.mkdir(parents=True, exist_ok=True)

        with rank_path.open("a", encoding="utf-8") as f:
            for cut in cuts.data:
                for sup in cut.supervisions:
                    if sup.text is None:
                        continue
                    record = {
                        "batch": n,
                        "rank": self.global_rank,
                        "cut_id": cut.id,
                        "text": sup.text,
                        "normalized_text": (
                            sup.normalized_text
                            if sup.normalized_text is not None
                            else ""
                        ),
                    }
                    print(json.dumps(record, ensure_ascii=False), file=f)

    def __len__(self) -> int:
        """
        Estimate the total length if we are a finite pipeline, otherwise
        raise `InfiniteLengthError`, may also raise .
        """
        if self.world_size == 1:
            return self._len_single_gpu()

        return ceil_div(self._len_single_gpu(), by=self.world_size)

    def _len_single_gpu(self) -> int:
        if isinstance(self.stats, InfiniteStats):
            raise InfiniteLengthError()

        match self.config.sampler:
            case SimpleBatch(batch_size):
                size = batch_size
            case SimpleDurationBatch(batch_size):
                size = batch_size
            case DynamicBucketing1D() | Bucketing1D():
                size = self.config.sampler.batch_size
            case FixedBucketing2D():
                # Assume equal bucket distribution
                size = mean(x[0] for x in self.config.sampler.bins)
            case _:
                raise ValueError(f"__len__ not available for {self.config.sampler}")

        match size, self.config.sampler_drop_last:
            case int(n), True:
                return self.stats.total_utt // n
            case int(n), False:
                return ceil_div(self.stats.total_utt, by=n)
            case float(n), _:
                # Don't bother with drop_last (this is approx anyway)
                return int(self.stats.total_utt / n + 0.5)
            case DurationBatch(dur, quad), _:
                if quad is not None:
                    raise UnknownLengthError()

                # Don't bother with drop_last (this is approx anyway)
                return int(self.stats.total_dur / dur + 0.5)

    def __iter__(self) -> Generator[PipelineBatch]:
        """
        Yield batches of data from the pipeline.
        """
        total_dur = self.resume_info.total_dur

        @beartype
        def type_hint() -> Generator[tuple[int, StreamingBatch]]:
            """
            This lets the type checker know the data loader's yield type.
            """
            for it, batch in enumerate(self.dloader, start=self.resume_info.total_its):
                yield it, batch

        for total_its, (cuts, audio, tokens, prefix) in type_hint():
            # Should only run for debugging
            self._maybe_dump_audio(total_its, *audio.as_tuple())
            self._maybe_dump_memory(cuts, tokens.lens if tokens is not None else None)
            self._maybe_dump_cuts(total_its, cuts)

            # Use non_blocking as we transfer multiple tensors
            # This is only async WRT to the host (device still sync)

            if self.config.use_cuda:
                audio = audio.to(device="cuda", non_blocking=True)

                if tokens is not None:
                    tokens = tokens.to(device="cuda", non_blocking=True)

                if prefix is not None:
                    prefix = prefix.to(device="cuda", non_blocking=True)

            batch_dur = sum(cut.duration for cut in cuts.data)
            total_dur = total_dur + batch_dur

            # Audio -> Spectrogram
            feats, feat_lens = self.mel_spectrogram(*audio.as_tuple())

            feats, mel_alpha = self.mel_normalizer(
                feats, feat_lens, total_dur=total_dur
            )

            if self.spec_augment is not None:
                feats = self.spec_augment(feats, feat_lens)

            if not self.config.eoi_append:
                stack_factor = self.config.feature_frame_stacking
            else:
                stack_factor = self.config.total_frame_stacking

            feats = torch.einsum("B F T -> B T F", feats).contiguous()

            # Frame stacking, can use view based impl as contiguity is guaranteed
            # If we are adding EOI in encoder frame rate we can fuse frame stacks
            feats, feat_lens = _contiguous_frame_fold(
                feats, feat_lens, stack_factor, _Kind.STACK
            )

            if self.config.eoi_append is not None:
                # We are already in over-stacked frame rate
                B, _, F = feats.shape

                # If every single one must have EOI we must grow the array.
                # Ok to use garbage as only inside feat_lens is read.
                nulls = torch.zeros((1, 1, 1), dtype=feats.dtype, device=feats.device)
                nulls = nulls.expand(B, 1, F)
                feats = torch.concat([feats, nulls], dim=1)

                # Actually add the EOI frames
                index = feat_lens[:, None, None].to(torch.int64).expand_as(feats)
                feats.scatter_(dim=1, index=index, src=self.eoi.expand_as(feats))

                match self.config.eoi_append:
                    case AlwaysEOI():
                        feat_lens += 1
                    case ProbEOI(drop_fraction=p) if 0.0 < p < 1.0:
                        feat_lens += torch.rand_like(feat_lens, dtype=torch.float32) > p
                    case _:
                        raise ValueError(f"Bad EOI config: {self.config.eoi_append}")

                # Back down to feature frame rate
                feats, feat_lens = _contiguous_frame_fold(
                    feats, feat_lens, self.config.encoder_frame_stacking, _Kind.SPLIT
                )

                # Might be able to truncate
                feats = feats[:, : torch.max(feat_lens).item(), :]

            # Downstream expects permuted
            feats = torch.einsum("B T F -> T B F", feats)

            self._maybe_dump_features(cuts, feats, feat_lens)

            yield PipelineBatch(
                meta=Metadata(
                    total_dur=total_dur,
                    total_its=total_its,
                    batch_dur=batch_dur,
                    mel_alpha=mel_alpha,
                ),
                cuts=cuts,
                feats=TimeFirst(data=feats, lens=feat_lens),
                tokens=tokens,
                prefix=prefix,
            )


class _Kind(Enum):
    STACK = auto()
    SPLIT = auto()


@beartype
def _contiguous_frame_fold(x: torch.Tensor, x_lens: torch.Tensor, n: int, kind: _Kind):
    """
    Move the data <-> time partitioning of a tensor.

    This function does not actually do any copying; it returns a view of the
    input tensor with reshaped dimensions.
    """

    if n == 1:
        # Short-circuit, n == 1 is a noop
        return x, x_lens

    # Strictly only needs contiguity in the last two dims
    assert x.is_contiguous() and x.stride(-1) == 1

    T, H = x.shape[-2:]

    if kind == _Kind.STACK:
        assert torch.all(x_lens % n == 0)
        assert T % n == 0
        T = T // n
        H = H * n
        x_lens = x_lens // n
    else:
        assert H % n == 0
        T = T * n
        H = H // n
        x_lens = x_lens * n

    return x.view(*x.shape[:-2], T, H), x_lens
