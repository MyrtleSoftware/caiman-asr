from functools import partial

import torch
from beartype import beartype
from beartype.typing import Literal, NamedTuple, Optional
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset import AudioSamples

from caiman_asr_train.core.batches import BatchFirst, TimeFirst
from caiman_asr_train.lhotse.config import DataConfig
from caiman_asr_train.utils.math import round_up


@beartype
class IdentityDataset(torch.utils.data.Dataset):
    """
    A dummy dataset that returns the input CutSet as is.
    """

    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


@beartype
class StreamingBatch(NamedTuple):
    # Metadata only (no audio/features in memory)
    cuts: CutSet

    # Waveforms
    audio: BatchFirst

    # Tokenized prefix+transcript (if tokenizer was provided)
    tokens: Optional[TimeFirst] = None

    # Just the prefix (if prefix tokenizing)
    prefix: Optional[TimeFirst] = None


@beartype
class StreamingDataset(torch.utils.data.Dataset):
    def __init__(self, config: DataConfig):
        """
        In the Lhotse paradigm a dataset only performs the mapping from CutSet
        (minibatch) to audio/features. As we do feature extraction on the GPU,
        our dataset just does the audio loading/padding and batch aggregation.

        If config.dataset_workers > 0, the IO will be done in thread pool.
        """
        self.config = config
        self.audio = AudioSamples(num_workers=config.dataset_workers)
        self.head_silence_samples = config.win_sample_length - config.hop_sample_length

    def __getitem__(self, cuts: CutSet) -> StreamingBatch:
        """
        Returns:
            cuts (CutSet) - With in-memory data dropped (metadata only).

            audio (B, T).
            tokens (B, U) - optional.
            prefix_tokens (B, P) - optional.
        """

        # Such that first frame only requires hop_length samples for first frame
        # at inference time.
        if self.config.head_stream_padding:
            cuts = cuts.map(
                partial(
                    _pad,
                    extra_samples=self.head_silence_samples,
                    direction="left",
                )
            )

        # Mimic sending this much silence into microphone at the end of the stream.
        if self.config.final_silence_padding > 0.0:
            cuts = cuts.map(
                partial(
                    _pad,
                    extra_samples=self.config.final_silence_padding_samples,
                    direction="right",
                )
            )

        # Such that the last frame is complete -> pad the window with silence.
        # Because of the N-factor frame stacking from mel-features to encoder
        # output, it is required to pad to a multiple of N frames.
        if self.config.tail_stream_padding:
            cuts = cuts.map(
                partial(
                    _tail_pad,
                    multiple_of=self.config.total_frame_stacking,
                    hop_length=self.config.hop_sample_length,
                    n_fft=self.config.n_fft,
                )
            )

        # This function is responsible for IO
        audio, audio_lens = self.audio(cuts)  # type: ignore

        # The downstream expects int32 in the joint-net fwd pass
        tokens = self._extract_custom_as_batch(cuts, key="tokens", dtype=torch.int32)
        prefix = self._extract_custom_as_batch(cuts, key="prefix", dtype=torch.int32)

        return StreamingBatch(
            cuts=cuts.drop_in_memory_data(),
            audio=BatchFirst(data=audio, lens=audio_lens),
            tokens=tokens,
            prefix=prefix,
        )

    def _extract_custom_as_batch(
        self, cuts: CutSet, key: str, dtype: torch.dtype
    ) -> Optional[TimeFirst]:
        """
        Extracts a custom field from supervisions and returns it as a TimeFirst batch.
        """
        maybe_custom = [
            torch.from_numpy(getattr(seg, key))
            for cut in cuts.data
            for seg in cut.supervisions
            if seg.has_custom(key)
        ]

        if not maybe_custom:
            return None

        # Expectation is 1-2-1
        assert len(maybe_custom) == len(cuts)

        custom = torch.nn.utils.rnn.pad_sequence(maybe_custom, batch_first=False)
        custom_lens = torch.tensor([x.numel() for x in maybe_custom], dtype=dtype)

        return TimeFirst(data=custom, lens=custom_lens)


@beartype
def _tail_pad(cut: Cut, multiple_of: int, hop_length: int, n_fft: int) -> Cut:
    # The mel formula is n -> 1 + (n - n_fft) // hop_length
    # we want the result to be a multiple of `multiple_of`
    #
    # 1 + (n - n_fft) // hop_length = m * multiple_of

    assert (n := cut.num_samples) is not None

    # Pad to whole number of hops
    m = round_up(n - n_fft, multiple_of=hop_length) + n_fft
    assert (m - n_fft) % hop_length == 0

    # Pad up such that the number of hops is a multiple of `multiple_of`
    raw_hops = 1 + (m - n_fft) // hop_length
    pad_hops = round_up(raw_hops, multiple_of=multiple_of)
    m = m + hop_length * (pad_hops - raw_hops)
    assert (1 + (m - n_fft) // hop_length) % multiple_of == 0

    # Sanity check
    assert m >= n

    return cut.pad(num_samples=m, direction="right")


@beartype
def _pad(
    cut: Cut,
    extra_samples: int,
    direction: Literal["left", "right", "both"],
) -> Cut:
    """
    Pad with zeros in the audio domain.
    """
    assert cut.num_samples is not None
    return cut.pad(num_samples=cut.num_samples + extra_samples, direction=direction)
