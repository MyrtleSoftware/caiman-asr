from enum import Enum

import nvidia.dali.ops as ops
import torch
from beartype.typing import Tuple
from nvidia.dali import fn as dali_fn

from caiman_asr_train.data.mel_stats import MelStats


class NormType(Enum):
    """
    Type of log mel normalization to use.

    DATASET_STATS: use pre-computed dataset statistics (mean and std) for each mel bin and
        normalize each mel bin independently. Default at inference time.
    UTTERANCE_STATS: use an utterance-specific mean and std for each mel bin that is
        calculated over the time-dimension of the mel spectrogram. Models trained with
        this are not suitable for streaming. See --norm_over_utterance docs.
    BLENDED_STATS: blend between the two normalization types on a schedule. Training
        only.
    """

    DATASET_STATS = 0
    UTTERANCE_STATS = 1
    BLENDED_STATS = 2


# ANCHOR: MelFeatNormalizer_in_mdbook
class MelFeatNormalizer:
    """
    Perform audio normalization, optionally blending between two normalization types.

    The two types of normalization are:
        1. use pre-computed NormType.DATASET_STATS per mel bin and normalize each
        timestep independently
        2. use utterance-specific NormType.UTTERANCE_STATS per mel bin that are
        calculated over the time-dimension of the mel spectrogram

    The first of these is used for validation/inference. The second method isn't
    streaming compatible but is more stable during the early stages of training.
    Therefore, by default, the training script blends between the two methods on a
    schedule.

    ANCHOR_END: MelFeatNormalizer_in_mdbook

    The weighting of the two normalization methods `dataset_to_utt_ratio` is a float in
    [0, 1] where 0 means only utterance-specific stats are used and 1 means only the
    dataset stats are used. stats_to_utt_ratio is calculated based on the current step.

    This class is responsible for calculating stats_to_utt_ratio as well as performing
    the blended normalization in dali.
    """

    def __init__(
        self,
        mel_stats: MelStats | None,
        ramp_start_step: int | None,
        ramp_end_step: int | None,
        batch_size: int,
        starting_ratio: float,
        norm_type: NormType = NormType.BLENDED_STATS,
    ):
        if mel_stats is None:
            assert norm_type == NormType.UTTERANCE_STATS
        if ramp_start_step is None or ramp_end_step is None:
            assert (
                norm_type != NormType.BLENDED_STATS
            ), "Ramp params are required when using blended stats"
        self.ramp_start_step = ramp_start_step
        self.ramp_end_step = ramp_end_step
        self.batch_size = batch_size
        self.starting_ratio = starting_ratio
        self.mel_stats = mel_stats
        self.type = norm_type

        self._step = 0

        self.gen_weights_batched = ops.ExternalSource(self, num_outputs=2)

    @property
    def dataset_to_utt_ratio(self) -> float:
        return self._calc_ratio(self._step)

    def _calc_ratio(self, step: int) -> float:
        if self.type == NormType.DATASET_STATS:
            return 1.0
        elif self.type == NormType.UTTERANCE_STATS:
            return 0.0
        else:
            return self._calc_blended_ratio(step)

    def _calc_blended_ratio(self, step: int) -> float:
        if step <= self.ramp_start_step:
            return self.starting_ratio
        elif step >= self.ramp_end_step:
            return 1.0
        # linearly increase ratio from start_ratio to 1 over the blend period
        return self.starting_ratio + (step - self.ramp_start_step) / (
            self.ramp_end_step - self.ramp_start_step
        ) * (1 - self.starting_ratio)

    def __call__(self, input):
        """
        Perform the blended normalization.
        """
        utt_norm = dali_fn.normalize(input, axes=[1])
        if self.type == NormType.UTTERANCE_STATS:
            return utt_norm
        mel_weight, utt_weight = self.gen_weights_batched()
        assert self.mel_stats is not None
        mel_norm = dali_fn.normalize(
            input,
            axes=[1],
            mean=self.mel_stats.means.unsqueeze(-1),
            stddev=self.mel_stats.stddevs.unsqueeze(-1),
        )
        return mel_weight * mel_norm + utt_weight * utt_norm

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return batched tensors of (dataset_to_utt_ratio, 1 - dataset_to_utt_ratio).

        This is necessary because dali requires numpy arrays rather than python floats
        inside the pipeline object.
        """
        tensor_weight_batched = torch.tensor(
            [self.dataset_to_utt_ratio for _ in range(self.batch_size)]
        )
        return tensor_weight_batched, 1 - tensor_weight_batched

    def __iter__(self):
        return self

    def step(self, step: int) -> None:
        """
        Record the current step.

        This may change the current stats_to_utt_ratio.
        """
        self._step = step
