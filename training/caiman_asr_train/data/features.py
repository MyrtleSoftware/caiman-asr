# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from beartype import beartype


class BaseFeatures(nn.Module):
    """Base class for audio preprocessing."""

    def __init__(self):
        super(BaseFeatures, self).__init__()

    @torch.no_grad()
    def calculate_features(self, audio, audio_lens, step, training_steps):
        return audio, audio_lens, step, training_steps

    def __call__(self, x):
        audio, audio_lens, step, training_steps = x
        return self.calculate_features(audio, audio_lens, step, training_steps)


class SpecAugment(BaseFeatures):
    """Regularize by masking entire time steps/frequency bands.

    Implements SpecAugment (https://arxiv.org/abs/1904.08779)
    with adaptive masking (https://arxiv.org/abs/1912.05533), without time
    warping.

    Args:
        freq_masks (int): number of masks for frequency bands
        max_freq (int): maximum number of frequencies in a single mask
        time_masks (int or float): number of masks or adaptive percentage
        max_time (int or float): maximum number of masked time steps per mask,
            value 0 < 1 then denotes adaptive percentage
        ratio_spec_aug_on (float): ratio of training steps when SpecAugment is
            applied, 1.0 means always, 0.0 means never
        eoi_frames (int): The number of eoi mel-frames to ignore
    """

    FREQ_AXIS = 1  # Frequency axis in the spectrogram tensor
    TIME_AXIS = 2  # Time axis in the spectrogram tensor

    @beartype
    def __init__(
        self,
        eoi_frames: int,
        freq_masks: int = 0,
        max_freq: int = 10,
        time_masks: int = 0,
        max_time: int | float = 10,
        ratio_spec_aug_on: float = 1.0,
    ):
        super(SpecAugment, self).__init__()

        self.freq_masks = freq_masks
        self.freq_width = max_freq

        self.time_masks = time_masks
        self.time_width = max_time

        self.ratio_spec_aug_on = ratio_spec_aug_on
        assert (
            0.0 <= ratio_spec_aug_on <= 1.0
        ), "ratio_spec_aug_on must be in [0.0, 1.0]"

        self.eoi_frames = eoi_frames

    @beartype
    def spec_augment_off(self, step, training_steps) -> bool:
        """Check if SpecAugment should be applied.

        Args:
        ----
        step: current training step
        training_steps: total number of training steps

        Returns:
        -------
        bool: True if SpecAugment should not be applied based on the current step
        """
        if training_steps <= 0:
            # No training steps defined, SpecAugment is not applied
            return True
        if training_steps > 0 and step / training_steps > self.ratio_spec_aug_on:
            # SpecAugment is not applied, ratio_spec_aug_on is exceeded
            return True
        return False

    @torch.no_grad()
    @beartype
    def calculate_features(self, x, x_lens, step=0, training_steps=0):
        if self.spec_augment_off(step, training_steps):
            # SpecAugment is not applied, spec augment is off
            return x, x_lens, step, training_steps

        B, H, _ = x.shape

        if self.eoi_frames > 0:
            # Cache EOI for later
            eoi = x[0, :, x_lens[0] - self.eoi_frames : x_lens[0]].clone()
            eoi = eoi[None, :, :].expand(B, -1, -1)

        # time masks
        x = self._apply_masks(
            x=x,
            num_masks=self.time_masks,
            x_lens=x_lens,
            width=self.time_width,
            axis=self.TIME_AXIS,
        )
        # freq masks
        x = self._apply_masks(
            x=x,
            num_masks=self.freq_masks,
            x_lens=x_lens,
            width=self.freq_width,
            axis=self.FREQ_AXIS,
        )

        if self.eoi_frames > 0:
            # Repair any modified EOI frames
            at = torch.arange(self.eoi_frames, device=x.device)
            at = at[None, None, :].expand(B, H, -1)
            at = at + x_lens[:, None, None] - self.eoi_frames
            x.scatter_(dim=2, index=at, src=eoi)

        return x, x_lens, step, training_steps

    @beartype
    def _apply_masks(
        self,
        x: torch.Tensor,
        num_masks: int,
        x_lens: torch.Tensor,
        width: int | float,
        axis: int,
    ) -> torch.Tensor:
        """
        This function applies SpecAugment masks to the input tensor `x` along
        the specified axis. The masks are independently generated for each
        batch item and applied in parallel.

        For full details see original implementation:

        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/spectr_augment.py

        Args:
            x: (Batch, Freq, Time) tensor of spectrograms
            x_lens: (Batch,) tensor of lengths of the spectrograms
            num_masks: Number of masks to apply.
            axis: Axis along which to apply the masks, either FREQ_AXIS or TIME_AXIS.
            width: Maximum width of a mask either a number/fraction of time-steps if
                axis is TIME_AXIS, or number of frequency bands if axis is FREQ_AXIS.
        """

        assert axis in (
            self.FREQ_AXIS,
            self.TIME_AXIS,
        ), f"Axis can be ({self.FREQ_AXIS}) or ({self.TIME_AXIS}). Received: {axis=}"

        assert not (
            isinstance(width, float) and axis == self.FREQ_AXIS
        ), "Float width supported only with time axis."

        batch_size = x.shape[0]
        axis_length = x.shape[axis]

        # If width is float then it is transformed into a tensor
        if axis == self.TIME_AXIS and isinstance(width, float):
            width = torch.clamp(width * x_lens, max=axis_length).unsqueeze(1)

        # Generate [0-1) random numbers and then scale the tensors.
        # Use float32 dtype for begin/end mask markers before they are quantized to long.
        mask_width = (
            torch.rand(
                (batch_size, num_masks),
                device=x.device,
                dtype=torch.float32,
            )
            * width
        )
        mask_width = mask_width.long()
        mask_start = torch.rand(
            (batch_size, num_masks), device=x.device, dtype=torch.float32
        )

        if axis == self.TIME_AXIS:
            # length can only be used for the time axis
            mask_start = mask_start * (x_lens.unsqueeze(1) - mask_width)
        else:
            mask_start = mask_start * (axis_length - mask_width)

        mask_start = mask_start.long()
        mask_end = mask_start + mask_width

        # Create mask values using vectorized indexing
        indices = torch.arange(axis_length, device=x.device)
        # Create a mask_tensor with all the indices.
        # The mask_tensor shape is (batch_size, num_masks, axis_length).
        mask_tensor = (indices >= mask_start.unsqueeze(-1)) & (
            indices < mask_end.unsqueeze(-1)
        )

        # Reduce masks to one mask
        mask_tensor = mask_tensor.any(dim=1)

        # Create a final mask that aligns with the full tensor
        mask = torch.zeros_like(x, dtype=torch.bool)
        if axis == self.TIME_AXIS:
            mask_ranges = mask_tensor[:, None, :]
        else:  # axis == self.FREQ_AXIS
            mask_ranges = mask_tensor[:, :, None]
        mask[:, :, :] = mask_ranges

        # Apply the mask value
        return x.masked_fill(mask=mask, value=0.0)


def stack_subsample_frames(x, x_lens, stacking=1, subsampling=1):
    """Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    seq = [x]
    for n in range(1, stacking):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    x = torch.cat(seq, dim=1)[:, :, ::subsampling]

    if subsampling > 1:
        x_lens = torch.ceil(x_lens.float() / subsampling).int()

        if x.size(2) > x_lens.max().item():
            assert abs(x.size(2) - x_lens.max().item()) <= 1
            x = x[:, :, : x_lens.max().item()]

    return x, x_lens


class FrameSplicing(BaseFeatures):
    __constants__ = ["frame_subsampling", "frame_stacking"]

    def __init__(self, frame_stacking=1, frame_subsampling=1):
        super(FrameSplicing, self).__init__()
        self.frame_stacking = frame_stacking
        self.frame_subsampling = frame_subsampling

    def calculate_features(self, x, x_lens, step=0, training_steps=0):
        # frame splicing if required
        if self.frame_stacking > 1 or self.frame_subsampling > 1:
            x, x_lens = stack_subsample_frames(
                x, x_lens, self.frame_stacking, self.frame_subsampling
            )

        return x, x_lens, step, training_steps


class PermuteAudio(nn.Module):
    def forward(self, x):
        return (x[0].permute(2, 0, 1), *x[1:])
