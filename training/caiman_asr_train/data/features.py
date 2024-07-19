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


class BaseFeatures(nn.Module):
    """Base class for audio preprocessing."""

    def __init__(self):
        super(BaseFeatures, self).__init__()

    @torch.no_grad()
    def calculate_features(self, audio, audio_lens):
        return audio, audio_lens

    def __call__(self, x):
        audio, audio_lens = x
        return self.calculate_features(audio, audio_lens)


class SpecAugment(BaseFeatures):
    """Regularize by masking entire time steps/frequency bands.

    Implements SpecAugment (https://arxiv.org/abs/1904.08779)
    with adaptive masking (https://arxiv.org/abs/1912.05533), without time
    warping.

    Args:
        freq_masks (int): number of masks for frequency bands
        min_freq (int): minimum number of frequencies in a single mask
        max_freq (int or float): maximum number of frequencies in a single mask
        time_masks (int or float): number of masks or adaptive percentage
        min_time (int): minimum number of masked time steps per mask; applies
            only if max is non-adaptive
        max_time (int or float): maximum number of masked time steps per mask,
            value 0 < 1 then denotes adaptive percentage
        noise_magnitude (float): mask with N(0, noise_magnitude * std(sample))
            noise instead of zeros to stabilize training
    """

    def __init__(
        self,
        freq_masks=0,
        min_freq=0,
        max_freq=10,
        time_masks=0,
        min_time=0,
        max_time=10,
        noise_magnitude=0,
    ):
        super(SpecAugment, self).__init__()
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.freq_masks = freq_masks
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.time_masks = time_masks
        self.min_time = min_time
        self.max_time = max_time

        self.noise_magnitude = noise_magnitude

    @torch.no_grad()
    def calculate_features(self, x, x_lens):
        sh = x.shape
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

        for idx in range(sh[0]):
            for _ in range(self.freq_masks):
                w = torch.randint(self.min_freq, self.max_freq + 1, size=(1,)).item()
                f0 = torch.randint(0, max(1, sh[1] - w + 1), size=(1,))
                mask[idx, f0 : f0 + w] = 1

            # Adaptive time masking
            time_masks = self.time_masks
            if 0 < time_masks < 1.0:
                time_masks = int(round(x_lens[idx].item() * time_masks))

            max_time = self.max_time
            if 0 < max_time < 1.0:
                max_time = int(round(x_lens[idx].item() * max_time))

            for _ in range(time_masks):
                w = torch.randint(self.min_time, max_time + 1, size=(1,)).item()
                t0 = torch.randint(0, max(1, sh[2] - w + 1), size=(1,))
                mask[idx, :, t0 : t0 + w] = 1

        if self.noise_magnitude > 0:
            mean = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            std = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
            for idx in range(sh[0]):
                mean[idx, :, 0] = x[idx, :, : x_lens[idx]].mean(dim=1)
                std[idx, :, 0] = x[idx, :, : x_lens[idx]].std(dim=1)

            std *= self.noise_magnitude
            noise = (mean + torch.randn_like(x) * std).masked_fill(~mask, 0)
        else:
            noise = 0

        return x.masked_fill(mask, 0) + noise, x_lens


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

    def calculate_features(self, x, x_lens):
        # frame splicing if required
        if self.frame_stacking > 1 or self.frame_subsampling > 1:
            x, x_lens = stack_subsample_frames(
                x, x_lens, self.frame_stacking, self.frame_subsampling
            )

        return x, x_lens


class PermuteAudio(nn.Module):
    def forward(self, x):
        return (x[0].permute(2, 0, 1), *x[1:])
