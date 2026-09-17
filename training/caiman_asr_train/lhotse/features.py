import torch
import torchaudio
from beartype import beartype
from beartype.typing import Optional
from lhotse import Seconds
from torch import load
from torch.nn import Parameter

from caiman_asr_train.lhotse.config import BlendedNorm, DataConfig, GlobalNorm, UttNorm


@beartype
class MelSpectrogram(torch.nn.Module):
    def __init__(self, config: DataConfig):
        """
        Build spectrograms from input waveforms:

        Performs: Dithering -> Pre-emphasis -> MelSpectrogram -> to Decibels

        This all runs in pure PyTorch land.
        """
        super().__init__()
        self.config = config
        self.min_ratio = pow(10, config.cutoff_db / config.multiplier)
        self.pre_emph = torchaudio.transforms.Preemphasis(coeff=config.preemph_coeff)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_sample_length,
            hop_length=config.hop_sample_length,
            n_mels=config.n_mels,
            # To match DALI
            center=False,
            f_max=config.sample_rate // 2,  # Nyquist
            mel_scale="slaney",
            norm="slaney",
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps audio (B, T) to mel spectrograms (B, n_mels, T')
        """
        # Callchain, torch :: spectrogram -> stft, for output shape see:
        # https://docs.pytorch.org/docs/stable/generated/torch.stft.html
        x_lens = 1 + (x_lens - self.config.n_fft) // self.config.hop_sample_length

        if self.config.dither_coeff != 0.0:
            x = x + torch.randn_like(x) * self.config.dither_coeff

        x = self.pre_emph(x)
        x = self.mel_spec(x)

        # Convert to decibels using dali formula:
        #
        # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.to_decibels.html
        #
        # Technically this is not decibels. The formula for dB conversion is:
        #
        #   dB = 10 * log_10(input/reference)
        #
        # but this implements:
        #
        #   dB = log(10) * log_10(input/reference)
        #
        # To remain consistent with the old DALI version.
        x = self.config.multiplier * torch.log10(
            torch.clamp(x / self.config.reference, min=self.min_ratio)
        )

        # Sanity
        assert torch.max(x_lens).item() == x.shape[2]

        return x, x_lens


@beartype
class MelNormalizer(torch.nn.Module):
    def __init__(self, config: DataConfig):
        """
        Performs normalization of mel features, this can be local normalization
        (non-streaming, uses the entire utterance) or global normalization
        (using pre-computed statistics) or a linear interpolation of the two.

        This is a pure PyTorch implementation.
        """
        super().__init__()
        self.mel_conf = config.mel_norm

        if isinstance(self.mel_conf, GlobalNorm | BlendedNorm):

            def param(x, pow):
                return Parameter(
                    load(x, weights_only=False) ** pow, requires_grad=False
                )

            self.means = param(self.mel_conf.stats / "melmeans.pt", 1)
            self.stdvs = param(self.mel_conf.stats / "melvars.pt", 0.5)

            assert self.means.shape == (config.n_mels,)
            assert self.stdvs.shape == (config.n_mels,)

    def forward(
        self,
        feats: torch.Tensor,
        feat_lens: torch.Tensor,
        total_dur: Optional[Seconds] = None,
    ) -> tuple[torch.Tensor, Optional[float]]:
        """
        Functionally:
            feats : (B, Mel, T) -> (B, Mel, T), alpha

        Where alpha is the global blend fraction.
        """
        match total_dur, self.mel_conf:
            case None, BlendedNorm():
                raise ValueError("BlendedNorm requires total_dur")
            case _, None:
                return (feats, None)
            case _, GlobalNorm():
                return self._global_norm(feats), 1.0
            case _, UttNorm():
                return self._utt_norm(feats, feat_lens), 0.0
            case tot, BlendedNorm(ramp_beg=beg) if tot < beg:
                return self._utt_norm(feats, feat_lens), 0.0
            case tot, BlendedNorm(ramp_end=end) if tot > end:
                return self._global_norm(feats), 1.0
            case tot, BlendedNorm(ramp_beg=beg, ramp_end=end):
                lcl = self._utt_norm(feats, feat_lens)
                gbl = self._global_norm(feats)

                alpha = (tot - beg) / (end - beg)
                assert 0.0 <= alpha <= 1.0, f"{alpha=}, {tot=}, {beg=}, {end=}"
                return (1 - alpha) * lcl + alpha * gbl, alpha

    def _utt_stats(
        self, feats: torch.Tensor, feat_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean/stddev for each mel bin over the time dimension.
        """
        B, _, T = feats.shape

        padding = torch.arange(T, device=feat_lens.device)
        padding = padding[None, :].expand(B, T)
        padding = padding > feat_lens[:, None].expand(B, T)
        padding = padding[:, None, :]

        feats = feats.masked_fill_(padding, 0.0)
        feat_lens = feat_lens[:, None, None]

        means = torch.sum(feats, dim=2, keepdim=True) / feat_lens

        # Biased variance
        vars = (feats - means).masked_fill_(padding, 0.0)
        vars = torch.sum(vars * vars, dim=2, keepdim=True) / (feat_lens - 1)

        stdvs = vars**0.5

        return means, stdvs

    def _norm(self, feats: torch.Tensor, means, stdvs) -> torch.Tensor:
        return (feats - means) / stdvs

    def _utt_norm(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> torch.Tensor:
        return self._norm(feats, *self._utt_stats(feats, feat_lens))

    def _global_norm(self, feats: torch.Tensor) -> torch.Tensor:
        return self._norm(feats, self.means[:, None], self.stdvs[:, None])
