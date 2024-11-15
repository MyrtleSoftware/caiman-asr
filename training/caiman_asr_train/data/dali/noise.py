import numpy as np
from beartype.typing import List
from numba import jit


class NoiseAugmentationIterator:
    def __init__(
        self,
        batch_size: int,
        prob_noise,
    ):
        self.low = 30
        self.high = 60
        self.no_noise = 200  # i.e. the SNR is so high that there is 'no noise'
        self.batch_size = batch_size
        self.prob_noise = prob_noise

    def set_range(self, low: int, high: int):
        self.low = low
        self.high = high

    def get_range(self):
        return self.low, self.high

    def __iter__(self):
        return self

    def __next__(self):
        target_snrs = self._target_snrs(self.prob_noise)
        start_ratios = self._start_ratios()
        return (target_snrs, start_ratios)

    def _target_snrs(self, probability_apply: float) -> List[np.array]:
        """
        Generate array of target SNRs, used to determine the volume of noise to add.
        """
        return [
            (
                np.array([np.random.uniform(self.low, self.high)], dtype=np.float32)
                if np.random.uniform() < probability_apply
                else np.array([self.no_noise], dtype=np.float32)
            )
            for _ in range(self.batch_size)
        ]

    def _start_ratios(self) -> List[np.array]:
        """
        Generate array of start_ratios; used to determine the start point of the noise.
        """
        return [
            np.array([np.random.uniform(0, 1)], dtype=np.float32)
            for _ in range(self.batch_size)
        ]


class NoiseSchedule(object):
    """
    This class provides a noise augmentation signal-to-noise ratio (SNR)
    schedule permitting a delay of delay_steps before SNRs begin to
    decrease from their (usually high) initial range, after which they
    ramp down over ramp_steps to their final range. The final range varies
    depending on the type of noise augmentation:
        - background noise: SNR range of 0-30dB. These are chosen following the Google
            paper "Streaming end-to-end speech recognition for mobile devices",
            He, et al., 2018.
        - babble noise: SNR range of 15-30dB. These are chosen on the basis that
            babble must be significantly quieter than the speech for the ASR system to
            have _any_ chance of transcribing the correct signal.
    """

    def __init__(
        self,
        delay_steps: int,
        ramp_steps: int,
        initial_low: int,
        initial_high: int,
        train_loader,
    ):
        self.delay_steps = delay_steps
        self.ramp_steps = ramp_steps
        self.train_loader = train_loader
        self.initial_low = initial_low
        self.initial_high = initial_high

    def background_noise_set_range(self, low: int, high: int):
        if self.train_loader.pipeline.do_background_noise_aug:
            self.train_loader.pipeline.background_noise_iterator.set_range(low, high)

    def babble_noise_set_range(self, low: int, high: int):
        if self.train_loader.pipeline.do_babble_noise_aug:
            self.train_loader.pipeline.babble_noise_iterator.set_range(low, high)

    def get_noise(self):
        if self.train_loader.pipeline.do_background_noise_aug:
            bg = self.train_loader.pipeline.background_noise_iterator.get_range()
        else:
            bg = (-1, -1)

        if self.train_loader.pipeline.do_babble_noise_aug:
            bb = self.train_loader.pipeline.babble_noise_iterator.get_range()
        else:
            bb = (-1, -1)

        return bg, bb

    def adjust_snrs(self, step):
        # Set low and high to their initial settings until step passes delay_steps.
        if step <= self.delay_steps:
            self.background_noise_set_range(self.initial_low, self.initial_high)
            self.babble_noise_set_range(self.initial_low, self.initial_high)
            return self.get_noise()
        if step >= self.delay_steps + self.ramp_steps:
            # The final values for background noise are chosen to be 0dB / 30dB
            self.background_noise_set_range(0, 30)
            # final values for babble noise are chosen to be 15dB / 30dB on basis that
            # babble noise must be significantly quieter than the speech signal for ASR
            # system to have any chance of transcribing the correct signal.
            self.babble_noise_set_range(15, 30)
            return self.get_noise()
        # low and high are both linearly interpolated over the ramp from their initial
        # to final values.
        highdelta = int(
            (step - self.delay_steps) * (self.initial_high - 30.0) / self.ramp_steps
        )
        lowdelta_background = int(
            (step - self.delay_steps) * (self.initial_low - 0.0) / self.ramp_steps
        )
        lowdelta_babble = int(
            (step - self.delay_steps) * (self.initial_low - 15.0) / self.ramp_steps
        )
        self.background_noise_set_range(
            self.initial_low - lowdelta_background, self.initial_high - highdelta
        )
        self.babble_noise_set_range(
            self.initial_low - lowdelta_babble, self.initial_high - highdelta
        )
        return self.get_noise()


@jit(nopython=True)
def blend(
    speech: np.ndarray, noise: np.ndarray, target_snr: float, start_ratio: np.ndarray
) -> np.ndarray:
    """
    This function blends together the numpy arrays speech and noise
    to have a signal-to-noise-ratio (SNR) of target_srn.  The
    result is scaled to have the same power as the original input
    speech.  Note that noise files must be at least as long as
    speech files; use max_duration in config.

    Parameters
    ----------
    speech : np.ndarray
        Speech signal to be blended with noise.
    noise : np.ndarray
        Noise signal to be blended with speech.
    target_snr : float
        Target SNR for the blended signal.
    start_ratio : np.ndarray
        Length ratio [0, 1] that determines start point of noise clip. This must be a
        numpy array of shape (1,) to run with numba.
    """
    diff_frames = noise.shape[0] - speech.shape[0]
    if diff_frames < 0:
        # then noise is shorter than speech, use all of the noise clip
        diff_frames = 0

    start_frame = np.floor(diff_frames * start_ratio).astype(np.int64)[0]
    cropped_noise = noise[start_frame : start_frame + speech.shape[0]]
    # Power of a speech signal is given by its mean-square value
    speech_MS = (speech * speech).mean()
    speech_dB = 10.0 * np.log10(speech_MS)
    noise_dB = 10.0 * np.log10((cropped_noise * cropped_noise).mean())
    actual_snr = speech_dB - noise_dB
    inc_snr = target_snr - actual_snr
    inc_mag = np.power(10.0, inc_snr / 20.0)
    adj_noise = cropped_noise / inc_mag
    # At this point speech and adj_noise have the target SNR
    merged = speech + adj_noise
    # Scale merged to have the same power as the original speech signal
    merged_MS = (merged * merged).mean()
    result = merged * np.sqrt(speech_MS / merged_MS)
    return result


@jit(nopython=True)
def duplicate_then_blend(
    speech: np.ndarray, noise: np.ndarray, target_snr: float, start_ratio: np.ndarray
) -> np.ndarray:
    """Wrapper around blend that handles the case where noise is shorter than speech."""
    longer_noise = duplicate_audio(noise, speech.shape[0])
    return blend(speech, longer_noise, target_snr, start_ratio)


def blend_dali_api(
    out_speech: np.ndarray,
    out_noise: np.ndarray,
    out_target_snr: float,
    out_start_ratio: float,
    in_speech: np.ndarray,
    in_noise: np.ndarray,
    in_target_snr: float,
    in_start_ratio: float,
):
    """
    Blend function following the dali numba api.

    This requires:
        1) output args are passed as well as input args
        2) there are the same number (and type) of output args as input args
        3) results are copied into the out tensors (otherwise they will all be zeros)
    """
    out_speech[:] = duplicate_then_blend(
        in_speech, in_noise, in_target_snr, in_start_ratio
    )
    return out_speech, out_noise, out_target_snr, out_start_ratio


@jit(nopython=True)
def babble_batch(
    audio: List[np.ndarray],
    target_snrs: List[np.ndarray],
    start_ratios: List[np.ndarray],
) -> List[np.ndarray]:
    """
    This function adds babble noise to a batch of audio signals.
    """
    batch_size = len(audio)
    if batch_size == 1:
        raise ValueError("babble_batch requires a batch size of at least 2")

    output = [np.zeros_like(x, dtype=np.float32) for x in audio]
    for i in range(batch_size):
        # select babble sample. Take next sample in batch, or first if at the end
        babble = audio[(i + 1) % batch_size]
        # babble may be shorter than audio, so duplicate it
        babble = duplicate_audio(babble, audio[i].shape[0])
        # blend audio with babble
        output[i] = blend(audio[i], babble, target_snrs[i], start_ratios[i]).astype(
            np.float32
        )
    return output


@jit(nopython=True)
def duplicate_audio(
    input: np.ndarray,
    target_length: int,
    overlap_frames: int = 3200,
    min_length_growth_ratio: float = 0.6,
) -> np.ndarray:
    """
    Return audio that is longer than target_length by iteratively duplicating the input.

    In the overlap region a cosine window is applied to the input and output audio.

    Parameters
    ----------
    input
        The input audio to be duplicated.
    target_length
        The min length of the output audio.
    overlap_frames
        The number of frames to overlap when duplicating the input. This will not be
        respected in the case where the input is shorter than the overlap_frames.
    min_length_growth_ratio
        Where the input is shorter than overlap_frames, this is the minimum ratio of
        the current length the sample should grow by on each iteration.
    """
    cos_values_in = np.cos(np.linspace(0, np.pi / 2, overlap_frames))
    cos_values_out = cos_values_in[::-1]
    input = input.astype(np.float32)
    while input.shape[0] <= target_length:
        in_frame_idx = input.shape[0] - overlap_frames
        if in_frame_idx < min_length_growth_ratio * input.shape[0]:
            in_frame_idx = int(min_length_growth_ratio * input.shape[0])
            length_overlap = input.shape[0] - in_frame_idx
        else:
            length_overlap = overlap_frames

        # apply cosine window to the overlap
        first, second = input.copy(), input.copy()
        overlap = (
            first[in_frame_idx:] * cos_values_in[:length_overlap]
            + second[:length_overlap] * cos_values_out[:length_overlap]
        )
        input = np.concatenate((first[:in_frame_idx], overlap, second[length_overlap:]))

        # numba needs to know the type of the input explicitly so as not to fail
        # compilation on the second iteration
        input = input.astype(np.float32)
    return input


def babble_batch_dali_api(
    out_audio: List[np.ndarray],
    out_target_snrs: List[np.ndarray],
    out_start_ratios: List[np.ndarray],
    in_audio: List[np.ndarray],
    in_target_snrs: List[np.ndarray],
    in_start_ratios: List[np.ndarray],
):
    """
    Babble function following the dali numba api.

    See blend_dali_api docstring for details.
    """
    result = babble_batch(in_audio, in_target_snrs, in_start_ratios)
    for i in range(len(result)):
        out_audio[i][:] = result[i]
    return out_audio, out_target_snrs, out_start_ratios
