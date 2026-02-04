import pytest
import torch
from beartype import beartype

from caiman_asr_train.data.features import SpecAugment


@pytest.fixture
def dummy_input():
    # [Batch, Freq, Time]
    audio = torch.ones((2, 40, 100))
    audio_lens = torch.tensor([100, 80])
    return audio, audio_lens


@beartype
def test_shape_preservation(dummy_input: tuple[torch.Tensor, torch.Tensor]):
    aug = SpecAugment(eoi_frames=0, freq_masks=2, time_masks=2)
    audio, audio_lens = dummy_input
    out, out_lens, _, _ = aug([audio.clone(), audio_lens.clone(), 1, 1])
    assert out.shape == audio.shape
    assert torch.equal(out_lens, audio_lens)
    assert not torch.equal(
        out, audio
    ), "Output should not be equal to input after augmentation"


@beartype
@pytest.mark.parametrize("ratio_spec_aug_on", [0.0, 1.0])
def test_zero_training_steps(
    dummy_input: tuple[torch.Tensor, torch.Tensor], ratio_spec_aug_on: float
):
    aug = SpecAugment(
        eoi_frames=0,
        time_masks=1,
        max_time=10,
        ratio_spec_aug_on=ratio_spec_aug_on,
    )
    audio, audio_lens = dummy_input
    out, _, _, _ = aug([audio.clone(), audio_lens.clone(), 1, 0])
    assert torch.equal(out, audio), "No spec augment due to zero training steps"


@beartype
@pytest.mark.parametrize("ratio_spec_aug_on", [0.60, 0.80, 1.0])
def test_high_spec_aug_ratio(
    dummy_input: tuple[torch.Tensor, torch.Tensor], ratio_spec_aug_on: float
):
    eoi = 4
    aug = SpecAugment(
        eoi_frames=eoi, freq_masks=2, time_masks=2, ratio_spec_aug_on=ratio_spec_aug_on
    )
    audio, audio_lens = dummy_input
    out, _, _, _ = aug([audio.clone(), audio_lens.clone(), 30, 50])

    # Check EOI are not augmented
    for i in range(audio.shape[0]):
        assert torch.all(out[i, :, audio_lens[i] - eoi : audio_lens[i]] == 1.0)

    assert not torch.equal(out, audio), "Not equal"


@beartype
@pytest.mark.parametrize("ratio_spec_aug_on", [0.0, 0.59])
def test_low_spec_aug_ratio(
    dummy_input: tuple[torch.Tensor, torch.Tensor], ratio_spec_aug_on: float
):
    aug = SpecAugment(
        eoi_frames=0, freq_masks=0, time_masks=0, ratio_spec_aug_on=ratio_spec_aug_on
    )
    audio, audio_lens = dummy_input
    out, _, _, _ = aug([audio.clone(), audio_lens.clone(), 30, 50])
    assert torch.equal(out, audio), "Not equal"
