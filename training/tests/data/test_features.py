import pytest
import torch
from beartype import beartype

from caiman_asr_train.lhotse.spec_augment import SpecAugment


@pytest.fixture
def dummy_input():
    # [Batch, Freq, Time]
    audio = torch.ones((2, 40, 100))
    audio_lens = torch.tensor([100, 80])
    return audio, audio_lens


@beartype
def test_shape_preservation(dummy_input: tuple[torch.Tensor, torch.Tensor]):
    aug = SpecAugment(freq_masks=2, time_masks=2)
    audio, audio_lens = dummy_input
    out = aug(audio.clone(), audio_lens.clone())
    assert out.shape == audio.shape
    assert not torch.equal(
        out, audio
    ), "Output should not be equal to input after augmentation"


@beartype
def test_high_spec_aug_ratio(dummy_input: tuple[torch.Tensor, torch.Tensor]):
    aug = SpecAugment(freq_masks=2, time_masks=2)
    audio, audio_lens = dummy_input
    out = aug(audio.clone(), audio_lens.clone())
    assert not torch.equal(out, audio), "Equal"


@beartype
def test_low_spec_aug_ratio(dummy_input: tuple[torch.Tensor, torch.Tensor]):
    aug = SpecAugment(freq_masks=0, time_masks=0)
    audio, audio_lens = dummy_input
    out = aug(audio.clone(), audio_lens.clone())
    assert torch.equal(out, audio), "Not equal"
