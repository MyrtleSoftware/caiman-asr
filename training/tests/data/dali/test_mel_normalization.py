import nvidia.dali.ops
import nvidia.dali.types
import pytest
import torch
from nvidia.dali.pipeline import Pipeline

from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer, NormType
from caiman_asr_train.data.mel_stats import MelStats


class NormalizePipeline(Pipeline):
    def __init__(
        self,
        stats: MelStats,
        array,
        batch_size=1,
        device_id=nvidia.dali.types.CPU_ONLY_DEVICE_ID,
        num_threads=1,
        seed=42,
    ):
        super(NormalizePipeline, self).__init__(
            device_id=device_id,
            num_threads=num_threads,
            seed=seed,
            batch_size=batch_size,
        )
        self.input = nvidia.dali.ops.ExternalSource(source=lambda: [array])
        self.normalize = MelFeatNormalizer(
            mel_stats=stats,
            ramp_start_step=-1,
            ramp_end_step=-1,
            batch_size=batch_size,
            starting_ratio=1,
            norm_type=NormType.DATASET_STATS,
        )

    def define_graph(self):
        self.data = self.input()
        normalized_data = self.normalize(self.data)
        return normalized_data


@pytest.fixture()
def mel_dim():
    return 8


def normalize_causally(array, stats, mel_dim, time):
    means = stats.means
    stddev = stats.stddevs
    assert means.shape == torch.Size([mel_dim])
    assert stddev.shape == torch.Size([mel_dim])
    output = []
    times = array.shape[1]
    assert times == time

    for t in range(times):
        mels_this_time_step = array[:, t]
        assert mels_this_time_step.shape == torch.Size([mel_dim])
        norm_mels_this_time_step = (mels_this_time_step - means) / stddev
        assert norm_mels_this_time_step.shape == torch.Size([mel_dim])
        output.append(norm_mels_this_time_step)
    stacked_output = torch.stack(output, dim=1)
    assert stacked_output.shape == torch.Size([mel_dim, time])
    return stacked_output


def test_norm_is_causal(mel_dim, time=7):
    """
    Sanity check: is this op streaming compatible.
    """
    means = torch.randn(mel_dim)
    vars = torch.randn(mel_dim) ** 2
    fake_mels = torch.randn(mel_dim, time)
    stats = MelStats(means, vars)
    p = NormalizePipeline(stats, fake_mels)
    p.build()
    test_tensor = torch.tensor(p.run()[0].as_array().squeeze(0))
    ref_tensor = normalize_causally(fake_mels, stats, mel_dim, time)
    # check shapes
    assert test_tensor.shape == ref_tensor.shape
    # check values
    assert torch.allclose(test_tensor, ref_tensor)
