import numpy as np
import pytest

from caiman_asr_train.data.dali.manifest_ratios import AbsoluteManifestRatios
from caiman_asr_train.data.dali.sampler import BucketingSampler, SimpleSampler
from caiman_asr_train.data.dali.utils import _parse_json


def test_simple_sampler(test_data_dir):
    json_files = str(test_data_dir / "peoples-speech-short.json")
    output_files, _ = _parse_json(json_files)
    output_files = [output_files]
    sampler = SimpleSampler(
        total_batches=None, batch_size=1, global_batch_size=1, world_size=1
    )
    sampler.make_file_list(output_files, json_names=[json_files], manifest_ratios=None)
    assert not sampler.is_sampler_random()
    assert sampler.dataset_size == 2


@pytest.mark.parametrize("batch_size", [(1), (2), (4)])
def test_process_output_files(batch_size):
    training_steps = 8

    sampler = BucketingSampler(
        total_batches=training_steps * batch_size,
        batch_size=batch_size,
        global_batch_size=batch_size,
        world_size=1,
        resume_step=0,
        rng=np.random.default_rng(seed=0),
        num_buckets=2,
    )

    output_files = {
        "f1": {"duration": 1.0, "label": 1},
        "f2": {"duration": 100.0, "label": 2},
        "f3": {"duration": 2.0, "label": 3},
        "f4": {"duration": 10.0, "label": 4},
        "f5": {"duration": 3.0, "label": 5},
    }

    s_epochs = sampler._build_epochs(
        [output_files], ["test"], AbsoluteManifestRatios([1.0])
    )

    o_epochs = [sampler._order_epoch(epoch) for epoch in s_epochs]

    for a, b in zip(s_epochs, o_epochs):
        assert set(map(lambda x: x.label, a)) == set(map(lambda x: x.label, b))


@pytest.mark.parametrize("training_steps, gbs", [(8, 1), (8, 2), (8, 4)])
def test_bucketing_sampler(test_data_dir, training_steps, gbs):
    num_buckets = 2
    num_workers = 1
    resume_step = 0
    #
    json_files = str(test_data_dir / "test_long_file.json")
    output_files, _ = _parse_json(json_files)
    #
    rng = np.random.default_rng(seed=0)
    sampler = BucketingSampler(
        total_batches=training_steps,
        batch_size=gbs,
        global_batch_size=gbs * num_workers,
        world_size=num_workers,
        resume_step=resume_step,
        rng=rng,
        num_buckets=num_buckets,
    )
    returned_list, _ = sampler.process_output_files(
        [output_files], ["test"], AbsoluteManifestRatios([1.0])
    )
    assert sampler.is_sampler_random()
    assert len(returned_list) == training_steps * gbs
    #
    sampler = BucketingSampler(
        total_batches=training_steps,
        batch_size=gbs,
        global_batch_size=gbs * num_workers,
        world_size=num_workers,
        resume_step=resume_step,
        rng=rng,
        num_buckets=num_buckets,
    )
    new_returned_list, _ = sampler.process_output_files(
        [output_files], ["test"], AbsoluteManifestRatios([1.0])
    )
    assert new_returned_list != returned_list


@pytest.mark.parametrize(
    "resume_step, exp_length, training_steps, gbs",
    [
        (0, 8, 8, 1),
        (3, 5, 8, 1),
        (3, 13, 10, 1),
        (9, 7, 10, 1),
        (0, 24, 9, 2),
        (2, 12, 8, 2),
        (3, 10, 8, 2),
    ],
)
def test_bucketing_sampler_resume(
    test_data_dir, resume_step, exp_length, training_steps, gbs
):
    num_buckets = 2
    num_workers = 1
    #
    json_files = str(test_data_dir / "test_long_file.json")
    output_files, _ = _parse_json(json_files)
    #
    rng = np.random.default_rng(seed=0)

    sampler = BucketingSampler(
        total_batches=training_steps,
        batch_size=gbs,
        global_batch_size=gbs * num_workers,
        world_size=num_workers,
        resume_step=resume_step,
        rng=rng,
        num_buckets=num_buckets,
    )

    returned_list, _ = sampler.process_output_files(
        [output_files], ["test"], AbsoluteManifestRatios([1.0])
    )

    assert len(returned_list) == exp_length


@pytest.mark.parametrize("training_steps, gbs", [(8, 1), (16, 1), (8, 2), (16, 2)])
@pytest.mark.parametrize(
    "resume_step1, resume_step2",
    [
        (0, 3),
        (0, 5),
    ],
)
def test_resume_files_list(
    test_data_dir,
    training_steps,
    gbs,
    resume_step1,
    resume_step2,
):
    num_buckets = 2

    num_workers = 1
    json_files = str(test_data_dir / "test_long_file.json")
    output_files, _ = _parse_json(json_files)

    #
    rng = np.random.default_rng(seed=0)
    sampler1 = BucketingSampler(
        total_batches=training_steps,
        batch_size=gbs,
        global_batch_size=gbs * num_workers,
        world_size=num_workers,
        resume_step=resume_step1,
        rng=rng,
        num_buckets=num_buckets,
    )
    returned_list1, _ = sampler1.process_output_files(
        [output_files], ["test"], AbsoluteManifestRatios([1.0])
    )
    #

    rng = np.random.default_rng(seed=0)

    sampler2 = BucketingSampler(
        total_batches=training_steps,
        batch_size=gbs,
        global_batch_size=gbs * num_workers,
        world_size=num_workers,
        resume_step=resume_step2,
        rng=rng,
        num_buckets=num_buckets,
    )

    returned_list2, _ = sampler2.process_output_files(
        [output_files], ["test"], AbsoluteManifestRatios([1.0])
    )

    assert len(returned_list1[resume_step2 * gbs :]) == len(returned_list2)
    assert returned_list1[resume_step2 * gbs :] == returned_list2
