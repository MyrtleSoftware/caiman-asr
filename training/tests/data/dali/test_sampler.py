import math

import numpy as np
import pytest

from caiman_asr_train.data.dali.sampler import BucketingSampler, SimpleSampler
from caiman_asr_train.data.dali.utils import _parse_json


def test_simple_sampler(test_data_dir):
    json_files = str(test_data_dir / "peoples-speech-short.json")
    output_files, _ = _parse_json(json_files)
    sampler = SimpleSampler()
    sampler.make_file_list(output_files, json_names=[json_files])
    assert not sampler.is_sampler_random()
    assert sampler.get_dataset_size() == 2


@pytest.mark.parametrize("gbs", [(1), (2), (4)])
def test_process_output_files(gbs):
    num_buckets = 2
    batch_size = 1
    num_workers = 1
    training_steps = 8
    rng = np.random.default_rng(seed=0)
    resume_step = 0
    sampler = BucketingSampler(
        num_buckets, batch_size, num_workers, training_steps, gbs, rng, resume_step
    )
    output_files = {
        "f1": {"duration": 1, "label": "l1"},
        "f2": {"duration": 100, "label": "l2"},
        "f3": {"duration": 2, "label": "l3"},
        "f4": {"duration": 10, "label": "l4"},
        "f5": {"duration": 3, "label": "l4"},
    }
    shuffled_list = sampler.process_output_files(output_files)
    num_epochs = math.ceil(gbs * training_steps / len(output_files))
    assert len(shuffled_list) == len(output_files) * num_epochs
    # split per epoch
    l1 = shuffled_list[: len(output_files)]
    l2 = shuffled_list[len(output_files) : 2 * len(output_files)]

    # assert lists have same elements in each epoch
    assert set(l1) == set(l2)


@pytest.mark.parametrize("training_steps, gbs", [(8, 1), (8, 2), (8, 4)])
def test_bucketing_sampler(test_data_dir, training_steps, gbs):
    num_buckets = 2
    batch_size = 1
    num_workers = 1
    resume_step = 0
    #
    json_files = str(test_data_dir / "test_long_file.json")
    output_files, _ = _parse_json(json_files)
    #
    rng = np.random.default_rng(seed=0)
    sampler = BucketingSampler(
        num_buckets, batch_size, num_workers, training_steps, gbs, rng, resume_step
    )
    returned_list = sampler.process_output_files(output_files)
    assert sampler.is_sampler_random()
    assert len(returned_list) == training_steps * gbs
    #
    sampler = BucketingSampler(
        num_buckets, batch_size, num_workers, training_steps, gbs, rng, resume_step
    )
    new_returned_list = sampler.process_output_files(output_files)
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
    batch_size = 1
    num_workers = 1
    #
    json_files = str(test_data_dir / "test_long_file.json")
    output_files, _ = _parse_json(json_files)
    #
    rng = np.random.default_rng(seed=0)
    sampler = BucketingSampler(
        num_buckets, batch_size, num_workers, training_steps, gbs, rng, resume_step
    )
    returned_list = sampler.process_output_files(output_files)
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
    batch_size = 1
    num_workers = 1
    json_files = str(test_data_dir / "test_long_file.json")
    output_files, _ = _parse_json(json_files)
    #
    rng = np.random.default_rng(seed=0)
    sampler1 = BucketingSampler(
        num_buckets, batch_size, num_workers, training_steps, gbs, rng, resume_step1
    )
    returned_list1 = sampler1.process_output_files(output_files)
    #
    rng = np.random.default_rng(seed=0)
    sampler2 = BucketingSampler(
        num_buckets, batch_size, num_workers, training_steps, gbs, rng, resume_step2
    )
    returned_list2 = sampler2.process_output_files(output_files)
    assert len(returned_list1[resume_step2 * gbs :]) == len(returned_list2)
    assert returned_list1[resume_step2 * gbs :] == returned_list2
