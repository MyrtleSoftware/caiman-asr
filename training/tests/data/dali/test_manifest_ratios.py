#!/usr/bin/env python3

import copy

import numpy as np
import pytest
from beartype import beartype

from caiman_asr_train.data.dali import sampler
from caiman_asr_train.unittesting.data_loader import build_dataloader_util


@beartype
def count_dummy_files(fname_list: list[str]):
    """
    test_long_file.json contains 8 files,
    all of which have 'dummy' in the name
    """
    return sum("dummy" in fname for fname in fname_list)


@beartype
def count_ps_files(fname_list: list[str]):
    """peoples-speech-short.json contains two files,
    neither of which have 'dummy' in the name"""
    return len(fname_list) - count_dummy_files(fname_list)


@pytest.mark.parametrize(
    "pipeline_type,relative_ratios,expected_ps_ratio",
    [
        # Adjusting the relative_ratios arg should change
        # the ratio of People's Speech to dummy files
        # one epoch of the dataloader
        ("train", None, 2 / 8),
        ("train", [1.0, 1.0], 2 / 8),
        ("train", [2.0, 1.0], 4 / 8),
        ("train", [1.0, 2.0], 2 / 16),
        # But validation should be unaffected
        ("val", None, 2 / 8),
        ("val", [1.0, 1.0], 2 / 8),
        ("val", [2.0, 1.0], 2 / 8),
        ("val", [1.0, 2.0], 2 / 8),
    ],
)
def test_manifest_balancing_ratios(
    pipeline_type,
    relative_ratios,
    expected_ps_ratio,
    dataload_args,
    mini_config_fp,
    tokenizer,
    test_data_dir,
):
    training_steps = 10
    batch_size = 1
    global_batch_size = 1

    args = copy.deepcopy(dataload_args)
    args.training_steps = training_steps
    args.train_manifests = [
        str(test_data_dir / "peoples-speech-short.json"),
        str(test_data_dir / "test_long_file.json"),
    ]
    args.val_manifests = copy.deepcopy(args.train_manifests)
    args.relative_train_manifest_ratios = relative_ratios

    sampler_ = None
    if pipeline_type == "train":
        sampler_ = sampler.BucketingSampler(
            num_buckets=dataload_args.num_buckets,
            batch_size=batch_size,
            global_batch_size=global_batch_size,
            world_size=1,
            total_batches=training_steps * global_batch_size // batch_size,
            rng=np.random.default_rng(seed=dataload_args.seed),
            resume_step=0,
        )

    dataloader = build_dataloader_util(
        args,
        pipeline_type,
        batch_size,
        mini_config_fp,
        tokenizer,
        train_sampler=sampler_,
    )

    all_fnames = []
    for batch in dataloader:
        _, _, _, _, _, fnames = batch
        all_fnames.extend(fnames)
    assert (
        count_ps_files(all_fnames) / count_dummy_files(all_fnames) == expected_ps_ratio
    )
