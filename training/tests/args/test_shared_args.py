#! /usr/bin/env python3
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

from caiman_asr_train.args.shared import (
    check_directories_are_valid,
    validation_directories_provided,
)


@pytest.mark.parametrize("temp_dir1, temp_dir2", [("/tmp", "/tmp/")])
def test_validation_directories_provided(temp_dir1, temp_dir2):
    assert validation_directories_provided(temp_dir1, temp_dir2)


def test_check_directories_are_valid():
    with TemporaryDirectory() as temp_dir:
        dataset_dir = temp_dir
        f_txt = NamedTemporaryFile(suffix=".txt", dir=dataset_dir, delete=False)
        f_wav = open(f_txt.name.replace(".txt", ".wav"), "w")
        check_directories_are_valid(dataset_dir, dataset_dir, dataset_dir)
        f_wav.close()
