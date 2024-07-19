import tarfile

import numpy as np
import pytest

from caiman_asr_train.data.webdataset import WebDatasetReader
from caiman_asr_train.setup.text_normalization import NormalizeConfig, NormalizeLevel


@pytest.fixture()
def webdataset_reader(test_data_dir, tokenizer) -> WebDatasetReader:
    return WebDatasetReader(
        file_root=str(test_data_dir),
        tar_files=["webdataset-eg.tar"],
        batch_size=2,
        shuffle=True,
        tokenizer=tokenizer,
        normalize_config=NormalizeConfig(NormalizeLevel.LOWERCASE, [], True),
        num_buckets=2,
    )


@pytest.fixture()
def webdataset_reader_periods(test_data_dir, tokenizer) -> WebDatasetReader:
    return WebDatasetReader(
        file_root=str(test_data_dir),
        tar_files=["webdataset-eg-with-periods.tar"],
        batch_size=2,
        shuffle=True,
        tokenizer=tokenizer,
        normalize_config=NormalizeConfig(NormalizeLevel.LOWERCASE, [], True),
        num_buckets=2,
    )


def test_webdataset_returns_samples(webdataset_reader):
    seen_samples = 0
    for audio, transcript, raw_transcript in webdataset_reader:
        assert isinstance(transcript, np.ndarray)
        assert transcript.dtype == np.int32
        assert isinstance(raw_transcript, np.ndarray)
        assert raw_transcript.dtype == np.int32
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        seen_samples += 1

    assert (
        seen_samples == 2
    ), f"There should be exactly 2 samples in the test data, but there were {seen_samples}"


def test_webdataset_periods_returns_samples(webdataset_reader_periods):
    seen_samples = 0
    for audio, transcript, raw_transcript in webdataset_reader_periods:
        assert isinstance(transcript, np.ndarray)
        assert transcript.dtype == np.int32
        assert isinstance(raw_transcript, np.ndarray)
        assert raw_transcript.dtype == np.int32
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        seen_samples += 1

    assert (
        seen_samples == 2
    ), f"There should be exactly 2 samples in the test data, but there were {seen_samples}"


def test_webdataset_with_periods(test_data_dir):
    tar_file_path = str(test_data_dir / "webdataset-eg-with-periods.tar")
    tar_f = tarfile.open(tar_file_path, "r")
    for f in tar_f.getmembers():
        fname = f.name
        assert (
            fname.count(".") > 1
        ), f"There should be multiple periods in the filename, but is only one in {fname}"
