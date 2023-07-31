import numpy as np
import pytest

from rnnt_train.common.data.webdataset import WebDatasetReader


@pytest.fixture()
def webdatset_reader(test_data_dir, tokenizer) -> WebDatasetReader:
    # pass in a list of tar files,
    # could also use file_root=None, and tar_files=str(webdataset_fp) but instead:
    return WebDatasetReader(
        file_root=str(test_data_dir),
        tar_files=["*.tar"],
        batch_size=2,
        shuffle=True,
        tokenizer=tokenizer,
        normalize_transcripts=True,
        num_buckets=2,
    )


def test_webdataset_returns_samples(webdatset_reader):
    seen_samples = 0
    for audio, transcript in webdatset_reader:
        assert isinstance(transcript, np.ndarray)
        assert transcript.dtype == np.int32
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        seen_samples += 1

    assert (
        seen_samples == 2
    ), f"There should be exactly 2 samples in the test data, but there were {seen_samples}"
