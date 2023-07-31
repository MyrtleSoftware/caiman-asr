from pathlib import Path

import pytest

from rnnt_train.common.data.text import Tokenizer
from rnnt_train.rnnt import config


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    # use path relative to this file to find the testing data directory
    test_data_dir = Path(__file__).parent / "test_data"
    assert test_data_dir.is_dir(), f"{test_data_dir=} doesn't exist"
    return test_data_dir


@pytest.fixture(scope="session")
def webdataset_fp(test_data_dir) -> Path:
    return test_data_dir / "webdataset-eg.tar"


@pytest.fixture(scope="session")
def config_fp() -> str:
    # we test using the base config
    return "./configs/testing-1023sp_run.yaml"


@pytest.fixture()
def tokenizer(config_fp, test_data_dir) -> Tokenizer:
    cfg = config.load(config_fp)
    tokenizer_kw = config.tokenizer(cfg)
    # replace the spm with the one in the test data dir
    tokenizer_kw["sentpiece_model"] = str(test_data_dir / "librispeech29.model")
    return Tokenizer(**tokenizer_kw)
