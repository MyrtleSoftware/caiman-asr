from pathlib import Path
from typing import Callable, Optional

import pytest
import torch

from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.helpers import Checkpointer
from rnnt_train.common.seed import set_seed
from rnnt_train.rnnt import config
from rnnt_train.rnnt.model import RNNT


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    # use path relative to this file to find the testing data directory
    test_data_dir = Path(__file__).parent / "test_data"
    assert test_data_dir.is_dir(), f"{test_data_dir=} doesn't exist"
    return test_data_dir


@pytest.fixture(scope="session")
def config_fp() -> str:
    # we test using the base config
    return "./configs/testing-1023sp_run.yaml"


@pytest.fixture()
def tokenizer_kw(config_fp, test_data_dir):
    cfg = config.load(config_fp)
    tokenizer_kw = config.tokenizer(cfg)
    # replace the spm with the one in the test data dir
    tokenizer_kw["sentpiece_model"] = str(test_data_dir / "librispeech29.model")
    return tokenizer_kw


@pytest.fixture()
def tokenizer(tokenizer_kw) -> Tokenizer:
    return Tokenizer(**tokenizer_kw)


@pytest.fixture()
def mini_model_factory(
    test_data_dir, saved_tokenizer_num_labels
) -> Callable[[Optional[int]], RNNT]:
    """
    Fixture that generates a unique model each time it is called.

    Custom 'mini' model configuration for testing with 1.5k parameters.
    """

    def _create_model(seed=None):
        if seed is not None:
            set_seed(seed=seed)
        config_path = str(test_data_dir / "mini_config.yaml")
        cfg = config.load(config_path)
        rnnt_config = config.rnnt(cfg)
        model = RNNT(n_classes=saved_tokenizer_num_labels + 1, **rnnt_config)
        return model

    return _create_model


@pytest.fixture()
def checkpointer_tmp_dir(tmp_path) -> Checkpointer:
    # Checkpointer saves to a temporary directory
    return Checkpointer(tmp_path, "RNN-T")


@pytest.fixture()
def optimizer_factory() -> Callable[[RNNT], torch.optim.SGD]:
    """Create unique optimizer that is attached to given model each time fixture is called."""

    def _create_optimizer(model):
        return torch.optim.SGD(model.parameters(), lr=0.01)

    return _create_optimizer
