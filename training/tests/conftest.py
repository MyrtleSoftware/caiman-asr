from pathlib import Path

import pytest
import torch
import yaml
from beartype.typing import Callable, Optional, Tuple

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
    # we test using the testing config
    return str(Path(__file__).parent.parent / "configs/testing-1023sp_run.yaml")


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
def saved_tokenizer_num_labels(tokenizer) -> int:
    return tokenizer.num_labels


@pytest.fixture()
def mini_config_fp_factory(tmp_path, tokenizer_kw):
    """
    Load the config file and replace the relevant fields to make it a 'mini' config.

    'mini' model configuration has 1.5k parameters to speed up testing.
    """

    def _create_mini_config_fp(_config_fp):
        cfg = config.load(_config_fp)
        rnnt_config = config.rnnt(cfg)
        # replace the relevant fields to match
        rnnt_config["in_feats"] = 5
        rnnt_config["enc_n_hid"] = 8
        rnnt_config["enc_pre_rnn_layers"] = 1
        rnnt_config["enc_post_rnn_layers"] = 1
        rnnt_config["enc_stack_time_factor"] = 1
        rnnt_config["pred_n_hid"] = 4
        rnnt_config["pred_rnn_layers"] = 1
        rnnt_config["joint_n_hid"] = 4

        cfg["rnnt"] = rnnt_config
        cfg["tokenizer"] = tokenizer_kw
        # save the config to a temporary file
        named_tmp_file = str(tmp_path / "mini_config.yaml")
        with open(named_tmp_file, "w") as f:
            yaml.dump(cfg, f)

        return named_tmp_file

    return _create_mini_config_fp


@pytest.fixture()
def mini_model_factory(
    mini_config_fp_factory, saved_tokenizer_num_labels, config_fp
) -> Callable[[Optional[str], Optional[int]], Tuple[RNNT, str]]:
    """
    Fixture that generates a unique mini model each time it is called.
    """

    def _create_model(_config_fp=config_fp, seed=None) -> Tuple[RNNT, str]:
        if seed is not None:
            set_seed(seed=seed)
        mini_config_fp = mini_config_fp_factory(_config_fp)
        cfg = config.load(mini_config_fp)
        rnnt_config = config.rnnt(cfg)
        model = RNNT(n_classes=saved_tokenizer_num_labels + 1, **rnnt_config)
        return model, mini_config_fp

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
