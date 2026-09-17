from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml
from apex.optimizers import FusedLAMB
from beartype.typing import Callable, Optional, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.train_utils.build_optimizer import build_fused_lamb
from caiman_asr_train.unittesting.dataload_args import gen_dataload_args
from caiman_asr_train.unittesting.tokenizer import load_tokenizer_kw
from caiman_asr_train.utils.seed import set_seed


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    # use path relative to this file to find the testing data directory
    test_data_dir = Path(__file__).parent / "test_data"
    assert test_data_dir.is_dir(), f"{test_data_dir=} doesn't exist"
    return test_data_dir


@pytest.fixture(scope="session")
def config_fp() -> str:
    # Test using the testing config
    return str(Path(__file__).parent.parent / "configs/testing-1023sp.yaml")


@pytest.fixture()
def tokenizer_kw(tmp_path, config_fp, test_data_dir):
    return load_tokenizer_kw(tmp_path, config_fp, test_data_dir)


@pytest.fixture()
def val_tokenizer_kw(tmp_path, config_fp, test_data_dir):
    kw = load_tokenizer_kw(tmp_path, config_fp, test_data_dir)
    kw["sampling"] = 0.0
    kw["label_smoothing_kwargs"] = None
    return kw


@pytest.fixture()
def mel_stats_dir(test_data_dir) -> str:
    return str(test_data_dir)


@pytest.fixture()
def tokenizer(tokenizer_kw) -> Tokenizer:
    return Tokenizer(**tokenizer_kw)


@pytest.fixture()
def val_tokenizer(val_tokenizer_kw) -> Tokenizer:
    return Tokenizer(**val_tokenizer_kw)


@pytest.fixture()
def saved_tokenizer_num_labels(tokenizer) -> int:
    return tokenizer.num_labels


@pytest.fixture()
def mini_config_fp_factory(tmp_path, mel_stats_dir, ngram_subdir, test_data_dir):
    """
    Load the config file and replace the relevant fields to make it a 'mini' config.

    'mini' model configuration has 1.5k parameters to speed up testing.
    """

    def _update_langs(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Update the LANGS key
        cfg["langs"] = ["en"]

        # save the config to a temporary file
        mini_cfg_path = str(tmp_path / "mini_config.yaml")
        with open(mini_cfg_path, "w") as f:
            yaml.dump(cfg, f)

        return mini_cfg_path

    def _create_mini_config_fp(_config_fp):
        mini_cfg_fp = _update_langs(_config_fp)
        cfg = config.load(mini_cfg_fp)
        rnnt_config = config.rnnt(cfg)
        # replace the relevant fields to match
        rnnt_config["in_feats"] = 240
        rnnt_config["enc_n_hid"] = 8
        rnnt_config["enc_pre_rnn_layers"] = 1
        rnnt_config["enc_post_rnn_layers"] = 1
        rnnt_config["enc_stack_time_factor"] = 1
        rnnt_config["pred_n_hid"] = 4
        rnnt_config["pred_rnn_layers"] = 1
        rnnt_config["joint_n_hid"] = 4

        cfg["rnnt"] = rnnt_config

        # as mini_model_factory reloads the model config
        # which will reset the labels, drop labels here
        cfg["langs"] = ["en"]
        del cfg["tokenizer"]["labels"]

        # The test expect specific token outputs so we keep the
        # legacy settings fos now
        cfg["tokenizer"]["sentpiece_model"] = str(test_data_dir / "librispeech29.model")

        cfg["input_val"]["filterbank_features"]["stats_path"] = mel_stats_dir
        cfg["input_train"]["filterbank_features"]["stats_path"] = mel_stats_dir

        cfg["ngram"]["ngram_path"] = ngram_subdir

        # save the config to a temporary file
        with open(mini_cfg_fp, "w") as f:
            yaml.dump(cfg, f)

        return mini_cfg_fp

    return _create_mini_config_fp


@pytest.fixture()
def mini_config_fp(mini_config_fp_factory, config_fp) -> str:
    return mini_config_fp_factory(config_fp)


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
        rnnt = config.rnnt(cfg)

        if rnnt["conditional_encoder"] and rnnt["prefix_token_map"] is None:
            # Mock the pnc/nopnc behaviour, this is required because the
            # yaml config doesn't specify the prefix_token_map
            rnnt["prefix_token_map"] = {i: i for i in range(2)}

        model = RNNT(n_classes=saved_tokenizer_num_labels + 1, **rnnt)
        return model, mini_config_fp

    return _create_model


@pytest.fixture()
def checkpointer_tmp_dir(tmp_path) -> Checkpointer:
    # Checkpointer saves to a temporary directory
    return Checkpointer(tmp_path, "RNN-T")


@pytest.fixture()
def optimizer_factory() -> Callable[[Union[RNNT, DDP]], FusedLAMB]:
    """
    Create unique optimizer that is attached to given model each time fixture is called.
    """
    args = Namespace(lr=0.1, weight_decay=1e-6, beta1=0.999, beta2=0.99, clip_norm=1)

    def _create_optimizer(model):
        return build_fused_lamb(args=args, model=model, opt_eps=1e-9)

    return _create_optimizer


@pytest.fixture(scope="session")
def dataload_args(test_data_dir, tmp_path_factory) -> Namespace:
    return gen_dataload_args(test_data_dir, tmp_path_factory.mktemp("config"))


@pytest.fixture(scope="session")
def melmeans(test_data_dir) -> torch.Tensor:
    return torch.load(test_data_dir / "melmeans.pt", weights_only=False)


@pytest.fixture(scope="session")
def melvars(test_data_dir) -> torch.Tensor:
    return torch.load(test_data_dir / "melvars.pt", weights_only=False)


@pytest.fixture(scope="session")
def ngram_subdir(test_data_dir):
    return str(test_data_dir / "ngram")


@pytest.fixture(scope="session")
def ngram_path(test_data_dir) -> str:
    return str(test_data_dir / "ngram/ngram.binary")


# The autotuner prunes each backend's feasible set to the tree-ranker's top-K, but that K
# still far exceeds the handful of configs CI can afford (and must hold even when the
# feasible set is smaller than K). This fixture caps each backend to a few launchable tiles
# for any test that drives real kernel selection (autotune select_*, or a TileLangLSTM
# forward); the full grid stays covered by test_config_generators_yield_feasible_configs.
_MAX_TEST_CONFIGS = 4


@pytest.fixture()
def cap_autotune_configs(monkeypatch):
    """
    Cap every TileLang LSTM backend's feasible-config list (see ``_MAX_TEST_CONFIGS``)
    so each ``select_*`` call compiles/benches < 5 configs per backend instead of the
    full grid, and clear the ``@cache``'d select/tune memos around the test so the cap
    takes effect (an uncapped cache entry would mask it) and never leaks out.

    Imports are local so collecting non-TileLang tests never pulls in tilelang.
    """
    import dataclasses

    from caiman_asr_train.tilelstm.autotune import tuners
    from caiman_asr_train.tilelstm.autotune.select import select_fastest

    def _cap(bk):
        def capped(*a, **k):
            return bk.make_configs(*a, **k)[:_MAX_TEST_CONFIGS]

        return dataclasses.replace(bk, make_configs=capped)

    for k in list(tuners._BACKENDS):
        monkeypatch.setitem(tuners._BACKENDS, k, _cap(tuners._BACKENDS[k]))
    select_fastest.cache_clear()
    tuners.tune.cache_clear()
    yield
    select_fastest.cache_clear()
    tuners.tune.cache_clear()
