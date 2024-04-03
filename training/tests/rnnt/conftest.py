import pytest

from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.utils.seed import set_seed


@pytest.fixture()
def input_dim_fixture():
    return 240


@pytest.fixture()
def n_classes_fixture():
    return 100


@pytest.fixture()
def model_factory(
    input_dim_fixture,
    n_classes_fixture,
    enc_n_hid_fixture,
    pred_n_hid_fixture,
    joint_n_hid_fixture,
):
    def _gen_model(seed=0, local_rank=0):
        return model_setup(
            input_dim_fixture,
            n_classes_fixture,
            False,
            enc_n_hid_fixture,
            pred_n_hid_fixture,
            joint_n_hid_fixture,
            seed=seed,
            local_rank=local_rank,
        )

    return _gen_model


@pytest.fixture()
def legacy_model_factory(
    input_dim_fixture,
    n_classes_fixture,
    enc_n_hid_fixture,
    pred_n_hid_fixture,
    joint_n_hid_fixture,
):
    def _gen_model(seed=0, local_rank=0):
        return model_setup(
            input_dim_fixture,
            n_classes_fixture,
            True,
            enc_n_hid_fixture,
            pred_n_hid_fixture,
            joint_n_hid_fixture,
            seed=seed,
            local_rank=local_rank,
        )

    return _gen_model


@pytest.fixture()
def enc_n_hid_fixture():
    return 11


@pytest.fixture()
def pred_n_hid_fixture():
    return 12


@pytest.fixture()
def joint_n_hid_fixture():
    return 13


def model_setup(
    input_dim_fixture,
    n_classes_fixture,
    gpu_unavailable,
    enc_n_hid_fixture,
    pred_n_hid_fixture,
    joint_n_hid_fixture,
    seed=0,
    local_rank=0,
):
    set_seed(seed=seed, local_rank=local_rank)
    return RNNT(
        n_classes=n_classes_fixture,
        in_feats=input_dim_fixture,
        enc_n_hid=enc_n_hid_fixture,
        enc_pre_rnn_layers=2,
        enc_post_rnn_layers=3,
        enc_stack_time_factor=2,
        enc_dropout=0.0,
        enc_batch_norm=False,
        enc_freeze=False,
        pred_n_hid=pred_n_hid_fixture,
        pred_rnn_layers=2,
        pred_dropout=0.0,
        pred_batch_norm=False,
        joint_n_hid=joint_n_hid_fixture,
        joint_dropout=0.0,
        joint_net_lr_factor=1.0,
        joint_apex_transducer=None,
        joint_apex_relu_dropout=False,
        forget_gate_bias=1.0,
        custom_lstm=True,
        quantize=False,
        # Must have dropout 0.0 for deterministic results
        enc_rw_dropout=0.0,
        pred_rw_dropout=0.0,
        gpu_unavailable=gpu_unavailable,
    )
