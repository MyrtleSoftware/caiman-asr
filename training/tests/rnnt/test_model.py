import pytest
import torch
from apex.contrib.transducer import TransducerJoint

from rnnt_train.rnnt.loss import get_packing_meta_data
from rnnt_train.rnnt.model import RNNT


@pytest.fixture()
def apex_joint() -> TransducerJoint:
    return TransducerJoint(pack_output=False)


@pytest.fixture()
def apex_joint_dropout() -> TransducerJoint:
    return TransducerJoint(pack_output=False, dropout=True, dropout_prob=0.5)


def gen_random_f_and_g(
    batch_size: int, time_dim: int, feature_dim: int, decoded_length: int
) -> tuple:
    """
    Util to generate f, f_lens, g, g_lens matrices.
    """
    f = torch.rand(batch_size, time_dim, feature_dim)
    g = torch.rand(batch_size, decoded_length, feature_dim)
    f_lens = torch.randint(1, time_dim, size=(batch_size,), dtype=torch.int)
    f_lens[0] = time_dim
    g_lens = torch.randint(1, decoded_length, size=(batch_size,), dtype=torch.int)
    g_lens[-1] = decoded_length
    return f, f_lens, g, g_lens


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("time_dim", [4, 7])
def test_joint_apex_equivalent(batch_size, time_dim, apex_joint):
    """
    Check that JointTransducer and torch broadcasting are equivalent.
    """
    FEATURE_DIM = 2
    DECODED_LENGTH = 3
    f, f_lens, g, g_lens = gen_random_f_and_g(
        batch_size, time_dim, FEATURE_DIM, DECODED_LENGTH
    )
    if not torch.cuda.is_available():
        pytest.skip("Cuda not available so can't run this test")
    f, f_lens, g, g_lens = f.cuda(), f_lens.cuda(), g.cuda(), g_lens.cuda()
    h_broadcast = RNNT.torch_transducer_joint(f, g, f_lens, g_lens)
    h_apex = apex_joint(f, g, f_lens, g_lens)

    assert h_broadcast.size() == h_apex.size()
    assert len(h_broadcast.size()) == 4

    if batch_size == 1:
        # batch_size=1 is a special case of no padding
        assert torch.allclose(h_broadcast, h_apex)
    else:
        # But in general case we can't check `assert torch.allclose(h_broadcast, h_apex)`
        # as the apex implementation returns -1 in the locations that are beyond
        # the specified lengths
        for hs_broadcast, hs_apex, f_len, g_len in zip(
            h_broadcast, h_apex, f_lens, g_lens
        ):
            assert torch.allclose(hs_broadcast[:f_len, :g_len], hs_apex[:f_len, :g_len])


def test_apex_joint_dropout_not_eval(apex_joint, apex_joint_dropout):
    """
    Ensure apex implementation of TransducerJoint dropout isn't applied at eval time
    """
    FEATURE_DIM = 2
    DECODED_LENGTH = 3
    BATCH_SIZE = 4
    TIME_DIM = 6
    f, f_lens, g, g_lens = gen_random_f_and_g(
        BATCH_SIZE, TIME_DIM, FEATURE_DIM, DECODED_LENGTH
    )
    if not torch.cuda.is_available():
        pytest.skip("Cuda not available so can't run this test")
    f, f_lens, g, g_lens = f.cuda(), f_lens.cuda(), g.cuda(), g_lens.cuda()

    # In train mode, dropout shouldn't be applied by apex_joint_dropout and results
    # should be different to apex_joint results
    apex_joint.train()
    apex_joint_dropout.train()

    h = apex_joint(f, g, f_lens, g_lens)
    h_dropout = apex_joint_dropout(f, g, f_lens, g_lens)

    assert h.shape == h_dropout.shape
    # Note that dropout scales non-zeroed values by 1/(1-p) so we wouldn't expect h and
    # h_dropout to be equal even if, by chance, no element in h_dropout was zeroed
    assert not torch.allclose(h, h_dropout)

    # However, when in eval mode, dropout shouldn't be applied
    apex_joint.eval()
    apex_joint_dropout.eval()

    h = apex_joint(f, g, f_lens, g_lens)
    h_dropout = apex_joint_dropout(f, g, f_lens, g_lens)

    assert h.shape == h_dropout.shape
    assert torch.allclose(h, h_dropout)


@pytest.fixture()
def input_dim_fixture():
    return 240


@pytest.fixture()
def n_classes_fixture():
    return 100


@pytest.fixture()
def model_fixture(
    input_dim_fixture,
    n_classes_fixture,
    enc_n_hid_fixture,
    pred_n_hid_fixture,
    joint_n_hid_fixture,
):
    return model_setup(
        input_dim_fixture,
        n_classes_fixture,
        False,
        enc_n_hid_fixture,
        pred_n_hid_fixture,
        joint_n_hid_fixture,
    )


@pytest.fixture()
def legacy_model_fixture(
    input_dim_fixture,
    n_classes_fixture,
    enc_n_hid_fixture,
    pred_n_hid_fixture,
    joint_n_hid_fixture,
):
    return model_setup(
        input_dim_fixture,
        n_classes_fixture,
        True,
        enc_n_hid_fixture,
        pred_n_hid_fixture,
        joint_n_hid_fixture,
    )


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
):
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


def simulated_concatenation(
    batch_size,
    feats,
    feat_lens,
    txt,
    txt_lens,
    feats_first_part,
    feats_second_part,
    feat_lens_first_part,
    feat_lens_second_part,
    txt_first_part,
    txt_second_part,
    txt_lens_first_part,
    txt_lens_second_part,
    model,
    n_classes,
    joint_n_hid,
    device,
):
    """This function takes an RNNT and input features, which are divided into a
    first part and a second part. It checks that the following give the same
    outputs:

    - Applying the RNNT to the concatenation of the first part and the second part
    - Applying the RNNT to the first part, saving the state, and then applying the RNNT
      to the second part using the saved state
    """
    # Setup
    audio_seq_len = max(feat_lens)
    txt_seq_len = max(txt_lens)
    meta_data = get_packing_meta_data(
        feat_lens=feat_lens,
        txt_lens=txt_lens,
        enc_time_reduction=model.enc_stack_time_factor,
    )
    if device == "cuda":
        model.cuda()

    # Check that the lengths are consistent
    assert torch.equal(feat_lens, feat_lens_first_part + feat_lens_second_part)
    assert torch.equal(txt_lens, txt_lens_first_part + txt_lens_second_part)

    # Check that the model is deterministic
    log_probs, log_prob_lens, _ = model(
        feats, feat_lens, txt, txt_lens, batch_offset=meta_data["batch_offset"]
    )
    log_probs_duplicate, log_prob_lens_duplicate, _ = model(
        feats, feat_lens, txt, txt_lens, batch_offset=meta_data["batch_offset"]
    )
    assert torch.equal(log_probs, log_probs_duplicate)
    assert torch.equal(log_prob_lens, log_prob_lens_duplicate)

    # Check that the hyperparameters are consistent
    assert log_probs.shape == (
        batch_size,
        audio_seq_len // 2,
        txt_seq_len + 1,
        n_classes,
    )

    # Check that there are the same number of utterances in the first part and
    # the second part
    assert feat_lens_first_part.shape == feat_lens_second_part.shape == (batch_size,)
    assert txt_lens_first_part.shape == txt_lens_second_part.shape == (batch_size,)

    # Apply model to the first part to get logprobs
    log_probs_first_part, _, state = model(
        feats_first_part,
        feat_lens_first_part,
        txt_first_part,
        txt_lens_first_part,
        batch_offset=meta_data["batch_offset"],
    )

    # Apply model to the second part to get logprobs
    # Note that we pass in the state from the first part
    log_probs_second_part, _, _ = model(
        feats_second_part,
        feat_lens_second_part,
        txt_second_part,
        txt_lens_second_part,
        batch_offset=meta_data["batch_offset"],
        enc_state=state.enc_state,
        pred_net_state=state.pred_net_state,
    )

    # Check the shape of the outputs
    assert log_probs_first_part.shape == (
        batch_size,
        feats_first_part.shape[0] // 2,
        txt_first_part.shape[1] + 1,
        n_classes,
    )
    assert log_probs_second_part.shape == (
        batch_size,
        feats_second_part.shape[0] // 2,
        txt_second_part.shape[1] + 1,
        n_classes,
    )

    # We also want the intermediate activations
    # (sometimes easier to debug than the logprobs)

    # For the whole sequence
    (f, _), (g, _), _ = model.enc_pred(
        feats,
        feat_lens,
        txt,
        txt_lens,
    )

    # For the first part
    (
        (f_first_part, x_lens_first_part),
        (g_first_part, g_lens_first_part),
        state1,
    ) = model.enc_pred(
        feats_first_part,
        feat_lens_first_part,
        txt_first_part,
        txt_lens_first_part,
    )

    # For the second part
    (
        (f_second_part, x_lens_second_part),
        (g_second_part, g_lens_second_part),
        _,
    ) = model.enc_pred(
        feats_second_part,
        feat_lens_second_part,
        txt_second_part,
        txt_lens_second_part,
        enc_state=state1.enc_state,
        pred_net_state=state1.pred_net_state,
    )

    txt_seq_len_first_part = max(txt_lens_first_part)
    txt_seq_len_second_part = max(txt_lens_second_part)

    # Check shapes of the pred net output
    assert g.shape == (batch_size, txt_seq_len + 1, joint_n_hid)
    assert g_first_part.shape == (batch_size, txt_seq_len_first_part + 1, joint_n_hid)
    assert g_second_part.shape == (batch_size, txt_seq_len_second_part + 1, joint_n_hid)

    for i in range(batch_size):
        aud_len1 = x_lens_first_part[i]
        aud_len2 = x_lens_second_part[i]
        txt_len1 = g_lens_first_part[i]
        txt_len2 = g_lens_second_part[i]
        # Check that the first parts of the encoder output match
        assert torch.allclose(
            f[i, 0:aud_len1, :],
            f_first_part[i, 0:aud_len1, :],
            atol=1e-3,
        ), f"Precision error in element {i} of batch"
        # And that the second parts of the encoder output match
        assert torch.allclose(
            f[i, aud_len1 : aud_len1 + aud_len2, :],
            f_second_part[i, 0:aud_len2, :],
            atol=1e-3,
        ), f"Precision error in element {i} of batch"
        # Check that the first parts of the pred net output match
        assert torch.allclose(
            g[i, 0:txt_len1, :],
            g_first_part[i, 0:txt_len1, :],
            atol=1e-3,
        ), f"Precision error in element {i} of batch"
        # And that the second parts of the pred net output match
        assert torch.allclose(
            g[i, txt_len1 - 1 : txt_len1 + txt_len2 - 1, :],
            g_second_part[i, 0:txt_len2, :],
            atol=1e-3,
        ), f"Precision error in element {i} of batch"
        # Check that the first parts of the logprobs match
        assert torch.allclose(
            log_probs[i, 0:aud_len1, 0:txt_len1, :],
            log_probs_first_part[i, 0:aud_len1, 0:txt_len1, :],
            atol=1e-3,
        ), f"Precision error in element {i} of batch"
        # And that the second parts of the logprobs match
        assert torch.allclose(
            log_probs[
                i,
                aud_len1 : aud_len1 + aud_len2,
                txt_len1 - 1 : txt_len1 + txt_len2 - 1,
                :,
            ],
            log_probs_second_part[i, 0:aud_len2, 0:txt_len2, :],
            atol=1e-3,
        ), f"Precision error in element {i} of batch"


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("audio_seq_len", [40, 76])
@pytest.mark.parametrize("txt_seq_len", [10, 28])
@pytest.mark.parametrize("use_legacy", [True, False])
def test_no_padding(
    model_fixture,
    legacy_model_fixture,
    input_dim_fixture,
    n_classes_fixture,
    batch_size,
    audio_seq_len,
    txt_seq_len,
    use_legacy,
    joint_n_hid_fixture,
):
    """Set up inputs for the simulated concatenation test.
    This is the naive test where we don't take padding into account and
    assume all sequences are the same length"""
    # Note that the audio_seq_len must be divisible by 4
    assert audio_seq_len % 4 == 0
    # Note that the txt_seq_len must be divisible by 2
    assert txt_seq_len % 2 == 0

    device = "cpu" if use_legacy else "cuda"
    feats = torch.randn(audio_seq_len, batch_size, input_dim_fixture, device=device)
    # No padding:
    feat_lens = torch.tensor([audio_seq_len] * batch_size, device=device)
    txt = torch.randint(
        0,
        n_classes_fixture - 1,
        (batch_size, txt_seq_len),
        device=device,
    )
    # No padding:
    txt_lens = torch.tensor([txt_seq_len] * batch_size, device=device, dtype=torch.int)
    feats_first_half = feats[0 : audio_seq_len // 2, :, :]
    feats_second_half = feats[audio_seq_len // 2 :, :, :]
    feat_lens_first_half = torch.tensor(
        [audio_seq_len // 2] * batch_size, device=device
    )
    feat_lens_second_half = torch.tensor(
        [audio_seq_len - audio_seq_len // 2] * batch_size, device=device
    )
    txt_first_half = txt[:, 0 : txt_seq_len // 2]
    txt_second_half = txt[:, txt_seq_len // 2 :]
    txt_lens_first_half = torch.tensor(
        [txt_seq_len // 2] * batch_size, device=device, dtype=torch.int
    )
    txt_lens_second_half = torch.tensor(
        [txt_seq_len - txt_seq_len // 2] * batch_size, device=device, dtype=torch.int
    )
    if use_legacy:
        model = legacy_model_fixture
    else:
        model = model_fixture
    simulated_concatenation(
        batch_size,
        feats,
        feat_lens,
        txt,
        txt_lens,
        feats_first_half,
        feats_second_half,
        feat_lens_first_half,
        feat_lens_second_half,
        txt_first_half,
        txt_second_half,
        txt_lens_first_half,
        txt_lens_second_half,
        model,
        n_classes_fixture,
        joint_n_hid=joint_n_hid_fixture,
        device=device,
    )


@pytest.mark.parametrize("use_legacy", [True, False])
def test_padding(
    model_fixture,
    legacy_model_fixture,
    input_dim_fixture,
    n_classes_fixture,
    use_legacy,
    joint_n_hid_fixture,
):
    """Set up inputs for the simulated concatenation test.
    We do the padding in the same way as during training."""
    device = "cpu" if use_legacy else "cuda"
    if use_legacy:
        model = legacy_model_fixture
    else:
        model = model_fixture
    batch_size = 6
    # To make the precision tests work, it seems that all of feat_lens_first_part
    # must be divisible by 2, and the max of all_feat_lens must be divisible by 2.
    # Likely the reason is the StackTime---maybe an odd-length sequence gets cut off
    feat_lens_first_part = torch.tensor([158, 314, 264, 358, 982, 322])
    feat_lens_second_part = torch.tensor([265, 846, 338, 326, 952, 289])

    for x in feat_lens_first_part:
        assert x % 2 == 0

    feats_list_first_part = generate_unpadded_feats(
        feat_lens_first_part, input_dim_fixture, device
    )

    feats_list_second_part = generate_unpadded_feats(
        feat_lens_second_part, input_dim_fixture, device
    )

    all_feat_lens = feat_lens_first_part + feat_lens_second_part
    assert max(all_feat_lens) % 2 == 0

    unpad_all_feats = [
        torch.concat((f1, f2))
        for (f1, f2) in zip(feats_list_first_part, feats_list_second_part)
    ]

    for i, f in enumerate(unpad_all_feats):
        assert f.shape[0] == all_feat_lens[i]

    padded_all_feats, max_feat_len = generate_padded_feats(
        all_feat_lens, unpad_all_feats, input_dim_fixture, device
    )

    assert max_feat_len == 1934
    assert max_feat_len % 2 == 0
    assert padded_all_feats.shape == (max_feat_len, batch_size, input_dim_fixture)

    padded_first_part_feats, max_feat_len_first_part = generate_padded_feats(
        feat_lens_first_part, feats_list_first_part, input_dim_fixture, device
    )

    assert max_feat_len_first_part == 982
    assert max_feat_len_first_part % 2 == 0
    assert padded_first_part_feats.shape == (
        max_feat_len_first_part,
        batch_size,
        input_dim_fixture,
    )

    padded_second_part_feats, max_feat_len_second_part = generate_padded_feats(
        feat_lens_second_part, feats_list_second_part, input_dim_fixture, device
    )
    assert max_feat_len_second_part == 952
    assert padded_second_part_feats.shape == (
        max_feat_len_second_part,
        batch_size,
        input_dim_fixture,
    )

    txt_lens_first_part = torch.tensor([27, 17, 28, 18, 28, 47])
    txt_lens_second_part = torch.tensor([91, 45, 23, 53, 61, 28])

    txt_list_first_part = generate_unpadded_text(
        txt_lens_first_part, n_classes_fixture, device
    )
    txt_list_second_part = generate_unpadded_text(
        txt_lens_second_part, n_classes_fixture, device
    )

    all_txt_lens = txt_lens_first_part + txt_lens_second_part

    unpad_all_txt = [
        torch.concat((t1, t2))
        for (t1, t2) in zip(txt_list_first_part, txt_list_second_part)
    ]

    for i, t in enumerate(unpad_all_txt):
        assert t.shape[0] == all_txt_lens[i]

    padded_all_txt, max_txt_len = generate_padded_text(
        all_txt_lens, unpad_all_txt, device
    )

    assert max_txt_len == 118

    assert padded_all_txt.shape == (batch_size, max_txt_len)

    padded_first_part_txt, max_txt_len_first_part = generate_padded_text(
        txt_lens_first_part, txt_list_first_part, device
    )

    assert max_txt_len_first_part == 47
    assert padded_first_part_txt.shape == (batch_size, max_txt_len_first_part)

    padded_second_part_txt, max_txt_len_second_part = generate_padded_text(
        txt_lens_second_part, txt_list_second_part, device
    )

    assert max_txt_len_second_part == 91
    assert padded_second_part_txt.shape == (batch_size, max_txt_len_second_part)

    simulated_concatenation(
        batch_size,
        padded_all_feats,
        all_feat_lens,
        padded_all_txt,
        all_txt_lens,
        padded_first_part_feats,
        padded_second_part_feats,
        feat_lens_first_part,
        feat_lens_second_part,
        padded_first_part_txt,
        padded_second_part_txt,
        txt_lens_first_part,
        txt_lens_second_part,
        model,
        n_classes_fixture,
        joint_n_hid=joint_n_hid_fixture,
        device=device,
    )


def generate_padded_feats(feat_lens, unpad_feats, input_dim, device):
    """Given a list of unpadded audio feature tensors, returns a padded tensor containing the whole batch"""
    max_feat_len = max(feat_lens).item()
    padded_feats_list = [
        torch.cat((f, torch.zeros(max_feat_len - f.shape[0], input_dim, device=device)))
        for f in unpad_feats
    ]
    padded_feats = torch.stack(padded_feats_list, dim=1)
    return padded_feats, max_feat_len


def generate_padded_text(txt_lens, unpad_txt, device):
    """Given a list of unpadded text tensors, returns a padded tensor containing the whole batch"""
    max_txt_len = max(txt_lens).item()
    padded_txt_list = [
        torch.cat((t, torch.zeros(max_txt_len - t.shape[0], device=device)))
        for t in unpad_txt
    ]
    padded_txt = torch.stack(padded_txt_list, dim=0)
    return padded_txt, max_txt_len


def generate_unpadded_feats(feat_lens, input_dim, device):
    """Returns a list of fake audio feature tensors"""
    return [torch.randn(feat_len, input_dim, device=device) for feat_len in feat_lens]


def generate_unpadded_text(txt_lens, n_classes, device):
    """Returns a list of fake text tensors"""
    return [
        torch.randint(0, n_classes - 1, (txt_len,), device=device)
        for txt_len in txt_lens
    ]
