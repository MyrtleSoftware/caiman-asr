import pytest
import torch
from apex.contrib.transducer import TransducerJoint

from caiman_asr_train.core.batches import BatchFirst, TimeFirst
from caiman_asr_train.rnnt.model import RNNT


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
    h_broadcast = RNNT.torch_transducer_joint(f, g)
    h_apex = apex_joint(f, g, f_lens, g_lens)

    assert h_broadcast.size() == h_apex.size()
    assert len(h_broadcast.size()) == 4

    if batch_size == 1:
        # batch_size=1 is a special case of no padding
        assert torch.allclose(h_broadcast, h_apex)
    else:
        # But in general case checking `assert torch.allclose(h_broadcast, h_apex)`
        # cannot be conducted, as the apex implementation returns -1 in the locations
        # that are beyond the specified lengths
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
    # Note that dropout scales non-zeroed values by 1/(1-p) so it wouldn't be
    # expected for h and h_dropout to be equal even if, by chance, no element
    # in h_dropout was zeroed
    assert not torch.allclose(h, h_dropout)

    # However, when in eval mode, dropout shouldn't be applied
    apex_joint.eval()
    apex_joint_dropout.eval()

    h = apex_joint(f, g, f_lens, g_lens)
    h_dropout = apex_joint_dropout(f, g, f_lens, g_lens)

    assert h.shape == h_dropout.shape
    assert torch.allclose(h, h_dropout)


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
    model: RNNT,
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
    txt_seq_len = max(txt_lens)
    if device == "cuda":
        model.cuda()

    # Check that the lengths are consistent
    assert torch.equal(feat_lens, feat_lens_first_part + feat_lens_second_part)
    assert torch.equal(txt_lens, txt_lens_first_part + txt_lens_second_part)

    # Check that the model is deterministic

    fwd_a = model(
        feats=TimeFirst(feats, feat_lens),
        tokens=BatchFirst(txt, txt_lens).T,
        prefix=None,
        enc_state=None,
        pred_state=model.sos_for(batch_size),
    )

    fwd_b = model(
        feats=TimeFirst(feats, feat_lens),
        tokens=BatchFirst(txt, txt_lens).T,
        prefix=None,
        enc_state=None,
        pred_state=model.sos_for(batch_size),
    )

    assert torch.equal(fwd_a.packed_logits, fwd_b.packed_logits)

    def calc(fs, tx):
        return sum((t // 2) * (u + 1) for t, u in zip(fs, tx))

    # Check that the hyperparameters are consistent, outputs are packed
    assert fwd_a.packed_logits.shape == (calc(feat_lens, txt_lens), n_classes)

    # Check that there are the same number of utterances in the first part and
    # the second part
    assert feat_lens_first_part.shape == feat_lens_second_part.shape == (batch_size,)
    assert txt_lens_first_part.shape == txt_lens_second_part.shape == (batch_size,)

    # Apply model to the first part to get logprobs
    fwd1 = model(
        feats=TimeFirst(feats_first_part, feat_lens_first_part),
        tokens=BatchFirst(txt_first_part, txt_lens_first_part).T,
        prefix=None,
        enc_state=None,
        pred_state=model.sos_for(batch_size),
    )

    assert fwd1.state is not None

    # Apply model to the second part to get logprobs
    # Note that the state from the first part is passed in
    fwd2 = model(
        feats=TimeFirst(feats_second_part, feat_lens_second_part),
        tokens=BatchFirst(txt_second_part, txt_lens_second_part).T,
        prefix=None,
        enc_state=fwd1.state.enc,
        pred_state=fwd1.state.pred,
    )

    # Check the shape of the outputs
    assert fwd1.packed_logits.shape == (
        calc(feat_lens_first_part, txt_lens_first_part),
        n_classes,
    )
    assert fwd2.packed_logits.shape == (
        calc(feat_lens_second_part, txt_lens_second_part),
        n_classes,
    )

    print(fwd1.packed_logits.shape, feat_lens_first_part, txt_lens_first_part)
    print(fwd2.packed_logits.shape, feat_lens_second_part, txt_lens_second_part)

    # Need the intermediate activations (sometimes easier to debug than the logprobs)

    f = fwd_a.f.data
    g = fwd_b.g.data

    f_first_part, x_lens_first_part = fwd1.f.as_tuple()
    g_first_part, g_lens_first_part = fwd1.g.as_tuple()

    f_second_part, x_lens_second_part = fwd2.f.as_tuple()
    g_second_part, g_lens_second_part = fwd2.g.as_tuple()

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


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("audio_seq_len", [40, 76])
@pytest.mark.parametrize("txt_seq_len", [10, 28])
def test_no_padding(
    model_factory,
    input_dim_fixture,
    n_classes_fixture,
    batch_size,
    audio_seq_len,
    txt_seq_len,
    joint_n_hid_fixture,
):
    """Set up inputs for the simulated concatenation test.
    This is the naive test where padding is not taken into account and
    all sequences are assumed to be of the same length"""
    # Note that the audio_seq_len must be divisible by 4
    assert audio_seq_len % 4 == 0
    # Note that the txt_seq_len must be divisible by 2
    assert txt_seq_len % 2 == 0

    device = "cuda"
    feats = torch.randn(audio_seq_len, batch_size, input_dim_fixture, device=device)
    # No padding:
    feat_lens = torch.tensor(
        [audio_seq_len] * batch_size, device=device, dtype=torch.int
    )
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
        [audio_seq_len // 2] * batch_size, device=device, dtype=torch.int
    )
    feat_lens_second_half = torch.tensor(
        [audio_seq_len - audio_seq_len // 2] * batch_size,
        device=device,
        dtype=torch.int,
    )
    txt_first_half = txt[:, 0 : txt_seq_len // 2]
    txt_second_half = txt[:, txt_seq_len // 2 :]
    txt_lens_first_half = torch.tensor(
        [txt_seq_len // 2] * batch_size, device=device, dtype=torch.int
    )
    txt_lens_second_half = torch.tensor(
        [txt_seq_len - txt_seq_len // 2] * batch_size, device=device, dtype=torch.int
    )

    model = model_factory()

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


def test_padding(
    model_factory,
    input_dim_fixture,
    n_classes_fixture,
    joint_n_hid_fixture,
):
    """Set up inputs for the simulated concatenation test.
    Do the padding in the same way as during training."""
    device = "cuda"
    model = model_factory()
    batch_size = 6
    # To make the precision tests work, it seems that all of feat_lens_first_part
    # must be divisible by 2, and the max of all_feat_lens must be divisible by 2.
    # Likely the reason is the StackTime---maybe an odd-length sequence gets cut off
    feat_lens_first_part = torch.tensor(
        [158, 314, 264, 358, 982, 322], device=device, dtype=torch.int
    )
    feat_lens_second_part = torch.tensor(
        [264, 846, 338, 326, 952, 288], device=device, dtype=torch.int
    )

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

    txt_lens_first_part = torch.tensor(
        [27, 17, 28, 18, 28, 47], device=device, dtype=torch.int
    )
    txt_lens_second_part = torch.tensor(
        [91, 45, 23, 53, 61, 28], device=device, dtype=torch.int
    )

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

    print(padded_all_txt.dtype)

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
    """
    Given a list of unpadded audio features, returns padded tensor containing whole batch
    """
    max_feat_len = max(feat_lens).item()
    padded_feats_list = [
        torch.cat((f, torch.zeros(max_feat_len - f.shape[0], input_dim, device=device)))
        for f in unpad_feats
    ]
    padded_feats = torch.stack(padded_feats_list, dim=1)
    return padded_feats, max_feat_len


def generate_padded_text(txt_lens, unpad_txt, device):
    """Given list of unpadded text, returns a padded tensor containing the whole batch"""
    max_txt_len = max(txt_lens).item()
    padded_txt_list = [
        torch.cat(
            (t, torch.zeros(max_txt_len - t.shape[0], device=device, dtype=torch.int))
        )
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
