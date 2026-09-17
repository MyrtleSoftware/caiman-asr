from typing import Literal

import pytest
import torch
from beartype import beartype

from caiman_asr_train.core.lstm import CustomConfig, TorchConfig
from caiman_asr_train.core.model import RNNT, ConfigRNNT, TimeFirst


@beartype
def make_model(
    prefix: bool, backend: Literal["torch", "torch_all", "custom"]
) -> tuple[ConfigRNNT, RNNT]:

    n_classes = 10

    backends = {
        # "torch" matches the training path (no per-timestep all_h).
        "torch": TorchConfig(return_all=False),
        # "torch_all" returns all_h, enabling variable-length state threading
        # on the (CPU-capable) torch backend, like the custom backend.
        "torch_all": TorchConfig(return_all=True),
        "custom": CustomConfig(0.0),
    }

    config = ConfigRNNT(
        n_classes=n_classes,
        in_feats=80,
        enc_n_hid=128,
        enc_pre_rnn_layers=2,
        enc_post_rnn_layers=2,
        enc_stack_time_factor=2,
        enc_dropout=0.0,
        prefix_token_map={i: i for i in range(n_classes)} if prefix else None,
        pred_n_hid=3,
        pred_rnn_layers=2,
        pred_dropout=0.0,
        backend=backends[backend],
        weights_init_scale=None,
        forget_gate_bias=None,
        hidden_hidden_bias_scale=None,
        joint_n_hid=42,
        joint_dropout=0.0,
    )

    model = RNNT(config)
    model.eval()

    return config, model


@beartype
def make_feats(B: int, T: int, P: int, conf: ConfigRNNT) -> tuple[TimeFirst, TimeFirst]:

    prefix = TimeFirst(
        data=torch.randint(0, conf.n_classes, (P, B)),
        lens=torch.full((B,), P, dtype=torch.long),
    )

    feats = TimeFirst(
        data=torch.randn(T, B, conf.in_feats),
        lens=torch.randint(1, T + 1, (B,)),
    )

    # Make one feature fill the sequence
    feats.lens[0] = T

    # Make sure all multiples of stack time factor:
    n = conf.enc_stack_time_factor
    feats.lens = ((feats.lens + 1) // n) * n

    return prefix, feats


def test_encode_no_prefix():
    # Set seed
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="torch")

    B, T, P = 32, 100, 2

    _, feats = make_feats(B, T, P, conf)

    # Given all different lengths
    assert not (feats.lens[0] == feats.lens).all()

    # Pytorch backend can't doesn't handle variable length
    _, state = model.encode(state=None, feats=feats, prefix=None)
    assert state is None

    # Now fixed length
    feats = TimeFirst(feats.data, torch.full_like(feats.lens, T))

    # Should handle
    all_encs, state = model.encode(state=None, feats=feats, prefix=None)
    assert state is not None

    # Test splitting
    beg_feats = TimeFirst(feats.data[: T // 2], torch.full_like(feats.lens, T // 2))
    end_feats = TimeFirst(feats.data[T // 2 :], torch.full_like(feats.lens, T - T // 2))

    beg_encs, state = model.encode(state=None, feats=beg_feats, prefix=None)
    end_encs, state = model.encode(state=state, feats=end_feats, prefix=None)

    print(all_encs.data.shape, beg_encs.data.shape, end_encs.data.shape)

    assert torch.allclose(all_encs.data[: T // 4], beg_encs.data)
    assert torch.allclose(all_encs.data[T // 4 :], end_encs.data)


def test_encode_rejects_prefix_without_prefix_map():
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="torch")

    B, T, P = 32, 100, 2

    prefix, feats = make_feats(B, T, P, conf)

    with pytest.raises(AssertionError):
        model.encode(state=None, feats=feats, prefix=prefix)


def test_encode_split_no_prefix():
    # Seed
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="custom")

    model = model.cuda()

    B, T, P = 32, 100, 2

    _, f1 = make_feats(B, T, P, conf)
    _, f2 = make_feats(B, T, P, conf)

    f1 = f1.to(device="cuda")
    f2 = f2.to(device="cuda")

    e1, state1 = model.encode(state=None, feats=f1, prefix=None)
    assert state1 is not None
    e2, state2 = model.encode(state=state1, feats=f2, prefix=None)
    assert state2 is not None

    f3 = TimeFirst(torch.zeros((2 * T), B, f1.data.shape[2]), f1.lens + f2.lens).to(
        device="cuda"
    )

    for i in range(B):
        f3.data[: f1.lens[i], i] = f1.data[: f1.lens[i], i]
        f3.data[f1.lens[i] : f1.lens[i] + f2.lens[i], i] = f2.data[: f2.lens[i], i]

    e3, state3 = model.encode(state=None, feats=f3, prefix=None)
    assert state3 is not None

    a = state2.pre_stack.hidd
    b = state3.pre_stack.hidd

    print(a.shape, b.shape)
    print(a - b)
    assert torch.allclose(a, b)


def test_encode_split_no_prefix_torch():
    """
    Torch backend with return_all=True threads variable-length encoder state
    across a split, exactly like the custom backend, and runs on CPU.

    Before return_all existed the torch backend could not produce all_h, so
    test_encode_no_prefix had to assert state is None for variable lengths and
    only test splitting with fixed lengths. This is the variable-length
    equivalent (and the CPU coverage the removed legacy backend used to give).
    """
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="torch_all")

    B, T, P = 32, 100, 2

    _, f1 = make_feats(B, T, P, conf)
    _, f2 = make_feats(B, T, P, conf)

    # Lengths are variable: this is what the torch backend couldn't thread before.
    assert not (f1.lens[0] == f1.lens).all()

    e1, state1 = model.encode(state=None, feats=f1, prefix=None)
    assert state1 is not None
    e2, state2 = model.encode(state=state1, feats=f2, prefix=None)
    assert state2 is not None

    f3 = TimeFirst(torch.zeros((2 * T), B, f1.data.shape[2]), f1.lens + f2.lens)

    for i in range(B):
        f3.data[: f1.lens[i], i] = f1.data[: f1.lens[i], i]
        f3.data[f1.lens[i] : f1.lens[i] + f2.lens[i], i] = f2.data[: f2.lens[i], i]

    e3, state3 = model.encode(state=None, feats=f3, prefix=None)
    assert state3 is not None

    # Threading f1 then f2 must match a single pass over their concatenation.
    assert torch.allclose(state2.pre_stack.hidd, state3.pre_stack.hidd, atol=1e-5)


@beartype
def make_tokens(
    B: int, U: int, P: int, conf: ConfigRNNT
) -> tuple[TimeFirst, TimeFirst]:

    # Minus 1 as the last token is the blank

    tokens = TimeFirst(
        data=torch.randint(0, conf.n_classes - 1, (U, B)),
        lens=torch.randint(1, U + 1, (B,)),
    )

    prefix = TimeFirst(
        data=torch.randint(0, conf.n_classes - 1, (P, B)),
        lens=torch.full((B,), P, dtype=torch.long),
    )

    tokens.lens[0] = U

    return tokens, prefix


def test_encode_prefix():
    # Seed
    torch.manual_seed(42)

    conf, model = make_model(prefix=True, backend="torch")

    B, T, P = 32, 100, 2

    prefix, feats = make_feats(B, T, P, conf)

    # Given all different lengths
    assert not (feats.lens[0] == feats.lens).all()

    enc, state = model.encode(state=None, feats=feats, prefix=prefix)
    assert state is None

    print(enc.lens * conf.enc_stack_time_factor)
    print(feats.lens)
    assert (feats.lens == enc.lens * conf.enc_stack_time_factor).all()

    # Should be the same as separate calls
    prefix_state = model.encode(state=None, feats=None, prefix=prefix)
    assert prefix_state is not None

    enc1, state1 = model.encode(state=prefix_state, feats=feats, prefix=None)
    assert state1 is None
    assert torch.allclose(enc.data, enc1.data)


def test_predict_no_prefix():
    # Seed
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="custom")
    model = model.cuda()

    tokens, _ = make_tokens(B=32, U=20, P=2, conf=conf)
    tokens = tokens.to(device="cuda")

    # SoS
    sos = model.sos_for(tokens)
    g, state = model.predict(state=sos, prefix=None, tokens=tokens)
    assert (g.lens == tokens.lens + 1).all()

    # CustomSoS
    assert state is not None
    g, _ = model.predict(state=state, prefix=None, tokens=tokens)
    assert (g.lens == tokens.lens + 1).all()


def test_predict_split_no_prefix():
    # Seed
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="custom")
    model = model.cuda()

    B, U, P = 32, 100, 3

    tok_1, _ = make_tokens(B=B, U=U, P=P, conf=conf)
    tok_2, _ = make_tokens(B=B, U=U, P=P, conf=conf)

    tok_1 = tok_1.to(device="cuda")
    tok_2 = tok_2.to(device="cuda")

    g1, state_1 = model.predict(state=model.sos_for(tok_1), prefix=None, tokens=tok_1)

    assert state_1 is not None
    g2, state_2 = model.predict(state=state_1, prefix=None, tokens=tok_2)

    tok_3 = TimeFirst(
        torch.zeros((2 * U, B), dtype=torch.int), tok_1.lens + tok_2.lens
    ).to(device="cuda")

    for i in range(B):
        n = tok_1.lens[i]
        m = tok_2.lens[i]
        tok_3.data[:n, i] = tok_1.data[:n, i]
        tok_3.data[n : n + m, i] = tok_2.data[:m, i]

    g3, state_3 = model.predict(state=model.sos_for(tok_3), prefix=None, tokens=tok_3)

    assert state_2 is not None and state_3 is not None
    assert torch.allclose(state_2.embed, state_3.embed)
    assert torch.allclose(state_2.state.cell, state_3.state.cell)


def test_predict_split_no_prefix_torch():
    """
    Torch backend with return_all=True threads variable-length predictor
    state across a split, like the custom backend, and runs on CPU.

    predict() goes through _apply_rnn with penultimate=True, which can only
    return a (non-None) threaded state when all_h is available -- impossible on
    the torch backend until return_all.
    """
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="torch_all")

    B, U, P = 32, 100, 3

    tok_1, _ = make_tokens(B=B, U=U, P=P, conf=conf)
    tok_2, _ = make_tokens(B=B, U=U, P=P, conf=conf)

    g1, state_1 = model.predict(state=model.sos_for(tok_1), prefix=None, tokens=tok_1)

    assert state_1 is not None
    g2, state_2 = model.predict(state=state_1, prefix=None, tokens=tok_2)

    tok_3 = TimeFirst(torch.zeros((2 * U, B), dtype=torch.int), tok_1.lens + tok_2.lens)

    for i in range(B):
        n = tok_1.lens[i]
        m = tok_2.lens[i]
        tok_3.data[:n, i] = tok_1.data[:n, i]
        tok_3.data[n : n + m, i] = tok_2.data[:m, i]

    g3, state_3 = model.predict(state=model.sos_for(tok_3), prefix=None, tokens=tok_3)

    assert state_2 is not None and state_3 is not None
    assert torch.allclose(state_2.embed, state_3.embed, atol=1e-5)
    assert torch.allclose(state_2.state.cell, state_3.state.cell, atol=1e-5)


def test_predict_prefix():
    # Seed
    torch.manual_seed(42)

    conf, model = make_model(prefix=True, backend="custom")
    model = model.cuda()

    tokens, prefix = make_tokens(B=32, U=20, P=2, conf=conf)
    tokens = tokens.to(device="cuda")
    prefix = prefix.to(device="cuda")

    # Prefix and tokens

    # SoS
    g, state = model.predict(state=model.sos_for(tokens), prefix=prefix, tokens=tokens)
    assert (g.lens == tokens.lens + 1).all()

    # CustomSoS
    assert state is not None
    g, _ = model.predict(state=state, prefix=prefix, tokens=tokens)
    assert (g.lens == tokens.lens + 1).all()

    # Just prefix

    g, state = model.predict(state=model.sos_for(prefix), prefix=prefix, tokens=None)
    assert (g.lens == 1).all()

    assert state is not None
    g, _ = model.predict(state=state, prefix=prefix, tokens=None)
    assert (g.lens == 1).all()


def test_predict_just_sos():
    # Seed
    torch.manual_seed(42)

    conf, model = make_model(prefix=False, backend="custom")
    model = model.cuda()

    tokens, _ = make_tokens(B=32, U=20, P=2, conf=conf)
    tokens = tokens.to(device="cuda")

    sos = model.sos_for(tokens)

    _, sos_state = model.predict(state=sos, prefix=None, tokens=None)

    assert sos_state is not None

    assert torch.allclose(sos_state.embed, sos.embed)
    assert torch.allclose(sos_state.state.cell, sos.state.cell)

    _, a = model.predict(state=sos, prefix=None, tokens=tokens)
    _, b = model.predict(state=sos_state, prefix=None, tokens=tokens)

    assert a is not None and b is not None
    assert torch.allclose(a.embed, b.embed)
    assert torch.allclose(a.state.cell, b.state.cell)


@pytest.mark.parametrize("backend", ["torch", "torch_all"])
def test_default_enc_state_shape_and_zeroed(backend):
    conf, model = make_model(prefix=False, backend=backend)

    B = 4
    state = model.default_enc_state_for(B)

    for part, layers in [
        (state.pre_stack, conf.enc_pre_rnn_layers),
        (state.aft_stack, conf.enc_post_rnn_layers),
    ]:
        for tensor in part.as_tuple():
            assert tensor.shape == (layers, B, conf.enc_n_hid)
            assert not tensor.any(), "default encoder state must be zeroed"


def test_default_enc_state_deduces_batch_size():
    conf, model = make_model(prefix=False, backend="torch")

    _, feats = make_feats(B=6, T=8, P=2, conf=conf)

    from_time_first = model.default_enc_state_for(feats)
    from_batch_first = model.default_enc_state_for(feats.T)
    from_int = model.default_enc_state_for(6)

    for state in [from_time_first, from_batch_first, from_int]:
        assert state.pre_stack.hidd.shape == (
            conf.enc_pre_rnn_layers,
            6,
            conf.enc_n_hid,
        )


@pytest.mark.parametrize("backend", ["torch", "torch_all"])
@pytest.mark.parametrize("prefix", [False, True])
def test_default_enc_state_equivalent_to_none(backend, prefix):
    """
    The zeroed default state is what the LSTMs start from implicitly, so
    threading it explicitly must not change the encoder's output.
    """
    torch.manual_seed(42)

    conf, model = make_model(prefix=prefix, backend=backend)

    pfx, feats = make_feats(B=8, T=16, P=2, conf=conf)
    pfx = pfx if prefix else None

    implicit, _ = model.encode(state=None, prefix=pfx, feats=feats)
    explicit, _ = model.encode(
        state=model.default_enc_state_for(feats), prefix=pfx, feats=feats
    )

    assert torch.equal(implicit.data, explicit.data)
    assert torch.equal(implicit.lens, explicit.lens)
