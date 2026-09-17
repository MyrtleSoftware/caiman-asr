import pytest
import torch
from beartype import beartype
from beartype.typing import List, Optional

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.core.greedy import GreedyDecoder, GreedyResult
from caiman_asr_train.core.lstm import TorchConfig
from caiman_asr_train.core.model import RNNT as CoreRNNT
from caiman_asr_train.core.model import ConfigRNNT

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Comparisons against the batched decoder use the production "
    "custom-LSTM + apex-joint path, which needs CUDA",
)

IN_FEATS = 32
STACK_TIME_FACTOR = 2


# === CPU tests against the core (torch-backend) model


@beartype
def make_core_model(prefix: bool, seed: int) -> CoreRNNT:
    torch.manual_seed(seed)

    config = ConfigRNNT(
        n_classes=16,
        in_feats=IN_FEATS,
        enc_n_hid=32,
        enc_pre_rnn_layers=1,
        enc_post_rnn_layers=1,
        enc_stack_time_factor=STACK_TIME_FACTOR,
        enc_dropout=0.0,
        prefix_token_map={i: i for i in range(4)} if prefix else None,
        pred_n_hid=16,
        pred_rnn_layers=1,
        pred_dropout=0.0,
        # The backend hardware checkpoints are loaded with (see core.load_hw).
        backend=TorchConfig(return_all=True),
        weights_init_scale=None,
        forget_gate_bias=None,
        hidden_hidden_bias_scale=None,
        joint_n_hid=24,
        joint_dropout=0.0,
    )

    return CoreRNNT(config).eval()


@pytest.mark.parametrize(
    "with_map, prefix_token_ids",
    [
        (False, None),
        (False, [1, 2]),  # Prediction-network-only prefix
        (True, None),
        (True, [0, 2, 3]),  # Encoder + prediction-network prefix
    ],
)
def test_cpu_decode_basic(with_map, prefix_token_ids):
    model = make_core_model(prefix=with_map, seed=0)

    decoder = GreedyDecoder(model=model, prefix_token_ids=prefix_token_ids)

    T = 20
    torch.manual_seed(100)
    result = decoder.decode(torch.randn(T, IN_FEATS))

    assert isinstance(result, GreedyResult)
    assert len(result.tokens) == len(result.timestamps) == len(result.probs)

    blank = model.config.n_classes - 1
    T_enc = T // STACK_TIME_FACTOR

    assert all(0 <= tok < blank for tok in result.tokens)
    assert all(0 <= ts < T_enc for ts in result.timestamps)
    assert result.timestamps == sorted(result.timestamps)
    assert all(0 < p <= 1 for p in result.probs)


def test_cpu_max_symbol_per_sample_truncates():
    model = make_core_model(prefix=False, seed=1)

    torch.manual_seed(101)
    feats = torch.randn(24, IN_FEATS)

    full = GreedyDecoder(model=model).decode(feats)
    trunc = GreedyDecoder(model=model, max_symbol_per_sample=3).decode(feats)

    assert len(full.tokens) > 3, "Test is vacuous unless the full decode is longer"
    assert trunc.tokens == full.tokens[:3]
    assert trunc.timestamps == full.timestamps[:3]
    # Separate decodes are not bit-exact (CPU GEMMs vary with allocation
    # alignment), hence the tolerance.
    torch.testing.assert_close(trunc.probs, full.probs[:3], atol=1e-6, rtol=0)


def test_cpu_decode_encodings_matches_decode():
    prefix_token_ids = [1, 3]
    model = make_core_model(prefix=True, seed=2)

    decoder = GreedyDecoder(model=model, prefix_token_ids=prefix_token_ids)

    torch.manual_seed(102)
    feats = torch.randn(20, IN_FEATS)

    from_feats = decoder.decode(feats)

    prefix = TimeFirst(
        data=torch.tensor(prefix_token_ids).unsqueeze(1),
        lens=torch.tensor([len(prefix_token_ids)], dtype=torch.int32),
    )
    feats_tf = TimeFirst(feats.unsqueeze(1), torch.tensor([feats.shape[0]]))

    encs, _ = model.encode(state=None, feats=feats_tf, prefix=prefix)

    from_encs = decoder.decode_encodings(encs.data[:, 0])

    assert from_encs.tokens == from_feats.tokens
    assert from_encs.timestamps == from_feats.timestamps
    # Separate decodes are not bit-exact (CPU GEMMs vary with allocation
    # alignment), hence the tolerance.
    torch.testing.assert_close(from_encs.probs, from_feats.probs, atol=1e-6, rtol=0)


# === GPU comparisons against the source-of-truth batched greedy decoder


@beartype
def make_full_model(prefix: bool, n_classes: int, seed: int):
    """
    Build the production rnnt.model.RNNT (the batched decoder needs its packed
    apex joint) on the custom-LSTM backend, which threads predictor state.
    """
    from caiman_asr_train.rnnt.model import RNNT as FullRNNT

    torch.manual_seed(seed)

    model = FullRNNT(
        n_classes=n_classes,
        in_feats=IN_FEATS,
        enc_n_hid=64,
        enc_batch_norm=False,
        pred_batch_norm=False,
        enc_pre_rnn_layers=2,
        enc_post_rnn_layers=2,
        enc_stack_time_factor=STACK_TIME_FACTOR,
        enc_dropout=0.0,
        pred_dropout=0.0,
        joint_dropout=0.0,
        pred_n_hid=48,
        pred_rnn_layers=2,
        joint_n_hid=56,
        forget_gate_bias=1.0,
        custom_lstm=True,
        enc_rw_dropout=0.0,
        pred_rw_dropout=0.0,
        joint_apex_transducer="pack",
        conditional_encoder=prefix,
        prefix_token_map={i: i for i in range(4)} if prefix else None,
    )

    return model.cuda().eval()


@beartype
def make_batch(B: int, T: int, seed: int) -> TimeFirst:
    generator = torch.Generator().manual_seed(seed)

    data = torch.randn(T, B, IN_FEATS, generator=generator)

    # Variable lengths, multiples of the stack time factor.
    lens = (
        torch.randint(1, T // STACK_TIME_FACTOR + 1, (B,), generator=generator)
        * STACK_TIME_FACTOR
    )
    lens[0] = T

    return TimeFirst(data, lens).to("cuda")


@beartype
def assert_matches_batched(
    model,
    tokenizer,
    feats: TimeFirst,
    max_symbols_per_step: Optional[int] = 30,
    max_symbol_per_sample: Optional[int] = None,
    prefix_token_ids: Optional[List[int]] = None,
):
    """
    Decode each utterance with the batched decoder (at batch size 1) and with
    the core decoder, then compare.

    Batch size 1 keeps every tensor shape identical between the two decoders,
    so the LSTM streams stay bit-exact and argmax near-ties cannot flip. (At
    batch size > 1 greedy decodes are NOT bit-exact against single-utterance
    decodes -- batch-composition float noise -- so that comparison would be
    flaky by construction.) The joint float path still differs slightly (apex
    packed joint vs an unbatched add + joint_net), hence the probs tolerance.
    """
    from caiman_asr_train.rnnt.batched_greedy import RNNTBatchedGreedyDecoder
    from caiman_asr_train.utils.responses import split_batched_finals

    blank_idx = model.config.n_classes - 1

    batched = RNNTBatchedGreedyDecoder(
        model=model,
        blank_idx=blank_idx,
        eos_strategy=None,
        max_inputs_per_batch=int(1e7),
        tokenizer=tokenizer,
        max_symbols_per_step=max_symbols_per_step,
        max_symbol_per_sample=max_symbol_per_sample,
        prefix_token_ids=prefix_token_ids,
    )

    core = GreedyDecoder(
        model=model,
        blank_idx=blank_idx,
        max_symbols_per_step=max_symbols_per_step,
        max_symbol_per_sample=max_symbol_per_sample,
        prefix_token_ids=prefix_token_ids,
    )

    for i in range(feats.data.shape[1]):
        sub = feats.sub_batch(i, i + 1)

        tokens, timestamps, _, probs = split_batched_finals(batched.decode(sub))

        result = core.decode(sub.data[:, 0])

        assert result.tokens == tokens[0], f"Token mismatch for sample {i}"
        assert result.timestamps == timestamps[0], f"Timestamp mismatch, sample {i}"
        torch.testing.assert_close(
            result.probs,
            probs[0],
            atol=1e-5,
            rtol=0,
            msg=f"Prob mismatch for sample {i}",
        )


@requires_cuda
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "with_map, prefix_token_ids",
    [
        (False, None),
        (False, [1, 2]),  # Prediction-network-only prefix
        (True, [1]),
        (True, [0, 2, 3]),  # Encoder + prediction-network prefix
    ],
)
def test_matches_batched(tokenizer, seed, with_map, prefix_token_ids):
    model = make_full_model(with_map, tokenizer.num_labels + 1, seed)

    feats = make_batch(B=2, T=16, seed=seed)

    assert_matches_batched(model, tokenizer, feats, prefix_token_ids=prefix_token_ids)


@requires_cuda
@pytest.mark.parametrize(
    "max_symbols_per_step, max_symbol_per_sample",
    [
        (1, None),
        (2, 7),
        (30, 5),
        (None, 8),
    ],
)
def test_matches_batched_limits(tokenizer, max_symbols_per_step, max_symbol_per_sample):
    model = make_full_model(False, tokenizer.num_labels + 1, seed=0)

    feats = make_batch(B=2, T=12, seed=3)

    assert_matches_batched(
        model,
        tokenizer,
        feats,
        max_symbols_per_step=max_symbols_per_step,
        max_symbol_per_sample=max_symbol_per_sample,
    )


@requires_cuda
def test_matches_batched_limits_and_prefix(tokenizer):
    model = make_full_model(True, tokenizer.num_labels + 1, seed=4)

    feats = make_batch(B=2, T=16, seed=5)

    assert_matches_batched(
        model,
        tokenizer,
        feats,
        max_symbols_per_step=2,
        max_symbol_per_sample=9,
        prefix_token_ids=[0, 1],
    )
