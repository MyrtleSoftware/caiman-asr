from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from beartype.typing import Optional

from caiman_asr_train.core.batches import BatchFirst, TimeFirst
from caiman_asr_train.core.model import Hidden, StateEncoder, StatePred
from caiman_asr_train.lm.kenlm_ngram import NgramInfo
from caiman_asr_train.rnnt.beam import RNNTBeamDecoder
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.setup.val import ValSetup
from caiman_asr_train.utils.responses import split_batched_finals


@dataclass
class MockConfig:
    prefix_token_map = None
    pred_rnn_layers = 1
    pred_n_hid = 1


class MockModel(RNNT):
    """
    Patched model s.t. only the joint method returns useful values.

    Sufficient to test the decoders.
    """

    def __init__(self, vocab_size):
        self.joint_counter = 0
        self.blank_idx = 0
        self.vocab_size = vocab_size
        self.conditional_encoder = False
        self.config = MockConfig()

        # Bypass RNNT init
        torch.nn.Module.__init__(self)

        # Just for deduction by sos_for
        self.joint_enc = torch.nn.Linear(1, 1)

    def encode(
        self,
        *,
        state: Optional[StateEncoder],
        prefix: Optional[TimeFirst],
        feats: Optional[TimeFirst],
    ) -> tuple[TimeFirst, Optional[StateEncoder]]:
        assert feats is not None, "feats should be provided for this test"
        return feats, None

    def predict(
        self,
        *,
        state: StatePred,
        prefix: Optional[TimeFirst],
        tokens: Optional[TimeFirst],
    ) -> tuple[TimeFirst, Optional[StatePred]]:

        lens = torch.tensor([2])  # U + 1 i.e. add one to length

        x = torch.zeros(2, 1, 1)  # [T, B, H]
        s = torch.zeros(1, 1, 1)  # [L/T=1, B, H]

        return (
            TimeFirst(x, lens),
            # State will be ignored so just match shape
            StatePred(embed=s, state=Hidden(s, s)),
        )

    def joint(self, f: BatchFirst, g: BatchFirst):
        """
        On odd values of self.joint_counter return logits indicating blank index.

        Otherwise return logits indicating tokens with increasing idx.
        """
        large_neg, large_prob = -100, 10
        logits = torch.ones((1, self.vocab_size + 1)) * large_neg

        if self.joint_counter % 2 == 1:
            idx = self.blank_idx
        else:
            idx = 1 + (self.joint_counter // 2) % self.vocab_size

        logits[0, idx] = large_prob
        self.joint_counter += 1

        logits = logits.view(1, -1)  # [B*T*U, K]
        print(f"joint_counter={self.joint_counter}, idx={idx}, logits={logits.shape}")
        return (logits, 0)


@pytest.fixture
def args():
    return Namespace(
        max_symbol_per_sample=None,
        beam_width=4,
        temperature=1.5,
        sr_segment=0.0,
        max_inputs_per_batch=int(1e7),
        fuzzy_topk_logits=False,
        beam_decoder_procs_per_gpu=1,
        beam_no_partials=False,
        eos_is_terminal=False,
        eos_vad_threshold=float("inf"),
        beam_final_emission_thresh=float("inf"),
        eos_decoding="none",
        keywords_path=None,
        boost_keywords=False,
        keyword_boost_c0=0.3,
        keyword_boost_beta=0.9,
        keyword_boost_context_score=1.0,
        pnc_prefix=None,
        val_lang_prefix=None,
        top_k=5,
    )


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    test_data_dir = Path("/workspace/training/tests/test_data")
    assert test_data_dir.is_dir(), f"{test_data_dir=} doesn't exist"
    return test_data_dir


@pytest.fixture(scope="session")
def ngram_path(test_data_dir) -> str:
    return str(test_data_dir / "ngram/ngram.binary")


@pytest.mark.parametrize(
    """decoder_type, expected_tokens, expected_timesteps,
    expected_probs, prune_thresholds, use_ngram""",
    [
        ("greedy", [1, 2, 3, 4], [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0], None, False),
        ("beam", [2, 3, 4], [1, 2, 3], [1.0, 1.0, 1.0], (0.4, 1.5), False),
        ("beam", [5, 2, 3], [1, 2, 3], [1.0, 1.0, 1.0], None, False),
        ("beam", [2, 3, 4], [1, 2, 3], [1.0, 1.0, 1.0], (0.4, 1.5), True),
        ("beam", [5, 2, 5], [1, 2, 3], [1.0, 1.0, 1.0], None, True),
    ],
)
def test_decoders(
    args,
    decoder_type,
    expected_tokens,
    expected_timesteps,
    expected_probs,
    prune_thresholds,
    use_ngram,
    tokenizer,
    ngram_path,
):
    """
    Check results of greedy and beam decoders using MockModel.

    NOTE: The expected tokens/timesteps in this test aren't important in of themselves.
    This is particularly true for the beam decoder since altering the order in which the
    _joint_step method is called for each hypothesis will alter the result. The point of
    this test is to ensure that the internals of the decoders are working correctly and
    to facilitate refactoring when the implementation shouldn't change. If this test
    fails after a valid change to the decoder you will need to update the expected values.
    """
    time_dim = 4
    model_dim = 1
    vocab_size = 6
    if prune_thresholds is None:
        args.beam_prune_score_thresh = -1
        args.beam_prune_topk_thresh = -1
    else:
        args.beam_prune_score_thresh = prune_thresholds[0]
        args.beam_prune_topk_thresh = prune_thresholds[1]

    args.decoder = decoder_type

    model = MockModel(vocab_size)

    ngram_info = None
    if use_ngram:
        ngram_info = NgramInfo(ngram_path, 0.1)

    decoder = ValSetup().build_decoder(
        model,
        args,
        0,
        None,
        tokenizer,
        ngram_info=ngram_info,
        world_size=1,
        user_tokens=[],
        user_symbols_dict={},
    )

    feats = torch.randn((time_dim, 1, model_dim))
    feat_lens = torch.tensor([time_dim])

    tks, timesteps, _, probs = split_batched_finals(
        decoder.decode(TimeFirst(feats, feat_lens))
    )

    assert tks == [expected_tokens]
    assert timesteps == [expected_timesteps]
    assert probs == [expected_probs]


@pytest.mark.parametrize(
    "prefix_token_ids",
    [
        [1],
        [1, 2],
        [1, 2, 3],
    ],
)
def test_beam_prefix(prefix_token_ids, test_data_dir):
    """
    Test that prefix tokens are fed to the predictor exactly once,
    regardless of prefix length.

    The predictor is initialised by a single predict(prefix=..., tokens=None) call that
    advances the LSTM through all prefix tokens. After that, the initial hypothesis has
    y_last=-1 (SOS), so the first beam expansion call does not feed any token to the
    predictor (tokens=None). All subsequent expansion calls advance the predictor with
    decoded tokens via predict(tokens=[[token]]).
    """

    time_dim = 4
    model_dim = 1
    vocab_size = 6
    predict_calls = []

    class PredictTrackingMockModel(MockModel):
        def predict(self, *, state, prefix, tokens):
            predict_calls.append((prefix is not None, tokens))
            return super().predict(state=state, prefix=prefix, tokens=tokens)

    model = PredictTrackingMockModel(vocab_size)

    decoder = RNNTBeamDecoder(
        model=model,
        blank_idx=0,
        eos_strategy=None,
        sentpiece_model=str(test_data_dir / "librispeech29.model"),
        beam_width=1,
        prefix_token_ids=prefix_token_ids,
    )

    feats = torch.randn((time_dim, 1, model_dim))
    feat_lens = torch.tensor([time_dim])
    decoder.decode(TimeFirst(feats, feat_lens))

    # Filter initialization call (has prefix!=None)
    # All expansion calls should have prefix=None
    beam_expansion_tokens = [
        tokens for has_prefix, tokens in predict_calls if not has_prefix
    ]

    # sanity check beam successful
    assert len(beam_expansion_tokens) > 0, "No expansion calls were made"
    # incorrectly processing the last prefix token twice
    assert beam_expansion_tokens[0] is None, (
        f"First beam expansion called predict with tokens={beam_expansion_tokens[0]}, "
        f"expected None. This indicates init_hyp was called with prefix_token_ids, "
        f"causing y_last={prefix_token_ids[-1]} instead of -1."
    )
