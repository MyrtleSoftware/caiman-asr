from argparse import Namespace

import pytest
import torch

from caiman_asr_train.lm.kenlm_ngram import NgramInfo
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.setup.val import ValSetup


class MockModel(RNNT):
    """
    Patched model s.t. only the joint method returns useful values.

    Sufficient to test the decoders.
    """

    def __init__(self, vocab_size):
        self.joint_counter = 0
        self.blank_idx = 0
        self.vocab_size = vocab_size
        torch.nn.Module.__init__(self)

    def encode(self, x, x_lens, enc_state=None):
        return x.transpose(0, 1), x_lens, None

    def predict(self, y, pred_state=None, add_sos=True, special_sos=None):
        # return g, hid, all_hid
        return None, None, None

    def joint(self, f, g, f_len=None, g_len=None, batch_offset=None):
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
        return logits.view(1, 1, 1, -1)


@pytest.fixture
def args():
    return Namespace(
        max_symbol_per_sample=None,
        beam_width=4,
        temperature=1.5,
        sr_segment=0.0,
        max_inputs_per_batch=int(1e7),
        fuzzy_topk_logits=False,
    )


@pytest.mark.parametrize(
    """decoder_type, expected_tokens, expected_timesteps,
    expected_probs, prune_thresholds, use_ngram""",
    [
        ("greedy", [1, 2, 3, 4], [0, 1, 2, 3], [[1.0, 1.0, 1.0, 1.0]], None, False),
        ("beam", [2, 3, 4], [1, 2, 3], [], (0.4, 1.5), False),
        ("beam", [5, 2, 3], [1, 2, 3], [], None, False),
        ("beam", [2, 3, 4], [1, 2, 3], [], (0.4, 1.5), True),
        ("beam", [5, 2, 5], [1, 2, 3], [], None, True),
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
        model, args, 0, tokenizer, lm_info=None, ngram_info=ngram_info
    )

    feats = torch.randn((time_dim, 1, model_dim))
    feat_lens = torch.tensor([time_dim])

    tks, timesteps, probs = decoder.decode(feats, feat_lens)

    assert tks == [expected_tokens]
    assert timesteps == ([expected_timesteps] if expected_timesteps else [])
    assert probs == expected_probs
