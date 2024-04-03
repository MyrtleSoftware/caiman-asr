#!/usr/bin/env python3
import pytest
import torch

from caiman_asr_train.rnnt.unbatch_encoder import encode_lower_batch_size


@pytest.mark.parametrize("max_inputs_per_batch", [10000, 30000, 50000, 100000, 1000000])
def test_apply_encoder(model_factory, max_inputs_per_batch):
    model = model_factory().cuda()
    batch_size = 19
    seq_len = 100
    feat_dim = 240
    feats = torch.randn(seq_len, batch_size, feat_dim).cuda()
    feat_lens = torch.randint(1, seq_len, (batch_size,)).cuda()
    encs1, enc_lens1 = encode_lower_batch_size(
        model, feats, feat_lens, max_inputs_per_batch
    )
    encs2, enc_lens2, _ = model.encode(feats, feat_lens)
    assert torch.allclose(encs1, encs2, atol=1e-5)
    assert torch.allclose(enc_lens1, enc_lens2)
