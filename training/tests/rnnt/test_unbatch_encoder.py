#!/usr/bin/env python3
import pytest
import torch

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.rnnt.unbatch_encoder import encode_lower_batch_size


@pytest.mark.parametrize("max_inputs_per_batch", [10000, 30000, 50000, 100000, 1000000])
def test_apply_encoder(model_factory, max_inputs_per_batch):
    model = model_factory().cuda()
    batch_size = 19
    seq_len = 100
    feat_dim = 240
    feats = torch.randn(seq_len, batch_size, feat_dim).cuda()
    feat_lens = torch.randint(1, seq_len, (batch_size,)).cuda()

    # Make even
    feat_lens = (feat_lens // 2) * 2

    # The sub_batch ensures minimal-padding
    feats = TimeFirst(data=feats, lens=feat_lens).sub_batch(0, batch_size)

    encs1 = encode_lower_batch_size(
        model, feats, max_inputs_per_batch, prefix_token_ids=None
    )

    encs2, _ = model.encode(feats=feats, state=None, prefix=None)

    assert torch.allclose(encs1.lens, encs2.lens, atol=1e-4)

    for i in range(batch_size):
        # Only compare non-padded frames
        a = encs1.sub_batch(i, i + 1).data
        b = encs2.sub_batch(i, i + 1).data
        assert torch.allclose(a, b, atol=1e-4)
