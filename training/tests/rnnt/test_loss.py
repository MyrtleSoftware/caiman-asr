import pytest
import torch
from apex.contrib.transducer import TransducerJoint

from rnnt_train.rnnt.loss import apexTransducerLoss


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("time_dim", [2, 17])
def test_pack_no_pack_equivalent(batch_size, time_dim):
    """
    In this test we assume that there is no joint_net so the vocab size is the same as
    the joint hidden dimension.
    """
    torch.manual_seed(42)
    VOCAB_SIZE = 10  # including BLANK
    DECODED_LENGTH = 7
    label_length = DECODED_LENGTH - 1
    blank_idx = VOCAB_SIZE - 1
    f = torch.rand(batch_size, time_dim, VOCAB_SIZE)
    g = torch.rand(batch_size, DECODED_LENGTH, VOCAB_SIZE)
    y = torch.randint(0, blank_idx, (batch_size, label_length), dtype=int)
    f_lens = torch.randint(1, time_dim, size=(batch_size,), dtype=torch.int)
    f_lens[0] = time_dim
    y_lens = torch.randint(2, DECODED_LENGTH, size=(batch_size,), dtype=torch.int)
    y_lens[-1] = label_length
    g_lens = y_lens + 1

    if not torch.cuda.is_available():
        pytest.skip("Cuda not available so can't run this test")
    f, f_lens, g, g_lens, y, y_lens = (
        f.cuda(),
        f_lens.cuda(),
        g.cuda(),
        g_lens.cuda(),
        y.cuda(),
        y_lens.cuda(),
    )

    joint_no_pack = TransducerJoint(pack_output=False)
    loss_fn_no_pack = apexTransducerLoss(blank_idx=blank_idx, packed_input=False)
    h_no_pack = joint_no_pack(f, g, f_lens, g_lens)
    loss_no_pack = loss_fn_no_pack(h_no_pack, f_lens, y, y_lens, None, None)

    batch_offset = torch.cumsum(f_lens * g_lens, dim=0)
    max_f_len = max(f_lens)
    packed_batch = packed_batch = batch_offset[-1].item()
    joint_pack = TransducerJoint(pack_output=True)
    loss_fn_pack = apexTransducerLoss(blank_idx=blank_idx, packed_input=True)
    h_pack = joint_pack(
        f, g, f_lens, g_lens, batch_offset=batch_offset, packed_batch=packed_batch
    )
    loss_pack = loss_fn_pack(h_pack, f_lens, y, y_lens, batch_offset, max_f_len)

    assert len(h_no_pack.shape) == 4
    assert len(h_pack.shape) == 2

    assert torch.allclose(
        loss_no_pack, loss_pack
    ), f"loss_no_pack={loss_no_pack} != loss_pack={loss_pack}"
