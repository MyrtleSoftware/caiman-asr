import pytest
import torch
from apex.contrib.transducer import TransducerJoint
from apex.contrib.transducer import TransducerLoss as ApexLoss
from rnnt_ext.transducer.loss import TransducerLoss, TransducerLossFunc

if not torch.cuda.is_available():
    # Before jit import to avoid jit compilation on when we won't use it.
    pytest.skip("Cuda not available so can't run these test", allow_module_level=True)


def mock_data(
    batch_size,
    time_dim,
    dtype=torch.float32,
    real_vocab_size=9,  # i.e. chars in the alphabet
    max_decode_length=9,  # i.e longest char sequence
):
    VOCAB_SIZE = real_vocab_size + 1  # append blank to vocab

    label_length = max_decode_length - 1
    blank_offset = real_vocab_size

    # Each f, g, elem is projected to real_vocab_size (H) then extended to VOCAB_SIZE
    f = torch.rand(batch_size, time_dim, VOCAB_SIZE, dtype=dtype, requires_grad=True)

    g = torch.rand(
        batch_size, max_decode_length, VOCAB_SIZE, dtype=dtype, requires_grad=True
    )

    y = torch.randint(
        0, blank_offset - 1, (batch_size, label_length), dtype=torch.int32
    )

    f_lens = torch.randint(1, time_dim + 1, size=(batch_size,), dtype=torch.int32)
    f_lens[0] = time_dim
    y_lens = torch.randint(0, label_length + 1, size=(batch_size,), dtype=torch.int32)
    y_lens[-1] = label_length
    g_lens = y_lens + 1

    f, f_lens, g, g_lens, y, y_lens = (
        x.cuda() for x in [f, f_lens, g, g_lens, y, y_lens]
    )

    return blank_offset, f, f_lens, g, g_lens, y, y_lens


def gen_hidden(pack, f, g, f_lens, g_lens):
    if pack:
        batch_offset = torch.cumsum(f_lens * g_lens, dim=0)

        packed_batch = batch_offset[-1].item()

        h = TransducerJoint(pack_output=True)(
            f, g, f_lens, g_lens, batch_offset=batch_offset, packed_batch=packed_batch
        )

        return h, batch_offset
    else:
        h = TransducerJoint(pack_output=False)(f, g, f_lens, g_lens)

        return h, None


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("time_dim", [1, 2, 7])
@pytest.mark.parametrize("delay_penalty", [0.0, 0.05])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("eos_penalty", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("eos_idx", [None, 1])
@pytest.mark.parametrize("star_idx", [None, 2])
def test_pack_no_pack_equivalent(
    batch_size, time_dim, delay_penalty, dtype, eos_penalty, eos_idx, star_idx
):
    """
    In this test we assume that there is no joint_net so the vocab size is
    the same as the joint hidden dimension.
    """
    blank_offset, f, f_lens, g, g_lens, y, y_lens = mock_data(
        batch_size, time_dim, dtype
    )

    h_pack, batch_offset = gen_hidden(True, f, g, f_lens, g_lens)

    h_no_pack, _ = gen_hidden(False, f, g, f_lens, g_lens)

    max_f_len = torch.max(f_lens).item()

    loss_no_pack = TransducerLoss(
        packed_input=False,
    )(
        h_no_pack,
        y,
        f_lens,
        y_lens,
        blank_offset,
        eos_idx=eos_idx,
        star_idx=star_idx,
        batch_offset=batch_offset,
        max_f_len=max_f_len,
        delay_penalty=delay_penalty,
        eos_penalty=eos_penalty,
    )

    loss_pack = TransducerLoss(
        packed_input=True,
    )(
        h_pack,
        y,
        f_lens,
        y_lens,
        blank_offset,
        batch_offset=batch_offset,
        max_f_len=max_f_len,
        delay_penalty=delay_penalty,
        eos_idx=eos_idx,
        star_idx=star_idx,
        eos_penalty=eos_penalty,
    )

    assert len(h_no_pack.shape) == 4
    assert len(h_pack.shape) == 2

    assert torch.allclose(loss_no_pack, loss_pack), f"{loss_no_pack=} != {loss_pack=}"


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("time_dim", [1, 2, 7])
@pytest.mark.parametrize("pack", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_match_apex(batch_size, time_dim, pack, dtype):
    #

    bl_off, f, f_lens, g, g_lens, y, y_lens = mock_data(batch_size, time_dim, dtype)

    h, batch_offset = gen_hidden(pack, f, g, f_lens, g_lens)

    max_f_len = torch.max(f_lens).item()

    h0 = h.detach().clone().to(torch.float64)
    h1 = h.detach().clone()
    h2 = h.detach().clone()

    h0.requires_grad = True
    h1.requires_grad = True
    h2.requires_grad = True

    with torch.autocast("cuda"):
        loss = TransducerLoss(packed_input=pack)(
            h1,
            y,
            f_lens,
            y_lens,
            bl_off,
            batch_offset=batch_offset,
            max_f_len=max_f_len,
            delay_penalty=0.0,
        )

        apex_loss = ApexLoss(packed_input=pack, opt=0)(
            h2,
            y,
            f_lens,
            y_lens,
            bl_off,
            batch_offset=batch_offset,
            max_f_len=max_f_len,
        )

        assert apex_loss.dtype == loss.dtype

    ground = ApexLoss(packed_input=pack, opt=0)(
        h0,
        y,
        f_lens,
        y_lens,
        bl_off,
        batch_offset=batch_offset,
        max_f_len=max_f_len,
    )

    loss.mean().backward()
    apex_loss.mean().backward()
    ground.mean().backward()

    def mixed_check(a, b, ref):
        ap = a.to(ref.dtype)
        bp = b.to(ref.dtype)

        if torch.allclose(ap, bp):
            return True

        if torch.allclose(ap, ref):
            return True

        ar = torch.sum((ap - ref) ** 2)
        br = torch.sum((bp - ref) ** 2)

        print(f"{ar=}")
        print(f"{br=}")

        return ar < br or ar < 1e-6

    assert mixed_check(loss, apex_loss, ground)
    assert mixed_check(h1.grad, h2.grad, h0.grad)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("time_dim", [4, 7])
@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("delay_penalty", [0.0, 0.05, 0.99, 1.0, 2])
@pytest.mark.parametrize("eos_penalty", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("eos_idx", [None, 1])
@pytest.mark.parametrize("star_idx", [None, 2])
@pytest.mark.parametrize("star_penalty", [0.0, 0.1, 0.5])
def test_grad(
    batch_size,
    time_dim,
    pack,
    delay_penalty,
    eos_penalty,
    eos_idx,
    star_idx,
    star_penalty,
):
    blank_offset, f, f_lens, g, g_lens, y, y_lens = mock_data(
        batch_size, time_dim, dtype=torch.float64
    )

    if eos_idx is not None:
        for i in range(batch_size):
            y[i, y_lens[i].item() - 1] = eos_idx

    if star_idx is not None:
        while star_idx not in y:
            # Set some fraction of the labels to star_idx
            mask = torch.rand_like(y, dtype=torch.float32) < 0.1
            y = torch.where(mask, star_idx, y)

    h, batch_offset = gen_hidden(pack, f, g, f_lens, g_lens)

    def stub(x):
        return TransducerLossFunc.apply(
            x,
            y,
            f_lens,
            y_lens,
            batch_offset if pack else torch.empty(0),
            delay_penalty,
            torch.max(f_lens).item(),
            blank_offset,
            eos_penalty,
            eos_idx,
            star_penalty,
            star_idx,
            None,
            pack,
        )

    assert torch.autograd.gradcheck(stub, [h])
