import random

import torch

from caiman_asr_train.data.generate_mel_stats import (
    _chen_tree_reduce,
    _drop_zero_sentinels,
    _merge_welford_across_ranks,
    generate_stats,
)
from caiman_asr_train.utils.seed import set_seed


def test_drop_zeros():
    # Set up data
    max_nfeats = 1000
    meldim = 80

    n1 = torch.zeros(max_nfeats, meldim, dtype=torch.int64)
    m1 = torch.rand(max_nfeats, meldim, dtype=torch.float64)
    m2 = torch.rand(max_nfeats, meldim, dtype=torch.float64)

    for i in range(0, max_nfeats):
        n1[i] = random.randint(0, 2)

    n1_bak = n1.clone()
    m1_bak = m1.clone()
    m2_bak = m2.clone()

    # Drop zeros
    n1, m1, m2 = _drop_zero_sentinels(n1, m1, m2)

    assert not torch.any(n1 == 0)

    # Check non-zero rows are preserved
    acc = 0
    for i in range(0, max_nfeats):
        if n1_bak[i, 0] != 0:
            assert torch.allclose(n1[acc], n1_bak[i])
            assert torch.allclose(m1[acc], m1_bak[i])
            assert torch.allclose(m2[acc], m2_bak[i])
            acc += 1


def test_generate_mel_stats(dataload_args, tmpdir):
    dataload_args.output_dir = tmpdir
    dataload_args.mel_duration_sample = float("inf")
    dataload_args.loader_train_device = "cpu"
    generate_stats(dataload_args)

    # assert that the output files are created
    assert (tmpdir / "melmeans.pt").exists()
    assert (tmpdir / "melvars.pt").exists()
    assert (tmpdir / "meln.pt").exists()


def load(x):
    return torch.load(x, weights_only=False)


def test_gen_stats_no_augmentation(dataload_args, tmpdir):
    """
    Even when seed is different.

    This is to ensure that all data augmentation is off during stats generation.
    """
    dataload_args.output_dir = tmpdir
    dataload_args.mel_duration_sample = float("inf")
    dataload_args.loader_train_device = "cpu"
    dataload_args.seed = 0

    set_seed(dataload_args.seed)
    generate_stats(dataload_args)

    melmeans = load(str(tmpdir / "melmeans.pt"))
    melvars = load(str(tmpdir / "melvars.pt"))
    meln = load(str(tmpdir / "meln.pt"))

    # generate stats again
    dataload_args.seed = 2
    set_seed(dataload_args.seed)
    generate_stats(dataload_args)

    melmeans2 = load(str(tmpdir / "melmeans.pt"))
    melvars2 = load(str(tmpdir / "melvars.pt"))
    meln2 = load(str(tmpdir / "meln.pt"))

    assert torch.allclose(melmeans, melmeans2)
    assert torch.allclose(melvars, melvars2)
    assert torch.allclose(meln, meln2)


def test_stats_dont_change(dataload_args, tmpdir, test_data_dir):
    """
    Checks the stats are the same as ones stored on file.
    """
    dataload_args.output_dir = tmpdir
    dataload_args.mel_duration_sample = float("inf")
    dataload_args.loader_train_device = "cpu"
    generate_stats(dataload_args)

    melmeans = load(str(tmpdir / "melmeans.pt"))
    melvars = load(str(tmpdir / "melvars.pt"))
    meln = load(str(tmpdir / "meln.pt"))

    ref_dir = test_data_dir / "jsonl_stats"

    tol = 1e-6
    assert torch.allclose(meln, load(ref_dir / "meln.pt"))
    assert torch.allclose(melmeans, load(ref_dir / "melmeans.pt"), atol=tol)
    assert torch.allclose(melvars, load(ref_dir / "melvars.pt"), atol=tol)


def test_merge_welford_across_ranks_mocked_dist(monkeypatch):
    """Verify distributed Welford reduction across multiple ranks."""
    n1 = torch.tensor([[2, 2]], dtype=torch.int64)
    m1 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    m2 = torch.tensor([[0.5, 0.25]], dtype=torch.float64)

    n_ranks = [
        torch.tensor([[2, 2]], dtype=torch.int64),
        torch.tensor([[3, 3]], dtype=torch.int64),
        torch.tensor([[4, 4]], dtype=torch.int64),
    ]
    m1_ranks = [
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        torch.tensor([[4.0, 1.0]], dtype=torch.float64),
        torch.tensor([[2.0, 3.0]], dtype=torch.float64),
    ]
    m2_ranks = [
        torch.tensor([[0.5, 0.25]], dtype=torch.float64),
        torch.tensor([[1.5, 1.0]], dtype=torch.float64),
        torch.tensor([[2.0, 0.5]], dtype=torch.float64),
    ]

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)

    gather_payloads = [n_ranks, m1_ranks, m2_ranks]
    gather_idx = {"i": 0}

    def fake_all_gather(output_list, _input_tensor):
        payload = gather_payloads[gather_idx["i"]]
        for out, src in zip(output_list, payload):
            out.copy_(src)
        gather_idx["i"] += 1

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    got_n1, got_m1, got_m2 = _merge_welford_across_ranks(n1, m1, m2)

    cat_n1 = torch.cat(n_ranks, dim=0)
    cat_m1 = torch.cat(m1_ranks, dim=0)
    cat_m2 = torch.cat(m2_ranks, dim=0)
    nonzero = cat_n1[:, 0] != 0

    assert torch.all(nonzero)

    exp_n1, exp_m1, exp_m2 = _chen_tree_reduce(
        cat_n1[nonzero], cat_m1[nonzero], cat_m2[nonzero]
    )

    assert torch.equal(got_n1, exp_n1)
    assert torch.allclose(got_m1, exp_m1)
    assert torch.allclose(got_m2, exp_m2)
