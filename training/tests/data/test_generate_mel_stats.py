from argparse import Namespace

import pytest
import torch

from caiman_asr_train.args.norm_stats_generation import update_args_stats_generation
from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.generate_mel_stats import generate_stats
from caiman_asr_train.unittesting.dataload_args import update_dataload_args
from caiman_asr_train.utils.seed import set_seed


def update_args(args: Namespace, format: DataSource, batch_size: int = 1) -> Namespace:
    args = update_dataload_args(args, format)
    args = update_args_stats_generation(args, batch_size=batch_size)
    return args


@pytest.mark.parametrize("format", DataSource)
def test_generate_mel_stats(dataload_args, format, tmpdir):
    args = update_args(dataload_args, format, batch_size=1)

    args.output_dir = tmpdir
    generate_stats(args)

    # assert that the output files are created
    assert (tmpdir / "melmeans.pt").exists()
    assert (tmpdir / "melvars.pt").exists()
    assert (tmpdir / "meln.pt").exists()


@pytest.mark.parametrize("format", DataSource)
def test_gen_stats_no_augmentation(dataload_args, tmpdir, format):
    """
    Even when seed is different.

    This is to ensure that all data augmentation is off during stats generation.
    """
    args = update_args(dataload_args, format, batch_size=1)
    args.output_dir = tmpdir

    args.seed = 0
    set_seed(args.seed)
    generate_stats(args)

    melmeans = torch.load(str(tmpdir / "melmeans.pt"))
    melvars = torch.load(str(tmpdir / "melvars.pt"))
    meln = torch.load(str(tmpdir / "meln.pt"))

    # generate stats again
    args.seed = 2
    set_seed(args.seed)
    generate_stats(args)

    melmeans2 = torch.load(str(tmpdir / "melmeans.pt"))
    melvars2 = torch.load(str(tmpdir / "melvars.pt"))
    meln2 = torch.load(str(tmpdir / "meln.pt"))

    assert torch.allclose(melmeans, melmeans2)
    assert torch.allclose(melvars, melvars2)
    assert torch.allclose(meln, meln2)


REASON = """HF training is not supported. When training with format=HF,
            the training code uses json files for training and HF
            for validation. Hence there are no HF reference stats"""
DATASOURCES = [
    (
        pytest.param(x, marks=pytest.mark.xfail(reason=REASON))
        if x == DataSource.HUGGINGFACE
        else x
    )
    for x in DataSource
]


@pytest.mark.parametrize("format", DATASOURCES)
def test_stats_dont_change(format, dataload_args, tmpdir, test_data_dir, batch_size=1):
    """
    Checks the stats are the same as ones stored on file.
    """
    args = update_args(dataload_args, format, batch_size=1)
    args.output_dir = tmpdir
    generate_stats(args)

    melmeans = torch.load(str(tmpdir / "melmeans.pt"))
    melvars = torch.load(str(tmpdir / "melvars.pt"))
    meln = torch.load(str(tmpdir / "meln.pt"))

    ref_dir = test_data_dir / f"{format.name.lower()}_stats"

    tol = 1e-9
    assert torch.allclose(meln, torch.load(ref_dir / "meln.pt"))
    assert torch.allclose(melmeans, torch.load(ref_dir / "melmeans.pt"), rtol=tol)
    assert torch.allclose(melvars, torch.load(ref_dir / "melvars.pt"), rtol=tol)
