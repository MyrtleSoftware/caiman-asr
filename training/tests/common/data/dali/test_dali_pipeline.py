import copy
from argparse import Namespace

import pytest
import torch

from rnnt_train.common.data.build_dataloader import build_dali_loader
from rnnt_train.common.data.dali.data_loader import DaliDataLoader
from rnnt_train.common.data.webdataset import LengthUnknownError
from rnnt_train.rnnt import config


@pytest.fixture(scope="session")
def dataload_args(test_data_dir) -> Namespace:
    manifest_fp = str(test_data_dir / "peoples-speech-short.json")
    return Namespace(
        grad_accumulation_batches=1,
        val_manifests=[manifest_fp],
        train_manifests=[manifest_fp],
        dump_mel_stats=None,
        local_rank=0,
        num_buckets=6,
        dataset_dir=str(test_data_dir),
        dali_device="cpu",
        train_tar_files=None,
        val_tar_files=None,
        read_from_tar=False,
    )


@pytest.fixture(scope="session")
def dataload_args_webdataset(dataload_args) -> Namespace:
    dataload_args = copy.deepcopy(dataload_args)
    # use all tar files in test_data_dir
    dataload_args.train_tar_files = dataload_args.val_tar_files = ["*.tar"]
    dataload_args.read_from_tar = True
    return dataload_args


def build_dataloader_util(
    dataload_args,
    pipeline_type,
    batch_size,
    config_fp,
    tokenizer,
    deterministic_ex_noise: bool = False,
) -> DaliDataLoader:
    """
    Build dali dataloader helper function for testing.
    """
    cfg = config.load(config_fp)
    dataset_kw, features_kw, _, _ = config.input(cfg, pipeline_type)

    if deterministic_ex_noise:
        # make dataloader deterministic except for noise augmentation
        features_kw["dither"] = 0.0
        dataset_kw["speed_perturbation"] = None

    return build_dali_loader(
        dataload_args,
        pipeline_type,
        batch_size=batch_size,
        dataset_kw=dataset_kw,
        features_kw=features_kw,
        tokenizer=tokenizer,
        cpu=True,
        no_logging=True,
    )


@pytest.fixture(scope="session")
def saved_tensor_no_noise(test_data_dir) -> torch.Tensor:
    tensor_fp = test_data_dir / "audio_tensor_batch.pt"
    return torch.load(tensor_fp)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("pipeline_type", ["val", "train"])
@pytest.mark.parametrize("webdataset", [False, True])
def test_dali_dataloader_build(
    pipeline_type,
    batch_size,
    dataload_args,
    dataload_args_webdataset,
    config_fp,
    webdataset,
    tokenizer,
):
    if webdataset:
        args = dataload_args_webdataset
    else:
        args = dataload_args
    dataloader = build_dataloader_util(
        args, pipeline_type, batch_size, config_fp, tokenizer
    )

    dataset_length = 2  # to match the test data

    if not webdataset:
        assert len(dataloader) == dataset_length // batch_size
    else:
        # in webdataset case, the length of the dataset is unknown
        with pytest.raises(LengthUnknownError):
            len(dataloader)
    samples_seen = 0
    for batch in dataloader:
        audio, audio_lens, txt, txt_lens = batch
        B, H, T = audio.shape
        B2, U = txt.shape
        assert B == B2 == len(audio_lens) == len(txt_lens) == batch_size
        assert T == audio_lens.max()
        assert U == txt_lens.max()
        samples_seen += B
        if samples_seen > 10:
            # i.e. we saw an infinite loop here when cycle=True
            raise RuntimeError(
                f"There are only x2 samples in the dataset but {samples_seen=}. "
                "Are we cycling infinitely here?"
            )

    assert samples_seen == dataset_length

    # there was an issue where webdataset was not resetting the dataset
    # between epochs, so check that the dataset is reset:
    if not webdataset:
        return
    samples_seen = 0
    for batch in dataloader:
        audio, _, _, _ = batch
        B, H, T = audio.shape
        samples_seen += B

    assert (
        samples_seen == dataset_length
    ), f"{samples_seen} != {dataset_length}. Reset failed"


def test_shuffle_train_webdataset(
    dataload_args_webdataset,
    config_fp,
    tokenizer,
    batch_size=2,
    n_epochs=5,
):
    dataloader = build_dataloader_util(
        dataload_args_webdataset, "train", batch_size, config_fp, tokenizer
    )
    # after n_epochs we expect to see samples appear in different order since we are
    # shuffling the dataset
    # TODO: set seed to ensure this isn't flaky
    prev_txt = None
    for _ in range(n_epochs):
        for _, _, txt, _ in dataloader:
            if prev_txt is None or torch.allclose(prev_txt, txt):
                prev_txt = txt
                continue
            # we have shuffled
            break


def test_no_shuffle_val_webdataset(
    dataload_args_webdataset,
    config_fp,
    tokenizer,
    batch_size=2,
    n_epochs=5,
):
    dataloader = build_dataloader_util(
        dataload_args_webdataset, "val", batch_size, config_fp, tokenizer
    )
    # for n_epochs we expect to see samples appear in same order since no shuffle
    prev_txt = None
    for _ in range(n_epochs):
        for _, _, txt, _ in dataloader:
            if prev_txt is None or torch.allclose(prev_txt, txt):
                prev_txt = txt
                continue
            raise RuntimeError("We have shuffled when we shouldn't have")


def test_dali_equivalence(dataload_args, saved_tensor_no_noise, config_fp, tokenizer):
    """
    Test dali equivalence to tensors saved to file.

    This is so that we are alerted to changes in the preprocessing code as it may
    imply we need to change the Rust preprocessing code in the ASR server to match.
    """
    dali_dataloder_deterministic = build_dataloader_util(
        dataload_args, "val", 2, config_fp, tokenizer, True
    )
    for audio, _, _, _ in dali_dataloder_deterministic:
        assert torch.allclose(audio, saved_tensor_no_noise, atol=2e-04)
