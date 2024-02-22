import copy
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from beartype import beartype

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
        prob_background_noise=0.0,
        prob_babble_noise=0.0,
        train_tar_files=None,
        val_tar_files=None,
        read_from_tar=False,
        seed=1,
        turn_off_initial_padding=True,
        inspect_audio=False,
        prob_val_narrowband=0.0,
        prob_train_narrowband=0.0,
        output_dir=Path("/results"),
        n_utterances_only=None,
        noise_dataset=None,
        use_noise_audio_folder=False,
        noise_config=None,
        val_from_dir=False,
        val_audio_dir=None,
        val_txt_dir=None,
        dali_processes_per_cpu=1,
    )


@pytest.fixture(scope="session")
def dataload_args_webdataset(dataload_args) -> Namespace:
    dataload_args = copy.deepcopy(dataload_args)
    # use all tar files in test_data_dir
    dataload_args.train_tar_files = dataload_args.val_tar_files = ["webdataset-eg.tar"]
    dataload_args.read_from_tar = True
    return dataload_args


@pytest.fixture(scope="session")
def dataload_args_noise(dataload_args, test_data_dir) -> Namespace:
    """
    Update dataload_args to add noise
    """
    dataload_args = copy.deepcopy(dataload_args)
    dataload_args.prob_background_noise = 1.0
    dataload_args.noise_dataset = str(test_data_dir / "TestNoiseDataset")
    dataload_args.use_noise_audio_folder = True
    return dataload_args


@pytest.fixture(scope="session")
def dataload_args_noise_babble(dataload_args) -> Namespace:
    """Adds babble noise"""
    dataload_args = copy.deepcopy(dataload_args)
    dataload_args.prob_babble_noise = 1.0
    return dataload_args


@beartype
def build_dataloader_util(
    dataload_args,
    pipeline_type,
    batch_size,
    config_fp,
    tokenizer,
    deterministic_ex_noise: bool = False,
    max_transcript_len: int = 450,
) -> DaliDataLoader:
    """
    Build dali dataloader helper function for testing.
    """
    cfg = config.load(config_fp)
    cfg["input_train"]["audio_dataset"]["max_transcript_len"] = max_transcript_len
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
        world_size=1,
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
            # i.e. an infinite loop was encountered when cycle=True
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
    prev_txt = None
    shuffle_seen = False
    for idx in range(n_epochs):
        for _, _, txt, _ in dataloader:
            print(idx, txt)
            if prev_txt is None or torch.allclose(prev_txt, txt):
                prev_txt = txt
                continue
            shuffle_seen = True
            break
    assert (
        shuffle_seen
    ), f"Data hasn't changed order after {n_epochs=}. Data not shuffled?"


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
    # for n_epochs it's expected to see samples appear in same order since no
    # shuffling has happened
    prev_txt = None
    for _ in range(n_epochs):
        for _, _, txt, _ in dataloader:
            if prev_txt is None or torch.allclose(prev_txt, txt):
                prev_txt = txt
                continue
            raise RuntimeError("Shuffling has happened, even though it shouldn't have")


# One transcript is 33 chars long; the other is 34 chars long.
@pytest.mark.parametrize("max_transcript_len,expected_size", [(33, 1), (450, 2)])
def test_webdataset_size(
    dataload_args_webdataset,
    config_fp,
    tokenizer,
    max_transcript_len,
    expected_size,
):
    batch_size = 1
    dataloader = build_dataloader_util(
        dataload_args_webdataset,
        "train",
        batch_size,
        config_fp,
        tokenizer,
        max_transcript_len=max_transcript_len,
    )
    size = 0
    for _ in dataloader:
        size += 1
    assert size == expected_size


def test_dali_equivalence(dataload_args, saved_tensor_no_noise, config_fp, tokenizer):
    """
    Test dali equivalence to tensors saved to file.

    This is used to issue alerts on changes in the preprocessing code as it may
    imply that the Rust preprocessing code has to change in order to match the ASR server.
    """
    dali_dataloder_deterministic = build_dataloader_util(
        dataload_args, "val", 2, config_fp, tokenizer, True
    )
    for audio, _, _, _ in dali_dataloder_deterministic:
        assert torch.allclose(audio, saved_tensor_no_noise, atol=2e-04)


@pytest.mark.parametrize(
    "dataload_args_name",
    ["dataload_args_noise", "dataload_args_noise_babble"],
)
def test_dali_noise(
    saved_tensor_no_noise, config_fp, request, dataload_args_name, tokenizer
):
    dataload_args = request.getfixturevalue(dataload_args_name)
    noise_loader = build_dataloader_util(
        dataload_args, "train", 2, config_fp, tokenizer, True
    )
    # test two runs are different
    audio1 = [batch[0] for batch in noise_loader][0]
    audio2 = [batch[0] for batch in noise_loader][0]

    assert audio1.shape == audio2.shape
    assert not torch.allclose(audio1, torch.ones_like(audio1)), (
        "audio should not be all ones! This occurs if tensor of zeros "
        "is passed through a log()"
    )
    assert not torch.allclose(
        audio1, audio2
    ), "These should be different as applying noise augmentation"

    assert audio1.shape == saved_tensor_no_noise.shape
    assert not torch.allclose(
        audio1, saved_tensor_no_noise
    ), "These should be different as only one has noise augmentation applied"


def test_dali_low_noise(
    dataload_args_noise, saved_tensor_no_noise, config_fp, tokenizer
):
    noise_loader = build_dataloader_util(
        dataload_args_noise, "train", 2, config_fp, tokenizer, True
    )
    # now set noise augmentation SNR very high - hence noise aug should be ~0
    # dB is log scale so 400 is very large
    noise_loader.pipeline.background_noise_iterator.set_range(400, 400)

    _ = [batch[0] for batch in noise_loader][0]
    _ = [batch[0] for batch in noise_loader][0]
    audio_no_noise3 = [batch[0] for batch in noise_loader][0]
    audio_no_noise4 = [batch[0] for batch in noise_loader][0]
    audio_no_noise5 = [batch[0] for batch in noise_loader][0]

    assert torch.allclose(
        saved_tensor_no_noise, audio_no_noise4, atol=5e-04
    ), "These should be the same as we aren't applying augmentation in noise_loader"

    assert not torch.allclose(
        audio_no_noise3, audio_no_noise4, atol=5e-04
    ), "These are expected to be different"
    assert torch.allclose(audio_no_noise4, audio_no_noise5, atol=5e-04), (
        "These are expected to be the same when not applying augmentation"
        " in noise_loader"
    )
