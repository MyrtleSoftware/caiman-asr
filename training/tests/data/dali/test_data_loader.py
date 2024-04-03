import copy
from argparse import Namespace

import numpy as np
import pytest
import torch
from beartype import beartype

from caiman_asr_train.data.build_dataloader import build_dali_loader
from caiman_asr_train.data.dali import sampler
from caiman_asr_train.data.dali.data_loader import DaliDataLoader
from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.webdataset import LengthUnknownError
from caiman_asr_train.rnnt import config
from caiman_asr_train.setup.core import PipelineType
from caiman_asr_train.setup.dali import build_dali_yaml_config
from caiman_asr_train.setup.mel_normalization import build_mel_feat_normalizer
from caiman_asr_train.test_utils.dataload_args import update_dataload_args
from caiman_asr_train.utils.seed import set_seed


def test_samplers_initialization(
    dataload_args,
    mini_config_fp,
    tokenizer,
    batch_size=1,
    global_batch_size=1,
    training_steps=6,
    resume_step=0,
):
    dataloader1 = build_dataloader_util(
        dataload_args, "train", batch_size, mini_config_fp, tokenizer
    )
    assert len(dataloader1) == 2
    assert isinstance(dataloader1.sampler, sampler.SimpleSampler)
    assert not isinstance(dataloader1.sampler, sampler.BucketingSampler)

    bucketing_sampler = sampler.BucketingSampler(
        num_buckets=dataload_args.num_buckets,
        batch_size=batch_size,
        num_workers=1,
        training_steps=training_steps,
        global_batch_size=global_batch_size,
        rng=np.random.default_rng(seed=dataload_args.seed),
        resume_step=resume_step,
    )
    dataloader2 = build_dataloader_util(
        dataload_args,
        "train",
        batch_size,
        mini_config_fp,
        tokenizer,
        train_sampler=bucketing_sampler,
    )
    assert isinstance(dataloader2.sampler, sampler.SimpleSampler)
    assert isinstance(dataloader2.sampler, sampler.BucketingSampler)
    txt_list = []
    for idx in range(training_steps // len(dataloader2)):
        for _, _, txt, _ in dataloader2:
            txt_list.append(txt)
    assert len(txt_list) == training_steps
    assert torch.allclose(txt_list[0], txt_list[len(dataloader2)]) or torch.allclose(
        txt_list[0], txt_list[2 * len(dataloader2)]
    )


@pytest.fixture(scope="session")
def dataload_args_webdataset(dataload_args) -> Namespace:
    return update_dataload_args(dataload_args, DataSource.TARFILE)


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
    mini_config_fp,
    tokenizer,
    train_sampler=None,
    deterministic_ex_noise: bool = False,
    max_transcript_len: int = 450,
    normalize: bool = True,
) -> DaliDataLoader:
    """
    Build dali dataloader helper function for testing.
    """
    cfg = config.load(mini_config_fp)
    cfg["input_train"]["audio_dataset"]["max_transcript_len"] = max_transcript_len
    dataset_kw, features_kw, _, _ = config.input(cfg, pipeline_type)

    if deterministic_ex_noise:
        # make dataloader deterministic except for noise augmentation
        features_kw["dither"] = 0.0
        dataset_kw["speed_perturbation"] = None

    dali_yaml_config = build_dali_yaml_config(
        config_data=dataset_kw, config_features=features_kw
    )
    pipeline_type_enum = (
        PipelineType.TRAIN if pipeline_type == "train" else PipelineType.VAL
    )
    mel_feat_normalizer = (
        build_mel_feat_normalizer(
            dataload_args, dali_yaml_config, pipeline_type_enum, batch_size
        )
        if normalize
        else None
    )
    return build_dali_loader(
        dataload_args,
        pipeline_type,
        batch_size=batch_size,
        dali_yaml_config=dali_yaml_config,
        train_sampler=train_sampler,
        tokenizer=tokenizer,
        cpu=True,
        no_logging=True,
        mel_feat_normalizer=mel_feat_normalizer,
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
    mini_config_fp,
    webdataset,
    tokenizer,
):
    if webdataset:
        args = dataload_args_webdataset
    else:
        args = dataload_args
    dataloader = build_dataloader_util(
        args, pipeline_type, batch_size, mini_config_fp, tokenizer
    )
    dataset_length = 2  # to match the test data

    if not webdataset:
        assert len(dataloader) == dataset_length // batch_size
        if pipeline_type == "train":
            assert isinstance(dataloader.sampler, sampler.SimpleSampler)
            assert not isinstance(dataloader.sampler, sampler.BucketingSampler)
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
    mini_config_fp,
    tokenizer,
    batch_size=2,
    n_epochs=3,
):
    set_seed(dataload_args_webdataset.seed)
    dataloader = build_dataloader_util(
        dataload_args_webdataset, "train", batch_size, mini_config_fp, tokenizer
    )
    assert isinstance(dataloader, DaliDataLoader)
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
    mini_config_fp,
    tokenizer,
    batch_size=2,
    n_epochs=5,
):
    dataloader = build_dataloader_util(
        dataload_args_webdataset, "val", batch_size, mini_config_fp, tokenizer
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
    mini_config_fp,
    tokenizer,
    max_transcript_len,
    expected_size,
):
    batch_size = 1
    dataloader = build_dataloader_util(
        dataload_args_webdataset,
        "train",
        batch_size,
        mini_config_fp,
        tokenizer,
        max_transcript_len=max_transcript_len,
    )
    size = 0
    for _ in dataloader:
        size += 1
    assert size == expected_size


def test_dali_equivalence(
    dataload_args, saved_tensor_no_noise, mini_config_fp, tokenizer
):
    """
    Test dali equivalence to tensors saved to file.

    This is used to issue alerts on changes in the preprocessing code as it may
    imply that the Rust preprocessing code has to change in order to match the ASR server.

    This test was written when normalization over the utterance was the default so we
    set norm_over_utterance=True.
    """
    dataload_args = copy.deepcopy(dataload_args)
    dataload_args.norm_over_utterance = True
    dali_dataloder_deterministic = build_dataloader_util(
        dataload_args,
        "val",
        2,
        mini_config_fp,
        tokenizer,
        deterministic_ex_noise=True,
    )
    for audio, _, _, _ in dali_dataloder_deterministic:
        assert torch.allclose(audio, saved_tensor_no_noise, atol=2e-04)


@pytest.mark.parametrize(
    "dataload_args_name",
    ["dataload_args_noise", "dataload_args_noise_babble"],
)
def test_dali_noise(
    saved_tensor_no_noise, mini_config_fp, request, dataload_args_name, tokenizer
):
    dataload_args = request.getfixturevalue(dataload_args_name)
    noise_loader = build_dataloader_util(
        dataload_args,
        "train",
        2,
        mini_config_fp,
        tokenizer,
        deterministic_ex_noise=True,
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
    dataload_args_noise, saved_tensor_no_noise, mini_config_fp, tokenizer
):
    dataload_args = copy.deepcopy(dataload_args_noise)
    dataload_args.norm_over_utterance = True
    noise_loader = build_dataloader_util(
        dataload_args,
        "train",
        2,
        mini_config_fp,
        tokenizer,
        deterministic_ex_noise=True,
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


@pytest.mark.parametrize("format", DataSource)
@pytest.mark.parametrize("norm_over_utterance", [False, True])
def test_normalization(
    dataload_args,
    format,
    norm_over_utterance,
    mini_config_fp,
    tokenizer,
    melmeans,
    melvars,
):
    """
    For each data format check:
    1) that the norm_over_utterance flag works as expected
    2) that the mel stats norm works as expected
    """
    args = update_dataload_args(dataload_args, format)
    args.norm_over_utterance = norm_over_utterance

    shared_args = (args, "val", 1, mini_config_fp, tokenizer)
    loader_norm = build_dataloader_util(
        *shared_args,
        deterministic_ex_noise=True,
    )
    args.norm_over_utterance = False
    loader_no_norm = build_dataloader_util(
        *shared_args, deterministic_ex_noise=True, normalize=False
    )

    feats_norm = [batch[0] for batch in loader_norm][0]
    feats_no_norm = [batch[0] for batch in loader_no_norm][0]

    # audio is (batch, <mel dim>, time) and mean/var are (<mel dim>)
    if not norm_over_utterance:
        mean = melmeans.view(1, -1, 1)
        std = melvars.sqrt().view(1, -1, 1)
    else:
        # calculate the per-utterance norms over time dim=2
        mean = feats_no_norm.mean(2).unsqueeze(2)
        std = feats_no_norm.std(2).unsqueeze(2)

    expected_norm = (feats_no_norm - mean) / std
    tol = 1e-3 if norm_over_utterance else 1e-6
    assert torch.allclose(feats_norm, expected_norm, atol=tol, rtol=tol), (
        "It should be possible to recover the normalized audio by applying the "
        f"operation manually outside of dali. {format=}, {norm_over_utterance=}"
    )
    assert not torch.allclose(feats_norm, feats_no_norm), (
        "These should be different as one has mel stats normalization applied. "
        f"{format.name=}, {norm_over_utterance=}"
    )


def test_norm_blend(
    dataload_args,
    mini_config_fp,
    tokenizer,
    starting_ratio=0.25,
):
    args = update_dataload_args(dataload_args, DataSource.JSON)
    args.norm_starting_ratio = starting_ratio

    shared_args = (args, "train", 2, mini_config_fp, tokenizer)
    loader_blend_norm = build_dataloader_util(
        *shared_args,
        deterministic_ex_noise=True,
    )
    args.norm_over_utterance = True
    loader_utt_norm = build_dataloader_util(
        *shared_args,
        deterministic_ex_noise=True,
    )
    args.norm_over_utterance = False

    loader_dataset_norm = build_dataloader_util(
        args,
        "val",
        2,
        mini_config_fp,
        tokenizer,
        deterministic_ex_noise=True,
    )
    for batch1, batch2, batch3 in zip(
        loader_blend_norm, loader_utt_norm, loader_dataset_norm
    ):
        feats_blend = batch1[0]
        feats_utt = batch2[0]
        feats_dataset = batch3[0]
        expected = (1 - starting_ratio) * feats_utt + starting_ratio * feats_dataset

        assert not torch.allclose(feats_blend, feats_utt)
        assert torch.allclose(feats_blend, expected, atol=1e-6, rtol=1e-6), (
            "It should be possible to recover the blended norm audio by applying the "
            "operation manually outside of dali."
        )
