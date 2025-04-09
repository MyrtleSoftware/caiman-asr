from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import torch.multiprocessing as mp
from beartype import beartype
from beartype.typing import List, Optional

from caiman_asr_train.args.hugging_face import HuggingFaceArgs, build_hugging_face_args
from caiman_asr_train.args.noise_augmentation import NoiseAugmentationArgs
from caiman_asr_train.data.dali import sampler as dali_sampler
from caiman_asr_train.data.dali.data_loader import DaliDataLoader
from caiman_asr_train.data.dali.manifest_ratios import (
    ManifestRatios,
    build_manifest_ratios,
)
from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer
from caiman_asr_train.data.decide_on_loader import DataSource, decide_on_loader
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.dali import DaliYAMLConfig


@beartype
@dataclass
class DataLoaderArgs:
    """
    A subset of the args required to initialise DataLoaders.

    The function of this class is to switch between the train and val specific args in
    from_namespace.
    """

    grad_accumulation_batches: int
    json_names: Optional[List[str]]
    manifest_ratios: ManifestRatios
    num_buckets: Optional[int]
    prob_narrowband: float
    noise_augmentation_args: NoiseAugmentationArgs
    final_padding_secs: float
    hugging_face_args: Optional[HuggingFaceArgs]
    dali_device: str
    tar_files: Optional[List[str]] = None

    @classmethod
    def from_namespace(cls, args: Namespace, pipeline_type: str) -> "DataLoaderArgs":
        if pipeline_type not in ("train", "val"):
            raise ValueError(
                f"pipeline_type must be 'train' or 'val', not {pipeline_type}"
            )

        if pipeline_type == "train":
            grad_accumulation_batches = args.grad_accumulation_batches
            json_names = args.train_manifests

            manifest_ratios = build_manifest_ratios(
                train_manifest_ratios=args.train_manifest_ratios,
                relative_train_manifest_ratios=args.relative_train_manifest_ratios,
                canary_manifest_exponent=args.canary_exponent,
            )

            tar_files = args.train_tar_files
            num_buckets = args.num_buckets
            prob_narrowband = args.prob_train_narrowband
            noise_augmentation_args = NoiseAugmentationArgs(
                noise_dataset=args.noise_dataset,
                noise_config=args.noise_config,
                use_noise_audio_folder=args.use_noise_audio_folder,
                prob_background_noise=args.prob_background_noise,
                prob_babble_noise=args.prob_babble_noise,
            )
            hugging_face_args = None
            dali_device = args.dali_train_device
            final_padding_secs = 0.0

        else:
            grad_accumulation_batches = 1
            json_names = args.val_manifests
            manifest_ratios = None
            tar_files = args.val_tar_files
            num_buckets = 1
            prob_narrowband = args.prob_val_narrowband
            noise_augmentation_args = NoiseAugmentationArgs(
                noise_dataset=None,
                noise_config=None,
                use_noise_audio_folder=False,
                prob_background_noise=0.0,
                prob_babble_noise=0.0,
            )
            hugging_face_args = build_hugging_face_args(
                use_hugging_face=args.use_hugging_face,
                dataset=args.hugging_face_val_dataset,
                split=args.hugging_face_val_split,
                transcript_key=args.hugging_face_val_transcript_key,
                config=args.hugging_face_val_config,
            )
            dali_device = args.dali_val_device
            final_padding_secs = args.val_final_padding_secs

        return cls(
            noise_augmentation_args=noise_augmentation_args,
            grad_accumulation_batches=grad_accumulation_batches,
            json_names=json_names,
            manifest_ratios=manifest_ratios,
            num_buckets=num_buckets,
            tar_files=tar_files,
            prob_narrowband=prob_narrowband,
            hugging_face_args=hugging_face_args,
            dali_device=dali_device,
            final_padding_secs=final_padding_secs,
        )


def build_dali_loader(
    args: Namespace,
    pipeline_type: str,
    batch_size: int,
    dali_yaml_config: DaliYAMLConfig,
    tokenizer: Tokenizer,
    world_size: int,
    train_sampler: Optional[dali_sampler.Sampler] = None,
    cpu: bool = False,
    no_logging: bool = False,
    mel_feat_normalizer: Optional[MelFeatNormalizer] = None,
) -> DaliDataLoader:
    """
    Build dali dataloader.

    Args:
        args: train.py or val.py argparse arguments.
        pipeline_type: string in {'train', 'val'}
        batch_size: batch size of dataloder
        dali_yaml_config: DaliYAMLConfig
        tokenizer: Tokenizer for transcripts
        train_sampler: A sampler to control order and, optionally, sharding of the samples.
        cpu: bool. If True, then run DALI without CUDA. Note that this is different to
            the dali_device arg.
        no_logging: bool. If True, then silence the DALI debugging log.
        mel_feat_normalizer: class to normalize mel features.
    """
    dataload_args = DataLoaderArgs.from_namespace(args, pipeline_type)

    data_source = decide_on_loader(
        val_from_dir=args.val_from_dir,
        read_from_tar=args.read_from_tar,
        use_hugging_face=dataload_args.hugging_face_args is not None,
    )

    if data_source is not DataSource.JSON:
        sampler = None
    elif pipeline_type == "val":
        # setup/train.py and setup/val.py set the validation sampler to None,
        # and it's created here
        sampler = dali_sampler.SortedSampler(
            world_size=world_size,
            total_batches=None,
            batch_size=batch_size,
            global_batch_size=None,
        )
    elif train_sampler is None:
        # Note: This branch is probably dead code only used by the tests
        sampler = dali_sampler.SimpleSampler(
            world_size=world_size,
            total_batches=None,
            batch_size=batch_size,
            global_batch_size=args.global_batch_size,
        )
    else:
        sampler = train_sampler

    return DaliDataLoader(
        gpu_id=(
            args.local_rank if not cpu else None
        ),  # Use None as a device_id to run DALI without CUDA
        dataset_path=args.dataset_dir,
        dali_yaml_config=dali_yaml_config,
        json_names=dataload_args.json_names,
        manifest_ratios=dataload_args.manifest_ratios,
        batch_size=batch_size,
        sampler=sampler,
        grad_accumulation_batches=dataload_args.grad_accumulation_batches,
        pipeline_type=pipeline_type,
        mel_feat_normalizer=mel_feat_normalizer,
        num_cpu_threads=int(args.dali_processes_per_cpu * mp.cpu_count() / world_size),
        num_buckets=dataload_args.num_buckets,
        device_type=dataload_args.dali_device,
        tokenizer=tokenizer,
        tar_files=dataload_args.tar_files,
        val_from_dir=args.val_from_dir,
        audio_dir=args.val_audio_dir,
        txt_dir=args.val_txt_dir,
        no_logging=no_logging,
        seed=args.seed,
        turn_off_initial_padding=args.turn_off_initial_padding,
        final_padding_secs=dataload_args.final_padding_secs,
        inspect_audio=args.inspect_audio,
        prob_narrowband=dataload_args.prob_narrowband,
        output_dir=Path(args.output_dir),
        n_utterances_only=args.n_utterances_only,
        noise_augmentation_args=dataload_args.noise_augmentation_args,
        data_source=data_source,
        hugging_face_args=dataload_args.hugging_face_args,
    )
