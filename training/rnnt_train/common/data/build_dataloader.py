from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import torch.multiprocessing as mp
from beartype import beartype
from beartype.typing import List, Optional

from rnnt_train.common.data.dali import sampler as dali_sampler
from rnnt_train.common.data.dali.data_loader import DaliDataLoader
from rnnt_train.common.data.noise_augmentation_args import NoiseAugmentationArgs
from rnnt_train.common.data.text import Tokenizer


@beartype
@dataclass
class DataLoaderArgs:
    """
    A subset of the args required to initialise DataLoaders.

    The function of this class is to switch between the train and val specific args in
    from_namespace.
    """

    grad_accumulation_batches: int
    json_names: List[str]
    normalize: bool
    shuffle: bool
    num_buckets: Optional[int]
    prob_narrowband: float
    noise_augmentation_args: NoiseAugmentationArgs
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
            tar_files = args.train_tar_files
            normalize = not args.dump_mel_stats
            shuffle = True
            num_buckets = args.num_buckets
            prob_narrowband = args.prob_train_narrowband
            noise_augmentation_args = NoiseAugmentationArgs(
                noise_dataset=args.noise_dataset,
                noise_config=args.noise_config,
                use_noise_audio_folder=args.use_noise_audio_folder,
                prob_background_noise=args.prob_background_noise,
                prob_babble_noise=args.prob_babble_noise,
            )

        else:
            grad_accumulation_batches = 1
            json_names = args.val_manifests
            tar_files = args.val_tar_files
            normalize = not getattr(args, "streaming_normalization", False)
            shuffle = False
            num_buckets = 1
            prob_narrowband = args.prob_val_narrowband
            noise_augmentation_args = NoiseAugmentationArgs(
                noise_dataset=None,
                noise_config=None,
                use_noise_audio_folder=False,
                prob_background_noise=0.0,
                prob_babble_noise=0.0,
            )

        return cls(
            noise_augmentation_args=noise_augmentation_args,
            grad_accumulation_batches=grad_accumulation_batches,
            json_names=json_names,
            normalize=normalize,
            shuffle=shuffle,
            num_buckets=num_buckets,
            tar_files=tar_files,
            prob_narrowband=prob_narrowband,
        )


def build_dali_loader(
    args: Namespace,
    pipeline_type: str,
    batch_size: int,
    dataset_kw: dict,
    features_kw: dict,
    tokenizer: Tokenizer,
    world_size: int,
    train_sampler: Optional[dali_sampler.SimpleSampler] = None,
    cpu: bool = False,
    no_logging: bool = False,
) -> DaliDataLoader:
    """
    Build dali dataloader.

    Args:
        args: train.py or val.py argparse arguments.
        pipeline_type: string in {'train', 'val'}
        batch_size: batch size of dataloder
        dataset_kw: dictionary of dataset config
        features_kw: dictionary of features config
        tokenizer: Tokenizer for transcripts
        train_sampler: A sampler to control order and, optionally, sharding of the samples.
        cpu: bool. If True, then run DALI without CUDA. Note that this is different to
            the dali_device arg.
        no_logging: bool. If True, then silence the DALI debugging log.
    """
    if args.read_from_tar:
        sampler = None
    elif pipeline_type == "val" or train_sampler is None:
        sampler = dali_sampler.SimpleSampler()
    else:
        sampler = train_sampler

    dataload_args = DataLoaderArgs.from_namespace(args, pipeline_type)
    return DaliDataLoader(
        gpu_id=args.local_rank
        if not cpu
        else None,  # Use None as a device_id to run DALI without CUDA
        dataset_path=args.dataset_dir,
        config_data=dataset_kw,
        config_features=features_kw,
        json_names=dataload_args.json_names,
        batch_size=batch_size,
        sampler=sampler,
        grad_accumulation_batches=dataload_args.grad_accumulation_batches,
        pipeline_type=pipeline_type,
        normalize=dataload_args.normalize,
        num_cpu_threads=int(args.dali_processes_per_cpu * mp.cpu_count() / world_size),
        num_buckets=dataload_args.num_buckets,
        device_type=args.dali_device,
        tokenizer=tokenizer,
        tar_files=dataload_args.tar_files,
        read_from_tar=args.read_from_tar,
        val_from_dir=args.val_from_dir,
        audio_dir=args.val_audio_dir,
        txt_dir=args.val_txt_dir,
        no_logging=no_logging,
        seed=args.seed,
        turn_off_initial_padding=args.turn_off_initial_padding,
        inspect_audio=args.inspect_audio,
        prob_narrowband=dataload_args.prob_narrowband,
        output_dir=Path(args.output_dir),
        n_utterances_only=args.n_utterances_only,
        noise_augmentation_args=dataload_args.noise_augmentation_args,
    )
