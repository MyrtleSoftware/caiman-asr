#!/usr/bin/env python3

import datetime
import os
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Callable, Dict, List, Optional, Tuple, Union
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from rnnt_train.common.data import features
from rnnt_train.common.data.build_dataloader import build_dali_loader
from rnnt_train.common.data.dali import sampler as dali_sampler
from rnnt_train.common.data.dali.data_loader import DaliDataLoader
from rnnt_train.common.data.grad_noise_scheduler import GradNoiseScheduler
from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.helpers import Checkpointer, print_once
from rnnt_train.rnnt import config
from rnnt_train.rnnt.decoder import RNNTDecoder
from rnnt_train.rnnt.greedy import RNNTGreedyDecoder
from rnnt_train.rnnt.loss import ApexTransducerLoss
from rnnt_train.rnnt.model import RNNT
from rnnt_train.rnnt.sub_models import RNNTSubModels


class PipelineType(Enum):
    TRAIN = 1
    VAL = 2


TRAIN = PipelineType.TRAIN
VAL = PipelineType.VAL


CPU = torch.device("cpu")
CUDA = torch.device("cuda")


@beartype
class OptimizerWrapper:
    """Wrapper to control the optimizer and AMP scaling during training."""

    def __init__(
        self,
        args: Namespace,
        optimizer,
        scaler: Optional[GradScaler],
    ):
        self.args = args
        self.optimizer = optimizer
        self.scaler = scaler

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, total_norm: float) -> None:
        if not self.args.no_amp:
            self.do_scaler_step()
            self.scaler.update()
        else:
            # when not using AMP test for inf / NaN gradients
            if np.isfinite(total_norm):
                self.optimizer.step()

    def do_scaler_step(self) -> None:
        # pyTorch AMP step function unscales the gradients
        # if these gradients do not contain infs or NaNs, optimizer.step() is then called
        self.scaler.step(self.optimizer)

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


@beartype
@dataclass
class TrainingOnly:
    adjust_lr: Callable
    meta: dict
    checkpointer: Checkpointer
    grad_noise_scheduler: Optional[GradNoiseScheduler]
    optimizer_wrapper: OptimizerWrapper
    train_step_fn: Callable


@beartype
@dataclass
class DataObject:
    loader: DaliDataLoader
    dataset_kw: dict
    features_kw: dict


@beartype
@dataclass
class BuiltObjects:
    decoder: RNNTDecoder
    model: Union[RNNT, DDP, RNNTSubModels]
    ema_model: RNNT
    tokenizers: Dict[PipelineType, Tokenizer]
    tokenizers_kw: Dict[PipelineType, dict]
    loss_fn: ApexTransducerLoss
    multi_gpu: bool
    cfg: dict
    world_size: int
    feat_procs: Dict[PipelineType, torch.nn.Module]
    training_only: Optional[TrainingOnly]
    data_objects: Dict[PipelineType, DataObject]


@beartype
class Setup(ABC):
    def run(self, args: Namespace) -> BuiltObjects:
        """Initialise the objects required for each PipelineType."""
        # Must make output directory before the logging starts in initial_setup()
        os.makedirs(args.output_dir, exist_ok=True)
        multi_gpu, world_size, np_rng, cfg, batch_sizes = self.initial_setup(args)
        (
            tokenizers,
            blank_idx,
            tokenizers_kw,
            default_tokenizer,
            default_tokenizer_kw,
        ) = self.build_all_tokenizers(args, cfg)
        decoder, loss_fn = self.build_evaluation_objects(
            args, cfg, blank_idx, default_tokenizer, default_tokenizer_kw
        )
        data_objects, feat_procs = self.build_data_and_feat_proc(
            args, np_rng, world_size, batch_sizes, cfg, tokenizers
        )
        model, ema_model, training_only = self.build_model(
            args, cfg, default_tokenizer, multi_gpu, default_tokenizer_kw
        )
        return BuiltObjects(
            decoder=decoder,
            model=model,
            tokenizers=tokenizers,
            loss_fn=loss_fn,
            multi_gpu=multi_gpu,
            cfg=cfg,
            data_objects=data_objects,
            feat_procs=feat_procs,
            ema_model=ema_model,
            world_size=world_size,
            tokenizers_kw=tokenizers_kw,
            training_only=training_only,
        )

    def initial_setup(
        self, args: Namespace
    ) -> Tuple[bool, int, np.random.Generator, dict, Dict[PipelineType, int]]:
        self.torch_settings(args)
        multi_gpu, world_size = self.start_ddp(args)
        np_rng = self.seed_and_logging(args)
        cfg = self.build_cfg(args)
        batch_sizes = self.get_batch_sizes(args)
        return multi_gpu, world_size, np_rng, cfg, batch_sizes

    def build_all_tokenizers(
        self, args: Namespace, cfg: dict
    ) -> Tuple[
        Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict], Tokenizer, dict
    ]:
        tokenizers, blank_idx, tokenizers_kw = self.build_tokenizer(args, cfg)
        default_tokenizer = self.get_default(tokenizers)
        default_tokenizer_kw = self.get_default(tokenizers_kw)
        return (
            tokenizers,
            blank_idx,
            tokenizers_kw,
            default_tokenizer,
            default_tokenizer_kw,
        )

    def build_evaluation_objects(
        self,
        args: Namespace,
        cfg: dict,
        blank_idx: int,
        default_tokenizer: Tokenizer,
        default_tokenizer_kw: dict,
    ) -> Tuple[RNNTDecoder, ApexTransducerLoss]:
        loss_fn = self.build_loss_fn(blank_idx, cfg)
        lm_info = self.build_lm_info(args, default_tokenizer, default_tokenizer_kw)
        decoder = self.build_decoder(args, blank_idx, default_tokenizer, lm_info)
        return decoder, loss_fn

    @abstractmethod
    def build_model(
        self,
        args: Namespace,
        cfg: dict,
        tokenizer: Tokenizer,
        multi_gpu: bool,
        tokenizer_kw: dict,
    ) -> Tuple[Union[RNNT, DDP, RNNTSubModels], RNNT, Optional[TrainingOnly]]:
        pass

    def build_data_and_feat_proc(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        batch_sizes: Dict[PipelineType, int],
        cfg: dict,
        tokenizers: Dict[PipelineType, Tokenizer],
    ) -> Tuple[Dict[PipelineType, DataObject], Dict[PipelineType, torch.nn.Module]]:
        samplers = self.build_samplers(args, np_rng, world_size, batch_sizes)
        feat_procs = self.build_each_pipeline_type(self.build_a_feat_proc, args, cfg)
        data_objects = self.build_each_pipeline_type(
            self.build_data_object,
            args,
            cfg,
            tokenizers,
            batch_sizes,
            samplers,
            world_size,
        )
        return data_objects, feat_procs

    def build_decoder(
        self,
        args: Namespace,
        blank_idx: int,
        tokenizer: Tokenizer,
        lm_info: Optional[Tuple],
    ) -> RNNTDecoder:
        return RNNTGreedyDecoder(
            blank_idx=blank_idx,
            max_symbol_per_sample=args.max_symbol_per_sample,
            lm_info=lm_info,
        )

    def build_lm_info(
        self, args: Namespace, tokenizer: Tokenizer, tokenizer_kw: dict
    ) -> None:
        return None

    def build_loss_fn(self, blank_idx: int, cfg: dict) -> ApexTransducerLoss:
        """set up the loss function: if the TransducerJoint has packed_output=True then the
        input to the ApexTransducerLoss must have packed_input=True"""
        rnnt_config = config.rnnt(cfg)
        rnnt_config["gpu_unavailable"] = self.preferred_device() == CPU
        return ApexTransducerLoss(
            blank_idx=blank_idx,
            packed_input=rnnt_config["joint_apex_transducer"] == "pack",
        )

    def build_cfg(self, args: Namespace) -> dict:
        return config.load(args.model_config)

    def pipeline_type_to_str(self, pipeline_type: PipelineType) -> str:
        return {TRAIN: "train", VAL: "val"}[pipeline_type]

    def build_a_feat_proc(
        self, pipeline_type: PipelineType, args: Namespace, cfg: dict
    ) -> torch.nn.Module:
        (_, _, splicing_kw, specaugm_kw) = config.input(
            cfg, self.pipeline_type_to_str(pipeline_type)
        )
        feat_proc = torch.nn.Sequential(
            specaugm_kw and features.SpecAugment(**specaugm_kw) or torch.nn.Identity(),
            features.FrameSplicing(**splicing_kw),
            features.PermuteAudio(),
        )
        feat_proc.to(self.preferred_device())
        return feat_proc

    def build_data_object(
        self,
        pipeline_type: PipelineType,
        args: Namespace,
        cfg: dict,
        tokenizers: Dict[PipelineType, Tokenizer],
        batch_sizes: Dict[PipelineType, int],
        samplers: Dict[PipelineType, Optional[dali_sampler.SimpleSampler]],
        world_size: int,
    ) -> DataObject:
        print_once("Setting up datasets...")
        (dataset_kw, features_kw, _, _) = config.input(
            cfg, self.pipeline_type_to_str(pipeline_type)
        )
        self.modify_dataloader_kw(args, dataset_kw, features_kw, pipeline_type)
        loader = build_dali_loader(
            args,
            self.pipeline_type_to_str(pipeline_type),
            batch_size=batch_sizes[pipeline_type],
            dataset_kw=dataset_kw,
            features_kw=features_kw,
            tokenizer=tokenizers[pipeline_type],
            train_sampler=samplers[pipeline_type],
            cpu=self.preferred_device() == CPU,
            world_size=world_size,
        )
        return DataObject(loader=loader, dataset_kw=dataset_kw, features_kw=features_kw)

    def modify_dataloader_kw(
        self,
        args: Namespace,
        dataset_kw: dict,
        features_kw: dict,
        pipeline_type: PipelineType,
    ):
        pass

    def build_each_pipeline_type(self, builder: Callable, *other_args) -> dict:
        """For val, this returns {VAL: builder(VAL)}
        For train, this returns {TRAIN: builder(TRAIN), VAL: builder(VAL)}"""
        return {
            pipeline_type: builder(pipeline_type, *other_args)
            for pipeline_type in self.pipeline_types()
        }

    @abstractmethod
    def build_tokenizer(
        self, args: Namespace, cfg: dict
    ) -> Tuple[Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict]]:
        pass

    @abstractmethod
    def get_batch_sizes(self, args: Namespace) -> Dict[PipelineType, int]:
        pass

    @abstractmethod
    def build_samplers(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        batch_sizes: Dict[PipelineType, int],
    ) -> Dict[PipelineType, Optional[dali_sampler.SimpleSampler]]:
        pass

    @abstractmethod
    def pipeline_types(self) -> List[PipelineType]:
        pass

    @abstractmethod
    def seed_and_logging(self, args: Namespace) -> np.random.Generator:
        pass

    @abstractmethod
    def preferred_device(self) -> torch.device:
        pass

    def torch_settings(self, args: Namespace):
        if self.preferred_device() == CPU:
            # Set PyTorch to run on one CPU thread to ensure deterministic PyTorch output.
            torch.set_num_threads(1)
            if not args.skip_init:
                torch.set_num_interop_threads(1)

        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark

    @abstractmethod
    def start_ddp(self, args) -> Tuple[bool, int]:
        pass

    def start_ddp_with_gpu(
        self, args: Namespace, multi_gpu: bool, timeout: Dict[str, int]
    ) -> Tuple[bool, int]:
        """Only called in val.py and train.py"""
        assert torch.cuda.is_available()

        # set up distributed processing
        if multi_gpu:
            torch.cuda.set_device(args.local_rank)
            if not args.skip_init:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=datetime.timedelta(**timeout),
                )
            world_size = dist.get_world_size()
            print_once(f"Distributed processing with {world_size} GPUs\n")
        else:
            world_size = 1
        return multi_gpu, world_size

    @property
    def multi_gpu(self) -> bool:
        return int(os.environ.get("WORLD_SIZE", 1)) > 1

    def get_default(self, dictionary):
        return dictionary[TRAIN] if TRAIN in dictionary else dictionary[VAL]

    def __setattr__(self, name, value):
        raise AttributeError(
            """The Setup class doesn't allow setting attributes.
            Data must be visibly passed between methods inside run()."""
        )
