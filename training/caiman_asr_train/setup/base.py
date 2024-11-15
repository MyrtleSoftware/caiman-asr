#!/usr/bin/env python3

import datetime
import os
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Callable, Dict, List, Optional, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.data import features
from caiman_asr_train.data.build_dataloader import build_dali_loader
from caiman_asr_train.data.dali import sampler as dali_sampler
from caiman_asr_train.data.dali.data_loader import DaliDataLoader
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.evaluate.error_rates import ErrorRate, get_error_rate
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.lm.kenlm_ngram import NgramInfo, find_ngram_path
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.batched_greedy import RNNTBatchedGreedyDecoder
from caiman_asr_train.rnnt.beam import RNNTBeamDecoder
from caiman_asr_train.rnnt.decoder import RNNTDecoder
from caiman_asr_train.rnnt.eos_strategy import EOSBlank, EOSIgnore, EOSPredict
from caiman_asr_train.rnnt.loss import ApexTransducerLoss
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.parallel_decoder import ParallelDecoder, get_num_procs
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.setup.core import CPU, TRAIN, VAL, PipelineType
from caiman_asr_train.setup.dali import build_dali_yaml_config
from caiman_asr_train.setup.mel_normalization import build_mel_feat_normalizer
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.train_utils.grad_noise_scheduler import GradNoiseScheduler
from caiman_asr_train.train_utils.optimizer import OptimizerWrapper
from caiman_asr_train.train_utils.schedule import Schedule
from caiman_asr_train.utils.frame_width import encoder_output_frame_width
from caiman_asr_train.utils.user_tokens import get_user_token
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


@beartype
@dataclass
class TrainingOnly:
    adjust_lr: Callable
    meta: dict
    checkpointer: Checkpointer
    grad_noise_scheduler: Optional[GradNoiseScheduler]
    optimizer_wrapper: OptimizerWrapper
    train_step_fn: Callable
    dp_scheduler: Schedule
    star_scheduler: Schedule


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
    error_rate: ErrorRate


@beartype
class Setup(ABC):
    def get_model_cls(self, args: Namespace):
        return RNNT

    def run(self, args: Namespace) -> BuiltObjects:
        """Initialise the objects required for each PipelineType."""
        # Must make output directory before the logging starts in initial_setup()
        os.makedirs(args.output_dir, exist_ok=True)
        multi_gpu, world_size, np_rng, cfg, batch_sizes = self.initial_setup(args)
        (
            tokenizers,
            blank_idx,
            eos_idx,
            star_idx,
            tokenizers_kw,
            default_tokenizer,
            default_tokenizer_kw,
        ) = self.build_all_tokenizers(args, cfg)
        model, ema_model, training_only = self.build_model(
            args, cfg, default_tokenizer, multi_gpu, default_tokenizer_kw
        )
        decoder, loss_fn = self.build_evaluation_objects(
            ema_model,
            args,
            cfg,
            blank_idx,
            eos_idx,
            star_idx,
            default_tokenizer,
            world_size,
        )
        data_objects, feat_procs = self.build_data_and_feat_proc(
            args, np_rng, world_size, batch_sizes, cfg, tokenizers, training_only
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
            error_rate=get_error_rate(self.get_default(data_objects).dataset_kw),
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
        Dict[PipelineType, Tokenizer],
        int,
        Optional[int],
        Optional[int],
        Dict[PipelineType, dict],
        Tokenizer,
        dict,
    ]:
        tokenizers, blank_idx, tokenizers_kw = self.build_tokenizer(cfg)

        assert tokenizers, "No tokenizers built"
        eos_idxs = [get_user_token("eos", cfg, tok) for tok in tokenizers.values()]
        assert all(eos_idxs[0] == idx for idx in eos_idxs), "Tokenizers disparate EOS"
        eos_idx = eos_idxs[0]

        if eos_idx is not None:
            assert args.eos_decoding != "none", "EOS decoding strategy not specified!"
        else:
            assert args.eos_decoding == "none", "EOS decoding specified, no EOS token!"
            assert args.eos_alpha == 1.0, "EOS alpha specified but no EOS token!"
            assert args.eos_beta == 0.0, "EOS beta specified but no EOS token!"
            assert not args.eos_is_terminal, "EOS terminal specified but no EOS token!"

        star_idxs = [get_user_token("star", cfg, tok) for tok in tokenizers.values()]
        assert all(star_idxs[0] == idx for idx in star_idxs)
        star_idx = star_idxs[0]

        default_tokenizer = self.get_default(tokenizers)
        default_tokenizer_kw = self.get_default(tokenizers_kw)
        return (
            tokenizers,
            blank_idx,
            eos_idx,
            star_idx,
            tokenizers_kw,
            default_tokenizer,
            default_tokenizer_kw,
        )

    def build_evaluation_objects(
        self,
        ema_model,
        args: Namespace,
        cfg: dict,
        blank_idx: int,
        eos_idx: Optional[int],
        star_idx: Optional[int],
        default_tokenizer: Tokenizer,
        world_size: int,
    ) -> Tuple[RNNTDecoder, ApexTransducerLoss]:
        loss_fn = self.build_loss_fn(blank_idx, eos_idx, star_idx, cfg)
        ngram_info = self.build_ngram_info(args, cfg.get("ngram"))
        decoder = self.build_decoder(
            ema_model,
            args,
            blank_idx,
            eos_idx,
            default_tokenizer,
            ngram_info,
            world_size,
            user_tokens=[i for i in [star_idx, eos_idx] if i is not None],
        )
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
        training_only: Optional[TrainingOnly],
    ) -> Tuple[Dict[PipelineType, DataObject], Dict[PipelineType, torch.nn.Module]]:
        resume_step = self.get_pipeline_resume_step(args, training_only)
        samplers = self.build_samplers(
            args, np_rng, world_size, batch_sizes, resume_step
        )
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
        ema_model,
        args: Namespace,
        blank_idx: int,
        eos_idx: Optional[int],
        tokenizer: Tokenizer,
        ngram_info: Optional[NgramInfo],
        world_size: int,
        user_tokens: List[int],
    ) -> RNNTDecoder:
        if eos_idx is None:
            eos_strategy = None
            assert args.eos_decoding == "none"
        elif args.eos_decoding == "ignore":
            eos_strategy = EOSIgnore(eos_idx=eos_idx)
        elif args.eos_decoding == "blank":
            eos_strategy = EOSBlank(eos_idx=eos_idx)
        elif args.eos_decoding == "predict":
            eos_strategy = EOSPredict(
                eos_idx=eos_idx, alpha=args.eos_alpha, beta=args.eos_beta
            )
        else:
            raise ValueError(f"Unknown EOS decoding strategy: {args.eos_decoding}")

        if args.decoder == "greedy":
            return RNNTBatchedGreedyDecoder(
                model=ema_model,
                blank_idx=blank_idx,
                eos_strategy=eos_strategy,
                max_inputs_per_batch=args.max_inputs_per_batch,
                max_symbol_per_sample=args.max_symbol_per_sample,
                tokenizer=tokenizer,
            )
        else:
            inf = float("inf")

            if args.eos_vad_threshold != inf or args.beam_final_emission_thresh != inf:
                width = encoder_output_frame_width(args.model_config)
            else:
                width = None

            kwargs = {
                "model": ema_model,
                "blank_idx": blank_idx,
                "eos_strategy": eos_strategy,
                "max_inputs_per_batch": args.max_inputs_per_batch,
                "max_symbol_per_sample": args.max_symbol_per_sample,
                "beam_width": args.beam_width,
                "sentpiece_model": tokenizer.sentpiece_model,
                "temperature": args.temperature,
                "beam_prune_score_thresh": args.beam_prune_score_thresh,
                "beam_prune_topk_thresh": args.beam_prune_topk_thresh,
                "ngram_info": ngram_info,
                "fuzzy_topk_logits": args.fuzzy_topk_logits,
                "return_partials": not args.beam_no_partials,
                "user_tokens": user_tokens,
                "eos_is_terminal": args.eos_is_terminal,
                "eos_vad_threshold": args.eos_vad_threshold,
                "final_emission_thresh": args.beam_final_emission_thresh,
                "frame_width": width,
            }

            nprocs = get_num_procs(args, world_size)

            if nprocs > 1:
                return ParallelDecoder(
                    nprocs,
                    args.beam_min_decode_batch_size_per_proc,
                    RNNTBeamDecoder,
                    **kwargs,
                )

            return RNNTBeamDecoder(**kwargs)

    def build_loss_fn(
        self,
        blank_idx: int,
        eos_idx: Optional[int],
        star_idx: Optional[int],
        cfg: dict,
    ) -> ApexTransducerLoss:
        """set up the loss function: if the TransducerJoint has packed_output=True then the
        input to the ApexTransducerLoss must have packed_input=True"""
        rnnt_config = config.rnnt(cfg)
        rnnt_config["gpu_unavailable"] = self.preferred_device() == CPU

        return ApexTransducerLoss(
            blank_idx=blank_idx,
            eos_idx=eos_idx,
            star_idx=star_idx,
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
        user_symbols = list(get_all_user_tokens(cfg).values())
        dali_yaml_config = build_dali_yaml_config(
            config_data=dataset_kw,
            config_features=features_kw,
            user_symbols=user_symbols,
        )
        mel_feat_normalizer = build_mel_feat_normalizer(
            args,
            dali_yaml_config,
            batch_size=batch_sizes[pipeline_type],
            pipeline_type=pipeline_type,
        )
        loader = build_dali_loader(
            args,
            self.pipeline_type_to_str(pipeline_type),
            batch_size=batch_sizes[pipeline_type],
            dali_yaml_config=dali_yaml_config,
            tokenizer=tokenizers[pipeline_type],
            world_size=world_size,
            train_sampler=samplers[pipeline_type],
            mel_feat_normalizer=mel_feat_normalizer,
            cpu=self.preferred_device() == CPU,
        )
        return DataObject(loader=loader, dataset_kw=dataset_kw, features_kw=features_kw)

    def build_each_pipeline_type(self, builder: Callable, *other_args) -> dict:
        """For val, this returns {VAL: builder(VAL)}
        For train, this returns {TRAIN: builder(TRAIN), VAL: builder(VAL)}"""
        return {
            pipeline_type: builder(pipeline_type, *other_args)
            for pipeline_type in self.pipeline_types()
        }

    def build_ngram_info(
        self, args: Namespace, ngram_cfg: Optional[dict]
    ) -> Optional[NgramInfo]:
        """Build n-gram info if beam search is used and n-gram is not disabled."""

        if args.decoder == "beam" and not args.skip_ngram and ngram_cfg is not None:
            ngram_path = args.override_ngram_path or self._find_ngram_file(
                ngram_cfg["ngram_path"]
            )
            return NgramInfo(ngram_path, ngram_cfg["scale_factor"])
        if ngram_cfg is None:
            assert (
                not args.override_ngram_path
            ), "--override_ngram_path specified but no n-gram config found in model config"
        return None

    def _find_ngram_file(self, base_path: str) -> str:
        """Search for ngram file in given directory - if not found, raise error."""
        file = find_ngram_path(base_path)
        if file is None:
            raise FileNotFoundError(
                f"N-gram not found in {base_path}. Ensure you have a valid n-gram, "
                "or pass the `--skip_ngram` argument to disable n-grams during validation."
            )
        return file

    @abstractmethod
    def build_tokenizer(
        self, cfg: dict
    ) -> Tuple[Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict]]:
        pass

    @abstractmethod
    def get_batch_sizes(self, args: Namespace) -> Dict[PipelineType, int]:
        pass

    @abstractmethod
    def get_pipeline_resume_step(
        self, args: Namespace, training_only: Optional[TrainingOnly]
    ) -> Dict[PipelineType, int]:
        pass

    @abstractmethod
    def build_samplers(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        batch_sizes: Dict[PipelineType, int],
        resume_step: Dict[PipelineType, int],
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
