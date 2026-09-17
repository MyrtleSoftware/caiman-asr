#!/usr/bin/env python3

import os
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Callable, Dict, List, Optional, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.evaluate.error_rates import ErrorRate, get_error_rate
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.lhotse.config import DataConfig
from caiman_asr_train.lhotse.config_build import build_from_legacy_sources
from caiman_asr_train.lhotse.pipeline import LhotsePipeline, ResumeInfo
from caiman_asr_train.lm.kenlm_ngram import NgramInfo, find_ngram_path
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.batched_greedy import RNNTBatchedGreedyDecoder
from caiman_asr_train.rnnt.beam import RNNTBeamDecoder
from caiman_asr_train.rnnt.decoder import RNNTDecoder
from caiman_asr_train.rnnt.eos_strategy import EOSBlank, EOSIgnore, EOSPredict
from caiman_asr_train.rnnt.joint_loss import JointLoss, build_joint_loss
from caiman_asr_train.rnnt.loss import LossSpec, build_loss_fn, prewarm_loss_kernels
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.parallel_decoder import ParallelDecoder, get_num_procs
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.setup.core import CPU, CUDA, TRAIN, VAL, PipelineType
from caiman_asr_train.train_utils.distributed import (
    print_once,
)
from caiman_asr_train.train_utils.distributed import (
    start_ddp_with_gpu as shared_start_ddp_with_gpu,
)
from caiman_asr_train.train_utils.grad_noise_scheduler import GradNoiseScheduler
from caiman_asr_train.train_utils.optimizer import OptimizerWrapper
from caiman_asr_train.train_utils.schedule import Schedule
from caiman_asr_train.utils.frame_width import encoder_output_frame_width
from caiman_asr_train.utils.prefix import build_prefix_token_map
from caiman_asr_train.utils.user_tokens import get_user_token
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens

VAL_DTYPE = torch.float32


@beartype
@dataclass
class TrainingOnly:
    adjust_lr: Callable
    meta: dict
    checkpointer: Checkpointer
    grad_noise_scheduler: Optional[GradNoiseScheduler]
    optimizer_wrapper: OptimizerWrapper
    dp_scheduler: Schedule
    star_scheduler: Schedule


@beartype
@dataclass
class DataObject:
    loader: LhotsePipeline
    config: DataConfig
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
    val_joint_loss: JointLoss
    multi_gpu: bool
    cfg: dict
    world_size: int
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
        multi_gpu, world_size, np_rng, cfg = self.initial_setup(args)
        (
            tokenizers,
            blank_idx,
            eos_idx,
            star_idx,
            tokenizers_kw,
            default_tokenizer,
            default_tokenizer_kw,
        ) = self.build_all_tokenizers(args, cfg)
        if cfg["rnnt"].get("conditional_encoder", False):
            has_prefix_conditioning = any(
                (
                    args.pnc_prefix is not None,
                    args.val_lang_prefix is not None,
                    getattr(args, "conditional", False),
                    getattr(args, "conditional_language", False),
                )
            )
            assert has_prefix_conditioning, "Conditional encoder enabled but "
            "no prefix specified, check the train / val arguments."
            user_tokens, _ = get_all_user_tokens(cfg)
            prefix_token_map = build_prefix_token_map(
                user_tokens=user_tokens,
                tokenizer=default_tokenizer,
            )
        else:
            prefix_token_map = None
        spec = self.build_loss_spec(args, blank_idx, eos_idx, star_idx, cfg)
        prewarm_loss_kernels(spec, (spec.dtype, VAL_DTYPE))
        model, ema_model, training_only = self.build_model(
            args,
            cfg,
            default_tokenizer,
            multi_gpu,
            default_tokenizer_kw,
            spec,
            prefix_token_map=prefix_token_map,
        )
        decoder = self.build_evaluation_objects(
            ema_model,
            args,
            cfg,
            blank_idx,
            eos_idx,
            star_idx,
            default_tokenizer,
            world_size,
        )
        data_objects = self.build_data_and_feat_proc(
            args, np_rng, world_size, cfg, tokenizers, training_only
        )
        return BuiltObjects(
            decoder=decoder,
            model=model,
            tokenizers=tokenizers,
            val_joint_loss=build_joint_loss(
                ema_model,
                build_loss_fn(replace(spec, dtype=VAL_DTYPE)),
                float(config.rnnt(cfg)["joint_dropout"]),
            ).eval(),
            multi_gpu=multi_gpu,
            cfg=cfg,
            data_objects=data_objects,
            ema_model=ema_model,
            world_size=world_size,
            tokenizers_kw=tokenizers_kw,
            training_only=training_only,
            error_rate=get_error_rate(self.get_default(data_objects).dataset_kw),
        )

    def initial_setup(
        self, args: Namespace
    ) -> Tuple[bool, int, np.random.Generator, dict]:
        self.torch_settings(args)
        multi_gpu, world_size = self.start_ddp(args)
        np_rng = self.seed_and_logging(args)
        cfg = self.build_cfg(args)
        return multi_gpu, world_size, np_rng, cfg

    def build_all_tokenizers(self, args: Namespace, cfg: dict) -> Tuple[
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
    ) -> RNNTDecoder:
        ngram_info = self.build_ngram_info(args, cfg.get("ngram"))
        user_symbols_dict, _ = get_all_user_tokens(cfg)
        return self.build_decoder(
            ema_model,
            args,
            blank_idx,
            eos_idx,
            default_tokenizer,
            ngram_info,
            world_size,
            user_tokens=[i for i in [star_idx, eos_idx] if i is not None],
            user_symbols_dict=user_symbols_dict,
        )

    @abstractmethod
    def build_model(
        self,
        args: Namespace,
        cfg: dict,
        tokenizer: Tokenizer,
        multi_gpu: bool,
        tokenizer_kw: dict,
        spec: LossSpec,
        prefix_token_map: Optional[dict],
    ) -> Tuple[Union[RNNT, DDP, RNNTSubModels], RNNT, Optional[TrainingOnly]]:
        pass

    def build_data_and_feat_proc(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        cfg: dict,
        tokenizers: Dict[PipelineType, Tokenizer],
        training_only: Optional[TrainingOnly],
    ) -> Dict[PipelineType, DataObject]:
        return self.build_each_pipeline_type(
            self.build_data_object,
            args,
            cfg,
            tokenizers,
            world_size,
            training_only,
        )

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
        user_symbols_dict: Dict[str, str],
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

        def _ids(prefix):
            return self._get_prefix_token(prefix, user_symbols_dict, tokenizer)

        match args.pnc_prefix, args.val_lang_prefix:
            case None, None:
                prefix_token_ids = None
            case str(pnc), None:
                prefix_token_ids = _ids(pnc)
            case None, str(lang):
                prefix_token_ids = _ids(lang)
            case str(pnc), str(lang):
                prefix_token_ids = _ids(pnc) + _ids(lang)
            case pnc_prefix, val_lang_prefix:
                raise ValueError(
                    f"Unknown prefix combination: {pnc_prefix}, {val_lang_prefix}"
                )

        if args.decoder == "greedy":
            keyword_boost_context_score = (
                args.keyword_boost_context_score
                if args.keyword_boost_context_score is not None
                else 0.4
            )

            return RNNTBatchedGreedyDecoder(
                model=ema_model,
                blank_idx=blank_idx,
                eos_strategy=eos_strategy,
                max_inputs_per_batch=args.max_inputs_per_batch,
                max_symbol_per_sample=args.max_symbol_per_sample,
                tokenizer=tokenizer,
                keywords_path=args.keywords_path,
                boost_keywords=args.boost_keywords,
                keyword_boost_c0=args.keyword_boost_c0,
                keyword_boost_beta=args.keyword_boost_beta,
                keyword_boost_context_score=keyword_boost_context_score,
                ngram_info=ngram_info,
                top_k=args.top_k,
                prefix_token_ids=prefix_token_ids if prefix_token_ids else None,
            )
        else:
            inf = float("inf")

            if args.eos_vad_threshold != inf or args.beam_final_emission_thresh != inf:
                width = encoder_output_frame_width(args.model_config)
            else:
                width = None

            keyword_boost_context_score = (
                args.keyword_boost_context_score
                if args.keyword_boost_context_score is not None
                else 1.0
            )

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
                "keywords_path": args.keywords_path,
                "boost_keywords": args.boost_keywords,
                "keyword_boost_c0": args.keyword_boost_c0,
                "keyword_boost_beta": args.keyword_boost_beta,
                "keyword_boost_context_score": keyword_boost_context_score,
                "frame_width": width,
                "prefix_token_ids": prefix_token_ids,
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

    def build_loss_spec(
        self,
        args: Namespace,
        blank_idx: int,
        eos_idx: Optional[int],
        star_idx: Optional[int],
        cfg: dict,
    ) -> LossSpec:
        rnnt_config = config.rnnt(cfg)
        rnnt_config["gpu_unavailable"] = self.preferred_device() == CPU

        return LossSpec(
            backend=args.transducer_backend,
            blank_idx=blank_idx,
            eos_idx=eos_idx,
            star_idx=star_idx,
            packed_input=rnnt_config["joint_apex_transducer"] == "pack",
            joint_n_hid=rnnt_config["joint_n_hid"],
            device=self.preferred_device(),
            dtype=torch.float32 if getattr(args, "no_amp", True) else torch.float16,
            dp_mu=getattr(args, "dp_mu", 0.0),
            dp_clamp=getattr(args, "dp_clamp", None),
        )

    def build_cfg(self, args: Namespace) -> dict:
        cfg = config.load(args.model_config)
        return cfg

    def pipeline_type_to_str(self, pipeline_type: PipelineType) -> str:
        return {TRAIN: "train", VAL: "val"}[pipeline_type]

    def build_data_object(
        self,
        pipeline_type: PipelineType,
        args: Namespace,
        cfg: dict,
        tokenizers: Dict[PipelineType, Tokenizer],
        world_size: int,
        training_only: Optional[TrainingOnly],
    ) -> DataObject:
        print_once("Setting up datasets...")

        if pipeline_type == VAL and "loader_val_device" in args:
            use_cuda = args.loader_val_device == "gpu"
        elif pipeline_type == TRAIN and "loader_train_device" in args:
            use_cuda = args.loader_train_device == "gpu"
        else:
            use_cuda = self.preferred_device() == CUDA

        global_rank = dist.get_rank() if dist.is_initialized() else 0

        conf, dataset_kw, features_kw = build_from_legacy_sources(
            args,
            cfg,
            self.pipeline_type_to_str(pipeline_type),
            use_cuda=use_cuda,
            global_rank=global_rank,
            world_size=world_size,
        )

        if training_only is None:
            resume_info = ResumeInfo()
        else:
            resume_info = ResumeInfo(
                # Divide by world size because each GPU would have only seen
                # this fraction of the total training duration. Multiply by 3600
                # as checkpoint stores duration in hours.
                total_dur=training_only.meta["duration"] * 3600 / world_size,
                # It is not possible to know how many iters were done because
                # we don't know how many iters were used, this is currently
                # only used for seeding so just use steps as a proxy for iters
                total_its=training_only.meta["step"],
            )

        loader = LhotsePipeline(
            config=conf,
            global_rank=global_rank,
            world_size=world_size,
            tokenizer=tokenizers[pipeline_type],
            resume_info=resume_info,
        )

        return DataObject(
            loader=loader,
            config=conf,
            dataset_kw=dataset_kw,
            features_kw=features_kw,
        )

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
        """Build n-gram info if n-gram is not disabled."""

        # n-gram is disabled for greedy decoding by default
        if args.decoder == "greedy" and args.skip_ngram is None:
            args.skip_ngram = True

        if not args.skip_ngram and ngram_cfg is not None:
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

    def _get_prefix_token(
        self, prefix: str, user_symbols_dict: Dict[str, str], tokenizer: Tokenizer
    ) -> List[int]:
        """Get the token IDs for the prefix."""
        assert prefix in user_symbols_dict, (
            f"`{prefix}` token not found in user_symbols_dict. "
            "Add it to `user_symbols` in the model config (.yaml)."
        )
        tokens = tokenizer.tokenize(user_symbols_dict[prefix])
        if tokens:
            # 2 tokens: space token + control token
            assert (
                len(tokens) == 2
            ), f"Expected 2 tokens for {prefix}, got {len(tokens)}"
            return [tokens[1]]
        else:
            raise ValueError(f"No {prefix} token found in tokenizer")

    @abstractmethod
    def build_tokenizer(
        self, cfg: dict
    ) -> Tuple[Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict]]:
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
        return shared_start_ddp_with_gpu(
            multi_gpu,
            args.local_rank,
            getattr(args, "skip_init", False),
            timeout,
        )

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
