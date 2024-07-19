#!/usr/bin/env python3
import copy
from argparse import Namespace

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.data.dali import sampler as dali_sampler
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.log.tb_dllogger import init_log
from caiman_asr_train.log.tee import start_logging_stdout_and_stderr
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.setup.base import Setup, TrainingOnly
from caiman_asr_train.setup.core import CPU, CUDA, VAL, PipelineType
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.utils.num_weights import num_weights
from caiman_asr_train.utils.seed import set_seed


@beartype
class BaseValSetup(Setup):
    """Code shared between ValSetup and ValCPUSetup."""

    def build_model(
        self,
        args: Namespace,
        cfg: dict,
        tokenizer: Tokenizer,
        multi_gpu: bool,
        tokenizer_kw: dict,
    ) -> Tuple[Union[RNNT, DDP], RNNT, None]:
        rnnt_config = config.rnnt(cfg)
        rnnt_config["gpu_unavailable"] = self.preferred_device() == CPU
        model_cls = self.get_model_cls(args)
        model = model_cls(n_classes=tokenizer.num_labels + 1, **rnnt_config)
        model.to(self.preferred_device())

        print_once(f"Model size: {num_weights(model) / 10**6:.1f}M params\n")

        ema_model = copy.deepcopy(model).to(self.preferred_device())

        if multi_gpu:
            model = DDP(model)

        # setup checkpointer
        checkpointer = Checkpointer(args.output_dir, "RNN-T")

        # load checkpoint (modified to not need optimizer / meta args)
        checkpointer.load(args.ckpt, model, ema_model)

        return model, ema_model, None

    def build_tokenizer(
        self, args: Namespace, cfg: dict
    ) -> Tuple[Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict]]:
        checkpoint_tokenizer_kw = torch.load(args.ckpt, map_location=CPU).get(
            "tokenizer_kw"
        )
        tokenizer_kw = config.tokenizer(cfg)

        if checkpoint_tokenizer_kw is None:
            print_once("WARNING: This is an old RNN-T checkpoint")
            print_once("Cannot check if you are using the correct tokenizer\n")
            print_once(
                "Using the tokenizer keywords from the config file, which are "
                f"{tokenizer_kw}"
            )
        else:
            tokenizer_kw = checkpoint_tokenizer_kw
            print_once(
                "Using the tokenizer keywords from the RNN-T checkpoint, which are "
                f"{tokenizer_kw}"
            )

        # Notify user the sampling is off during evaluation
        # The default value is here for checkpoints that have a saved
        # checkpoint_tokenizer_kw without the key "sampling".
        tkn_samp = tokenizer_kw.get("sampling", 0.0)
        if float(tkn_samp) > 0:
            print_once("Please note that sampling is off during evaluation.")
        tokenizer_kw["sampling"] = 0.0

        # Initialize Tokenizer w/ tokenizer_kw from checkpoint or from base*.yaml file
        tokenizer = Tokenizer(**tokenizer_kw)
        blank_idx = tokenizer.num_labels
        return ({VAL: tokenizer}, blank_idx, {VAL: tokenizer_kw})

    def get_batch_sizes(self, args: Namespace) -> Dict[PipelineType, int]:
        return {VAL: args.val_batch_size}

    def get_pipeline_resume_step(
        self, args: Namespace, training_only: Optional[TrainingOnly] = None
    ) -> Dict[PipelineType, int]:
        return {VAL: 0}

    def seed_and_logging(self, args: Namespace) -> np.random.Generator:
        np_rng = set_seed(args.seed, args.local_rank)

        # start the logging
        start_logging_stdout_and_stderr(args.output_dir, args.timestamp, "validation")
        if not args.skip_init:
            init_log(args)
        return np_rng

    def build_samplers(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        batch_sizes: Dict[PipelineType, int],
        resume_step: Dict[PipelineType, int],
    ) -> Dict[PipelineType, Optional[dali_sampler.SimpleSampler]]:
        return {VAL: None}

    def pipeline_types(self) -> List[PipelineType]:
        return [VAL]


@beartype
class ValSetup(BaseValSetup):
    def preferred_device(self) -> torch.device:
        return CUDA

    def start_ddp(self, args) -> Tuple[bool, int]:
        # When validation takes a long time (e.g. on a ~40 hour dataset), one
        # process may finish long before the others. If it waits more than
        # `timeout`, it will crash. Increasing the default (30 min) to a year
        # so that it never crashes.
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group # noqa
        return self.start_ddp_with_gpu(args, self.multi_gpu, timeout={"weeks": 52})


@beartype
class ValCPUSetup(BaseValSetup):
    def preferred_device(self) -> torch.device:
        return CPU

    def start_ddp(self, args) -> Tuple[bool, int]:
        return False, 1

    @property
    def multi_gpu(self) -> bool:
        return False
