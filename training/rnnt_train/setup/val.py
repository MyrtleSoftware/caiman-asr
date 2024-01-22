#!/usr/bin/env python3
import copy
from argparse import Namespace

import numpy as np
import torch
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from torch.nn.parallel import DistributedDataParallel

from rnnt_train.common.data.dali import sampler as dali_sampler
from rnnt_train.common.data.dali.data_loader import DaliDataLoader
from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.helpers import Checkpointer, num_weights, print_once
from rnnt_train.common.seed import set_seed
from rnnt_train.common.tb_dllogger import init_log
from rnnt_train.rnnt import config
from rnnt_train.rnnt.loss import apexTransducerLoss
from rnnt_train.rnnt.model import RNNT
from rnnt_train.setup.base import CPU, CUDA, VAL, PipelineType, Setup


@beartype
class BaseValSetup(Setup):
    """Code shared between val.py and valCPU.py"""

    def build_model(
        self,
        args: Namespace,
        cfg: dict,
        tokenizer: Tokenizer,
        multi_gpu: bool,
        tokenizer_kw: dict,
        world_size: int,
        loss_fn: apexTransducerLoss,
        data_loader: DaliDataLoader,
    ) -> Tuple[Union[RNNT, DistributedDataParallel], RNNT, None]:
        rnnt_config = config.rnnt(cfg)
        rnnt_config["gpu_unavailable"] = self.preferred_device() == CPU
        model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
        model.to(self.preferred_device())

        print_once(f"Model size: {num_weights(model) / 10**6:.1f}M params\n")

        ema_model = copy.deepcopy(model).to(self.preferred_device())

        if multi_gpu:
            model = DistributedDataParallel(model)

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
                f"Using the tokenizer keywords from the config file, which are {tokenizer_kw}"
            )
        else:
            tokenizer_kw = checkpoint_tokenizer_kw
            print_once(
                f"Using the tokenizer keywords from the RNN-T checkpoint, which are {tokenizer_kw}"
            )

        # Initialize Tokenizer w/ tokenizer_kw from checkpoint or from base*.yaml file
        tokenizer = Tokenizer(**tokenizer_kw)
        blank_idx = tokenizer.num_labels
        return ({VAL: tokenizer}, blank_idx, {VAL: tokenizer_kw})

    def get_batch_sizes(self, args: Namespace) -> Dict[PipelineType, int]:
        return {VAL: args.val_batch_size}

    def seed_and_logging(self, args: Namespace) -> np.random.Generator:
        if args.seed is not None:
            np_rng = set_seed(args.seed, args.local_rank)

        if not args.skip_init:
            init_log(args)
        return np_rng

    def build_samplers(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        batch_sizes: Dict[PipelineType, int],
    ) -> Dict[PipelineType, Optional[dali_sampler.SimpleSampler]]:
        return {VAL: None}

    def pipeline_types(self) -> List[PipelineType]:
        return [VAL]


@beartype
class ValSetup(BaseValSetup):
    def preferred_device(self) -> torch.device:
        return CUDA

    def use_torch_dist(self, args: Namespace) -> bool:
        return self.more_than_one_gpu()

    def start_ddp(self, args) -> Tuple[bool, int]:
        multi_gpu = self.use_torch_dist(args)
        # When validation takes a long time (e.g. on a ~40 hour dataset), one
        # process may finish long before the others. If it waits more than
        # `timeout`, it will crash. We increase the default (30 min) to a year
        # so that it never crashes.
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        return self.start_ddp_with_gpu(args, multi_gpu, timeout={"weeks": 52})
