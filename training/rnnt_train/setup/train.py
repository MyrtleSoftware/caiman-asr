#!/usr/bin/env python3
import copy
from argparse import Namespace

import numpy as np
import torch
from apex.optimizers import FusedLAMB
from beartype import beartype
from beartype.typing import Callable, Dict, List, Optional, Tuple, Union
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from rnnt_train.common.batch_splitting import train_step_batch_split
from rnnt_train.common.build_optimizer import build_optimizer
from rnnt_train.common.data.dali import sampler as dali_sampler
from rnnt_train.common.data.grad_noise_scheduler import (
    GradNoiseScheduler,
    switch_on_grad_noise_scheduler,
)
from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.helpers import Checkpointer, num_weights, print_once
from rnnt_train.common.optimizers import lr_policy
from rnnt_train.common.seed import set_seed
from rnnt_train.common.tb_dllogger import init_log
from rnnt_train.common.train_aux import train_step
from rnnt_train.rnnt import config
from rnnt_train.rnnt.model import RNNT
from rnnt_train.rnnt.sub_models import RNNTSubModels
from rnnt_train.setup.base import (
    CUDA,
    TRAIN,
    VAL,
    OptimizerWrapper,
    PipelineType,
    Setup,
    TrainingOnly,
)
from rnnt_train.utils.model_schema import check_schema_training


@beartype
class TrainSetup(Setup):
    def build_model(
        self,
        args: Namespace,
        cfg: dict,
        tokenizer: Tokenizer,
        multi_gpu: bool,
        tokenizer_kw: dict,
    ) -> Tuple[Union[RNNT, DDP, RNNTSubModels], RNNT, TrainingOnly]:
        """Build RNNT model and associated training objects such as optimizer"""
        rnnt_config = config.rnnt(cfg)
        if args.weights_init_scale is not None:
            rnnt_config["weights_init_scale"] = args.weights_init_scale
        if args.hidden_hidden_bias_scale is not None:
            rnnt_config["hidden_hidden_bias_scale"] = args.hidden_hidden_bias_scale
        model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
        model.cuda()

        check_schema_training(model.state_dict(), args.skip_state_dict_check)

        print_once(f"Model size: {num_weights(model) / 10**6:.1f}M params\n")

        scaler = None
        if not args.no_amp:
            scaler = GradScaler()

        # optimization
        optimizer = self.build_optimizer(args, model)
        initial_lrs = [group["lr"] for group in optimizer.param_groups]
        print_once(f"Starting with LRs: {initial_lrs}")

        if args.ema > 0:
            ema_model = copy.deepcopy(model).cuda()
        else:
            ema_model = None

        def adjust_lr(step):
            return lr_policy(
                optimizer,
                initial_lrs,
                args.min_lr,
                step,
                args.warmup_steps,
                args.hold_steps,
                half_life_steps=args.half_life_steps,
            )

        model = self.build_dist_model(model, args, multi_gpu)

        # load checkpoint
        meta = {"best_wer": 10**6, "start_epoch": 0, "step": 1}
        checkpointer = Checkpointer(args.output_dir, "RNN-T")

        if args.resume or args.fine_tune:
            assert args.resume ^ args.fine_tune, "cannot both resume and fine_tune"

        if args.ckpt is not None:
            assert (
                args.resume ^ args.fine_tune
            ), "You specified a checkpoint but did not choose resume/fine_tune"

        # when resuming, a specified checkpoint overrules any last checkpoint
        # that may be found
        # when resuming, keep optimizer state and meta info
        if args.resume:
            args.ckpt = args.ckpt or checkpointer.last_checkpoint()
            assert args.ckpt is not None, "no checkpoint to resume from"
            previous_tokenizer_kw = checkpointer.load(
                args.ckpt, model, ema_model, optimizer, meta
            )

        # fine-tuning involves taking a trained model and re-training it after some
        # change in model / data
        # when fine-tuning, a specified checkpoint is expected
        # when fine-tuning, optimizer state and meta info are not kept
        if args.fine_tune:
            assert args.ckpt is not None, "no checkpoint to fine_tune from"
            previous_tokenizer_kw = checkpointer.load(args.ckpt, model, ema_model)

        if args.resume or args.fine_tune:
            if previous_tokenizer_kw is None:
                print_once("WARNING: This is an old RNN-T checkpoint")
                print_once(
                    "Cannot check if you are resuming/fine-tuning using the correct "
                    "tokenizer\n"
                )
            elif previous_tokenizer_kw != tokenizer_kw:
                print_once(
                    f"WARNING: The checkpoint's previous tokenizer keywords were "
                    f"{previous_tokenizer_kw}, but the config file is trying to train "
                    f"using {tokenizer_kw}\n"
                )
        print_once(f"Using the tokenizer keywords {tokenizer_kw}")

        grad_noise_scheduler = self.build_grad_noise_scheduler(args, cfg)

        optimizer_wrapper = self.get_optimizer_wrapper(args, optimizer, scaler)
        training_only = TrainingOnly(
            adjust_lr=adjust_lr,
            meta=meta,
            checkpointer=checkpointer,
            grad_noise_scheduler=grad_noise_scheduler,
            optimizer_wrapper=optimizer_wrapper,
            train_step_fn=self.get_train_step_fn(args),
        )
        return model, ema_model, training_only

    def build_optimizer(self, args: Namespace, model: RNNT) -> FusedLAMB:
        return build_optimizer(args=args, model=model)

    def get_optimizer_wrapper(
        self,
        args: Namespace,
        optimizer,
        scaler: Optional[GradScaler],
    ) -> OptimizerWrapper:
        return OptimizerWrapper(args, optimizer, scaler)

    def modify_dataloader_kw(
        self,
        args: Namespace,
        dataset_kw: dict,
        features_kw: dict,
        pipeline_type: PipelineType,
    ):
        if pipeline_type == TRAIN:
            # if mel stats are being collected these are for use in inference
            # streaming normalization the stats should therefore reflect the processing
            # that will be used in inference
            if args.dump_mel_stats:
                dataset_kw["speed_perturbation"] = None
                dataset_kw["trim_silence"] = False

    def build_tokenizer(
        self, args: Namespace, cfg: dict
    ) -> Tuple[Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict]]:
        tokenizer_kw = config.tokenizer(cfg)
        tokenizer = Tokenizer(**tokenizer_kw)
        blank_idx = tokenizer.num_labels

        tokenizer_kw_val = config.tokenizer(cfg)
        tokenizer_val = Tokenizer(**tokenizer_kw_val)
        return (
            {TRAIN: tokenizer, VAL: tokenizer_val},
            blank_idx,
            {TRAIN: tokenizer_kw, VAL: tokenizer_kw_val},
        )

    def build_dist_model(
        self, model: RNNT, args: Namespace, multi_gpu: bool
    ) -> Union[RNNT, DDP, RNNTSubModels]:
        if args.batch_split_factor > 1:
            return RNNTSubModels.from_RNNT(model, multi_gpu)
        else:
            return DDP(model) if multi_gpu else model

    def build_grad_noise_scheduler(
        self, args: Namespace, cfg: dict
    ) -> Optional[GradNoiseScheduler]:
        # If use of grad noise initiate the grad noise scheduler, otherwise set to None
        rnnt_config = config.rnnt(cfg)
        grad_noise_scheduler = None
        if switch_on_grad_noise_scheduler(
            cfg=cfg, enc_freeze=rnnt_config["enc_freeze"]
        ):
            grad_noise_conf = config.grad_noise_scheduler(cfg)
            grad_noise_conf["seed"] = args.seed
            grad_noise_scheduler = GradNoiseScheduler(**grad_noise_conf)
        return grad_noise_scheduler

    def get_batch_sizes(self, args: Namespace) -> Dict[PipelineType, int]:
        # Effective batch size per GPU after grad accumulation
        accum_batch_size = int(args.global_batch_size / args.num_gpus)
        train_batch_size = accum_batch_size // args.grad_accumulation_batches
        self.batch_checks(args, train_batch_size, accum_batch_size)
        return {TRAIN: train_batch_size, VAL: args.val_batch_size}

    def batch_checks(
        self, args: Namespace, train_batch_size: int, accum_batch_size: int
    ) -> None:
        assert (
            accum_batch_size % args.grad_accumulation_batches == 0
        ), f"{accum_batch_size=} % {args.grad_accumulation_batches=} != 0"

        if args.batch_split_factor != 1:
            assert (
                train_batch_size % args.batch_split_factor == 0
            ), f"{train_batch_size=} % {args.batch_split_factor=} != 0"

    def build_samplers(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        batch_sizes: Dict[PipelineType, int],
    ) -> Dict[PipelineType, Optional[dali_sampler.SimpleSampler]]:
        if args.read_from_tar is None:
            train_sampler = None
        elif args.num_buckets is not None:
            train_sampler = dali_sampler.BucketingSampler(
                args.num_buckets, batch_sizes[TRAIN], world_size, args.epochs, np_rng
            )
        else:
            train_sampler = dali_sampler.SimpleSampler()
        return {TRAIN: train_sampler, VAL: None}

    def pipeline_types(self) -> List[PipelineType]:
        return [TRAIN, VAL]

    def seed_and_logging(self, args: Namespace) -> np.random.Generator:
        np_rng = set_seed(args.seed, args.local_rank)
        # np_rng is used for buckets generation, and needs the same seed on every worker

        init_log(args)
        return np_rng

    def preferred_device(self) -> torch.device:
        return CUDA

    def start_ddp(self, args) -> Tuple[bool, int]:
        # While training, one of the process may finish long before
        # the others. If it waits more than `timeout`, it will crash. We
        # increase the default (30 min) to 4hrs to reduce chance of crashing.
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group # noqa
        return self.start_ddp_with_gpu(args, self.multi_gpu, timeout={"hours": 4})

    def get_train_step_fn(self, args: Namespace) -> Callable:
        return train_step if args.batch_split_factor == 1 else train_step_batch_split
