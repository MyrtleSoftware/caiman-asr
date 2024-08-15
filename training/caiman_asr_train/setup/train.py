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

from caiman_asr_train.data.dali import sampler as dali_sampler
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.export.model_schema import check_schema_training
from caiman_asr_train.log.tb_dllogger import init_log
from caiman_asr_train.log.tee import start_logging_stdout_and_stderr
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.delay_schedules import (
    ConstantDelayPenalty,
    LinearDelayPenaltyScheduler,
)
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.setup.base import Setup, TrainingOnly
from caiman_asr_train.setup.core import CUDA, TRAIN, VAL, PipelineType
from caiman_asr_train.train_utils.batch_splitting import train_step_batch_split
from caiman_asr_train.train_utils.build_optimizer import build_optimizer
from caiman_asr_train.train_utils.core import train_step
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.train_utils.grad_noise_scheduler import (
    GradNoiseScheduler,
    switch_on_grad_noise_scheduler,
)
from caiman_asr_train.train_utils.lr import lr_policy
from caiman_asr_train.train_utils.optimizer import OptimizerWrapper
from caiman_asr_train.utils.num_weights import num_weights
from caiman_asr_train.utils.seed import set_seed


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
        model_cls = self.get_model_cls(args)
        model = model_cls(n_classes=tokenizer.num_labels + 1, **rnnt_config)
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
        meta = {"best_wer": 10**6, "start_epoch": 1, "step": 1}
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

        assert meta["step"] < args.training_steps, (
            f"Model already trained for steps={meta['step']} and "
            f"training_steps={args.training_steps}. No training to do. Exiting."
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
            dp_scheduler=self.build_delay_penalty_scheduler(args),
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
        bound = (
            None
            if args.grad_scaler_lower_bound_log2 is None
            else 2**args.grad_scaler_lower_bound_log2
        )

        return OptimizerWrapper(args, optimizer, scaler, lower_bound=bound)

    def build_tokenizer(
        self, args: Namespace, cfg: dict
    ) -> Tuple[Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict]]:
        tokenizer_kw = config.tokenizer(cfg)
        tokenizer = Tokenizer(**tokenizer_kw)
        blank_idx = tokenizer.num_labels

        # tokenizer for validation dataset remove sampling effect
        tokenizer_kw_val = config.tokenizer(cfg)
        tokenizer_kw_val["sampling"] = 0.0
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

    def build_delay_penalty_scheduler(
        self, args: Namespace
    ) -> ConstantDelayPenalty | LinearDelayPenaltyScheduler:
        if args.delay_penalty == "linear_schedule":
            return LinearDelayPenaltyScheduler(
                warmup_steps=args.dp_warmup_steps,
                warmup_penalty=args.dp_warmup_penalty,
                ramp_penalty=args.dp_ramp_penalty,
                final_steps=args.dp_final_steps,
                final_penalty=args.dp_final_penalty,
            )
        else:
            return ConstantDelayPenalty(float(args.delay_penalty))

    def get_batch_sizes(self, args: Namespace) -> Dict[PipelineType, int]:
        # Effective batch size per GPU after grad accumulation
        accum_batch_size = int(args.global_batch_size / args.num_gpus)
        train_batch_size = accum_batch_size // args.grad_accumulation_batches
        self.batch_checks(args, train_batch_size, accum_batch_size)
        return {TRAIN: train_batch_size, VAL: args.val_batch_size}

    def get_pipeline_resume_step(
        self, args: Namespace, training_only: Optional[TrainingOnly]
    ) -> Dict[PipelineType, int]:
        if not args.resume:
            resume_training_step = 0
        else:
            resume_training_step = training_only.meta["step"]
        return {TRAIN: resume_training_step, VAL: 0}

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

    @beartype
    def build_samplers(
        self,
        args: Namespace,
        np_rng: np.random.Generator,
        world_size: int,
        batch_sizes: Dict[PipelineType, int],
        resume_step: Dict[PipelineType, int],
    ) -> Dict[PipelineType, Optional[dali_sampler.SimpleSampler]]:
        if args.num_buckets > 0:
            train_sampler = dali_sampler.BucketingSampler(
                num_buckets=args.num_buckets,
                batch_size=batch_sizes[TRAIN],
                num_workers=world_size,
                training_steps=args.training_steps,
                global_batch_size=args.global_batch_size,
                rng=np_rng,
                resume_step=resume_step[TRAIN],
            )
        else:
            train_sampler = dali_sampler.SimpleSampler(world_size=world_size)
        return {TRAIN: train_sampler, VAL: None}

    def pipeline_types(self) -> List[PipelineType]:
        return [TRAIN, VAL]

    def seed_and_logging(self, args: Namespace) -> np.random.Generator:
        # np_rng is used for buckets generation, and needs the same seed on every worker
        np_rng = set_seed(args.seed, args.local_rank)

        # start the logging
        start_logging_stdout_and_stderr(args.output_dir, args.timestamp, "training")
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
