#!/usr/bin/env python3
import copy
from argparse import Namespace

import numpy as np
import torch
from apex.optimizers import FusedLAMB
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.export.model_schema import check_schema_training
from caiman_asr_train.log.tb_dllogger import init_log
from caiman_asr_train.log.tee import start_logging_stdout_and_stderr
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.config import get_tokenizer_conf
from caiman_asr_train.rnnt.loss import LossSpec, build_loss_fn
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.setup.base import Setup, TrainingOnly
from caiman_asr_train.setup.check_tokenizer import check_tokenizer_kw
from caiman_asr_train.setup.core import CUDA, TRAIN, VAL, PipelineType
from caiman_asr_train.train_utils.build_optimizer import build_optimizer
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.train_utils.grad_noise_scheduler import (
    GradNoiseScheduler,
    switch_on_grad_noise_scheduler,
)
from caiman_asr_train.train_utils.lr import lr_policy
from caiman_asr_train.train_utils.optimizer import OptimizerWrapper
from caiman_asr_train.train_utils.schedule import (
    ConstantSchedule,
    DPSchedule,
    Schedule,
    StepSchedule,
)
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
        spec: LossSpec,
        prefix_token_map: Optional[dict] = None,
    ) -> Tuple[Union[RNNT, DDP, RNNTSubModels], RNNT, TrainingOnly]:
        """Build RNNT model and associated training objects such as optimizer"""
        rnnt_config = config.rnnt(cfg)
        if args.weights_init_scale is not None:
            rnnt_config["weights_init_scale"] = args.weights_init_scale
        if args.hidden_hidden_bias_scale is not None:
            rnnt_config["hidden_hidden_bias_scale"] = args.hidden_hidden_bias_scale
        if rnnt_config.get("conditional_encoder", False):
            assert (
                prefix_token_map is not None
            ), "prefix_token_map must be provided for conditional encoder"
            rnnt_config["prefix_token_map"] = prefix_token_map
        model_cls = self.get_model_cls(args)
        model = model_cls(n_classes=tokenizer.num_labels + 1, **rnnt_config)
        model.cuda()

        check_schema_training(
            model.state_dict(),
            args.skip_state_dict_check,
            rnnt_config.get("conditional_encoder", False),
        )

        print_once(f"Model size: {num_weights(model) / 10**6:.1f}M params\n")

        scaler = None
        if not args.no_amp:
            # growth_interval halved from torch's default of 2000: recovery
            # from a backoff costs +1 log2 per interval of clean steps, so a
            # deep instability at the default pace takes 12-18k steps to
            # regrow to a healthy 2^12-2^15 scale. The cost is at most one
            # extra probe overflow (a skipped step) per 1000 steps.
            scaler = torch.amp.GradScaler("cuda", growth_interval=1000)

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

        model = self.build_dist_model(
            model, args, multi_gpu, spec, rnnt_config["joint_dropout"]
        )

        # load checkpoint
        meta = {"best_wer": float("inf"), "start_epoch": 1, "step": 1, "duration": 0.0}
        checkpointer = Checkpointer(
            args.output_dir, "RNN-T", allow_partial_load=args.allow_partial_checkpoint
        )

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
            checkpoint_tokenizer_kw = checkpointer.load(
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
            checkpoint_tokenizer_kw = checkpointer.load(args.ckpt, model, ema_model)

        if args.resume or args.fine_tune:
            check_tokenizer_kw(checkpoint_tokenizer_kw, tokenizer_kw)

        print_once(f"Using the tokenizer keywords {tokenizer_kw}")

        grad_noise_scheduler = self.build_grad_noise_scheduler(args, cfg)

        optimizer_wrapper = self.get_optimizer_wrapper(args, optimizer, scaler)
        training_only = TrainingOnly(
            adjust_lr=adjust_lr,
            meta=meta,
            checkpointer=checkpointer,
            grad_noise_scheduler=grad_noise_scheduler,
            optimizer_wrapper=optimizer_wrapper,
            dp_scheduler=self.build_delay_penalty_scheduler(args),
            star_scheduler=self.build_star_scheduler(args),
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

        return OptimizerWrapper(
            args,
            optimizer,
            scaler,
            lower_bound=bound,
            norm_skip_factor=args.grad_norm_skip_factor,
            skip_decay_momentum=args.skip_decay_momentum,
        )

    def build_tokenizer(
        self, cfg: dict
    ) -> Tuple[Dict[PipelineType, Tokenizer], int, Dict[PipelineType, dict]]:
        tokenizer_kw = get_tokenizer_conf(cfg)
        tokenizer = Tokenizer(**tokenizer_kw)

        blank_idx = tokenizer.num_labels

        # tokenizer for validation dataset remove sampling effect
        tokenizer_kw_val = get_tokenizer_conf(cfg)
        tokenizer_kw_val["sampling"] = 0.0
        tokenizer_kw_val["label_smoothing_kwargs"] = None
        tokenizer_val = Tokenizer(**tokenizer_kw_val)

        return (
            {TRAIN: tokenizer, VAL: tokenizer_val},
            blank_idx,
            {TRAIN: tokenizer_kw, VAL: tokenizer_kw_val},
        )

    def build_dist_model(
        self,
        model: RNNT,
        args: Namespace,
        multi_gpu: bool,
        spec: LossSpec,
        dropout_prob: float,
    ) -> RNNTSubModels:
        return RNNTSubModels.from_RNNT(
            model, multi_gpu, loss_fn=build_loss_fn(spec), dropout_prob=dropout_prob
        )

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

    def build_star_scheduler(self, args: Namespace) -> StepSchedule:
        return StepSchedule(
            initial_value=args.star_initial_value,
            final_value=args.star_final_value,
            toggle_step=args.star_toggle_step,
            wer_threshold=args.star_wer_threshold,
        )

    def build_delay_penalty_scheduler(self, args: Namespace) -> Schedule:
        if args.delay_penalty == "wer_schedule":
            return DPSchedule(
                initial_value=args.dp_initial_value,
                final_value=args.dp_final_value,
                wer_threshold_initial=args.dp_wer_threshold_initial,
                wer_threshold_final=args.dp_wer_threshold_final,
                interpolation_steps=args.val_frequency,
            )
        else:
            return ConstantSchedule(float(args.delay_penalty))

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
