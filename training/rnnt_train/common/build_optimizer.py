#!/usr/bin/env python3
from argparse import Namespace

from apex.optimizers import FusedLAMB
from beartype import beartype

from rnnt_train.rnnt.model import RNNT


@beartype
def build_fused_lamb(args: Namespace, model: RNNT, opt_eps: float) -> FusedLAMB:
    """
    Build a FusedLAMB optimizer for the given model.
    """
    kw = {
        "params": model.param_groups(args.lr),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    return FusedLAMB(
        betas=(args.beta1, args.beta2), eps=opt_eps, max_grad_norm=args.clip_norm, **kw
    )


@beartype
def build_optimizer(args: Namespace, model: RNNT, world_size: int) -> FusedLAMB:
    """
    Top-level optimizer builder.
    """
    opt_eps = 1e-9
    return build_fused_lamb(args, model, opt_eps)
