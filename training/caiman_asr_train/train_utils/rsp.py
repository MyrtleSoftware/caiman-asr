#!/usr/bin/env python3
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.
import random
from argparse import Namespace

from beartype import beartype
from beartype.typing import Optional, Tuple

from caiman_asr_train.rnnt.model import StateRNNT
from caiman_asr_train.train_utils.distributed import print_once


def is_random_state_passing_on(batch_concat_list):
    """Checks for non-zero values in the list after the first element"""
    return sum(batch_concat_list[1:]) > 0


def set_rsp_delay_default(args) -> None:
    """
    Set default --rsp_delay if not specified using learning-rate schedule heuristic.

    The heuristic is that RSP should kick in after the learning rate has decayed
    to 1/8 of its initial value (i.e. after x3 half-lives have elapsed).

    This is based on the observation that WER degrades if RSP starts too early in the
    period when the model is learning quickly (during the hold period or soon after this).
    """
    if args.rsp_delay is not None:
        return
    rsp_delay = args.warmup_steps + args.hold_steps + args.half_life_steps * 3

    print_once(
        f"--rsp_delay not set. Setting {rsp_delay=} based on a learning "
        "rate schedule heuristic."
    )
    if args.training_steps < rsp_delay + 5000:
        print_once(
            f"WARNING: Training too short (steps = {args.training_steps}) to see a "
            f"benefit from RSP. Set --training_steps to >= {rsp_delay + 5000}."
        )
    args.rsp_delay = rsp_delay


def rsp_config_checks(args, cfg):
    """Checks that the arguments are valid for random state passing"""
    batch_concat_list = args.rsp_seq_len_freq
    # A list of probabilities should have no negative values
    assert all(x >= 0 for x in batch_concat_list)
    # and at least one positive value
    assert any(x > 0 for x in batch_concat_list)
    if is_random_state_passing_on(batch_concat_list):
        assert cfg["rnnt"]["custom_lstm"], "State passing requires custom_lstm=True"
        assert not cfg["rnnt"][
            "enc_batch_norm"
        ], "State passing hasn't been implemented with batch norm yet"
        assert not cfg["rnnt"][
            "pred_batch_norm"
        ], "State passing hasn't been implemented with batch norm yet"

        set_rsp_delay_default(args)


def generate_batch_history(batch_concat_list):
    """If batch_concat_list is [10,1], returns 1 with probability 10/11
    and 2 with probability 1/11.

    If batch_concat_list is [10,0,1], returns 1 with probability 10/11
    and 3 with probability 1/11.

    """
    return random.choices(range(len(batch_concat_list)), batch_concat_list)[0] + 1


@beartype
def rsp_end_step(
    rnnt_state: Optional[StateRNNT],
    loss_nan: bool,
    step: int,
    args: Namespace,
    batches_until_history_reset: int,
) -> Tuple[Optional[StateRNNT], int, bool]:
    rsp_on = is_random_state_passing_on(args.rsp_seq_len_freq)

    if not loss_nan:
        if rsp_on and step >= args.rsp_delay:
            # In this case the model should have output a non-None state
            assert rnnt_state is not None
    else:
        # Maybe the state is in a bad place and causes NaNs
        rnnt_state = None

    # Initially do not apply random state passing
    if not rsp_on or step < args.rsp_delay:
        rnnt_state = None

    batches_until_history_reset -= 1
    if batches_until_history_reset == 0:
        rnnt_state = None
        batches_until_history_reset = generate_batch_history(args.rsp_seq_len_freq)

    return rnnt_state, batches_until_history_reset, rsp_on and step >= args.rsp_delay
