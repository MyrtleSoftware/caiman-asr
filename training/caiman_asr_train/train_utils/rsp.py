#!/usr/bin/env python3
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.
import random
from argparse import Namespace

import torch
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Float, Int, jaxtyped

from caiman_asr_train.rnnt.state import PredNetState, RNNTState
from caiman_asr_train.train_utils.distributed import print_once

# flake8: noqa


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
        f"--rsp_delay not set. Setting {rsp_delay=} based on a learning rate schedule heuristic."
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
    rnnt_state: Optional[RNNTState],
    loss_nan: bool,
    step: int,
    args: Namespace,
    batches_until_history_reset: int,
) -> Tuple[Optional[RNNTState], int, bool]:
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


@jaxtyped(typechecker=beartype)
def get_last_nonpadded_states(
    all_hid: Tuple[
        Float[torch.Tensor, "layers seq batch hidden"],
        Float[torch.Tensor, "layers seq batch hidden"],
    ],
    lens: Int[torch.Tensor, "batch"],
    how_far_back: int = 0,
) -> Tuple[
    Float[torch.Tensor, "layers batch hidden"],
    Float[torch.Tensor, "layers batch hidden"],
]:
    """The last hidden state for each member of the batch has to be picked.
    This is not the last state in the tensor, since there's padding.
    Hence index using lens to get the last non-padded state."""
    return (
        all_hid[0][:, lens - 1 - how_far_back, range(len(lens)), :],
        all_hid[1][:, lens - 1 - how_far_back, range(len(lens)), :],
    )


def maybe_get_last_nonpadded(all_hid, lens):
    return None if all_hid is None else get_last_nonpadded_states(all_hid, lens)


@jaxtyped(typechecker=beartype)
def get_pred_net_state(
    y: Int[torch.Tensor, "batch seq"],
    all_pred_hid: Optional[
        Tuple[
            Float[torch.Tensor, "layers seq+1 batch hidden"],
            Float[torch.Tensor, "layers seq+1 batch hidden"],
        ]
    ],
    y_lens: Int[torch.Tensor, "batch"],
    g_lens: Int[torch.Tensor, "batch"],
):
    """Returns the PredNetState representing the pred net's internal state at
    the end of an utterance. The PredNetState consists of the last token and a
    LSTM state.

    Unintuitively: set how_far_back=1 to get the next-to-last LSTM state.
    Justification: Consider two utterances, with token sequences [I,
    like, cats] and [You, love, dogs]

    By training on the concatenation of these two utterances,
    the pred net calculation looks like:

    (A)
    SOS  I like cats You love dogs
       ↘  ↘    ↘    ↘   ↘    ↘   ↘
     0 ➡h1➡h2  ➡h3  ➡h4 ➡h5  ➡h6 ➡h7

    Here 0 is the zero vector which LSTMs use as the initial hidden
    state if you don't pass in a hidden state.

    By training on these utterances individually, the pred net
    calculations are:

    (B)
    SOS  I like cats
       ↘  ↘    ↘    ↘
     0 ➡h1➡h2  ➡h3  ➡h4

    (C)
    SOS You love dogs
       ↘   ↘    ↘    ↘
     0 ➡h1 ➡H2  ➡H3  ➡H4
    (where H_i is a different state from h_i)

    Note that for the second utterance the hidden states are different
    than in A, because the context of the first utterance is lost.
    This can be fixed by replacing (SOS, 0) in C with (cats, h3) from B:

    (D)
    cats You love dogs
        ↘   ↘    ↘    ↘
     h3 ➡h4 ➡h5  ➡h6  ➡h7

    Hence, D has the same hidden states as A since
    combining the tokens and states is done the same way as in A.

    Thus, it's required to save the last token (e.g. cats) and the
    next-to-last state (e.g. h3) to make this work.

    """
    if all_pred_hid is None:
        return None
    # Pick out the last token from each label sequence, using y_lens to index
    # because of the padding.
    last_tokens = torch.unsqueeze(y[range(len(y_lens)), y_lens - 1], 1)

    next_to_last_staggered_pred_hid = get_last_nonpadded_states(
        all_pred_hid, g_lens, how_far_back=1
    )
    return PredNetState(
        next_to_last_pred_state=next_to_last_staggered_pred_hid,
        last_token=last_tokens,
    )
