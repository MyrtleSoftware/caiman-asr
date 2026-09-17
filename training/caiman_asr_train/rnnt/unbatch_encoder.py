#!/usr/bin/env python3
import warnings

import torch
from beartype import beartype
from beartype.typing import List, Optional
from jaxtyping import jaxtyped

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.utils.prefix import broadcast_prefix


@jaxtyped(typechecker=beartype)
def encode_lower_batch_size(
    model: RNNT,
    feats: TimeFirst,
    max_inputs_per_batch: int,
    prefix_token_ids: Optional[List[int]],
) -> TimeFirst:
    """
    Passes a batch of features through the encoder in smaller batches. This
    prevents CUDA OOM errors on long audios (e.g. 1 hour) with the large model
    """
    B = feats.data.size(1)
    activations_in_one_input = feats.data.size(0) * feats.data.size(2)
    b = max_inputs_per_batch // activations_in_one_input
    if b > B:
        b = B
    elif b == 0:
        b = 1
        if feats.data.is_cuda:
            warnings.warn(
                f"Based on {max_inputs_per_batch=}, "
                "the GPU is too small to handle even batch size 1. "
                "Trying anyway..."
            )

    if prefix_token_ids:
        prefix = broadcast_prefix(prefix_token_ids, B, feats.data.device)
    else:
        prefix = None

    results = [
        model.encode(
            state=None,
            feats=feats.sub_batch(i, i + b),
            prefix=None if prefix is None else prefix.sub_batch(i, i + b),
        )[0]
        for i in range(0, B, b)
    ]

    encs = torch.nn.utils.rnn.pad_sequence(
        x.sub_batch(i, i + 1).data[:, 0] for x in results for i in range(x.data.size(1))
    )

    enc_lens = torch.cat([x.lens for x in results], dim=0)

    return TimeFirst(data=encs, lens=enc_lens)
