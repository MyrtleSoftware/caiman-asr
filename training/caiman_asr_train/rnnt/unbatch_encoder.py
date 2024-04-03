#!/usr/bin/env python3
import warnings

import torch
from beartype import beartype
from beartype.typing import Tuple
from einops import pack
from jaxtyping import Float, Int, jaxtyped

from caiman_asr_train.rnnt.model import RNNT


@jaxtyped(typechecker=beartype)
def encode_lower_batch_size(
    model: RNNT,
    feats: Float[torch.Tensor, "seq_len batch audio_feat_dim"],
    feat_lens: Int[torch.Tensor, "batch"],  # noqa: F821
    max_inputs_per_batch: int,
) -> Tuple[
    Float[torch.Tensor, "batch time enc_dim"], Int[torch.Tensor, "batch"]  # noqa: F821
]:
    """Passes a batch of features through the encoder in smaller batches.
    This prevents CUDA OOM errors on long audios (e.g. 1 hour) with the large model"""
    batch_size = feats.size(1)
    activations_in_one_input = feats.size(0) * feats.size(2)
    smaller_batch_size = max_inputs_per_batch // activations_in_one_input
    if smaller_batch_size > batch_size:
        smaller_batch_size = batch_size
    elif smaller_batch_size == 0:
        smaller_batch_size = 1
        if feats.is_cuda:
            warnings.warn(
                f"Based on {max_inputs_per_batch=}, "
                "the GPU is too small to handle even batch size 1. "
                "Trying anyway..."
            )
    results = [
        model.encode(
            feats[:, i : i + smaller_batch_size, :],
            feat_lens[i : i + smaller_batch_size],
        )
        for i in range(0, batch_size, smaller_batch_size)
    ]
    enc_list, enc_lens_list, _ = zip(*results)
    encs, _ = pack(list(enc_list), "* time enc_dim")
    enc_lens, _ = pack(list(enc_lens_list), "*")
    return encs, enc_lens
