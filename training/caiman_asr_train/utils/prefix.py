#!/usr/bin/env python3
import torch
from beartype import beartype
from beartype.typing import List
from einops import repeat

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.lhotse.text import _get_user_token


@beartype
def build_prefix_token_map(
    user_tokens: dict[str, str],
    tokenizer: Tokenizer,
) -> dict[int, int]:
    """Build a mapping from PnC/Lang prefix token IDs to sequential integers."""

    if not {"pnc", "nopnc"}.issubset(user_tokens):
        raise ValueError("'pnc' or 'nopnc' missing from user_tokens.")

    ordered_tokens = [
        _get_user_token(user_tokens["pnc"], tokenizer),
        _get_user_token(user_tokens["nopnc"], tokenizer),
    ]
    if ordered_tokens[0] == ordered_tokens[1]:
        raise ValueError("pnc and nopnc tokens must be unique.")

    lang_toks = [
        _get_user_token(v, tokenizer)
        for k, v in user_tokens.items()
        if k.startswith("lang_")
    ]
    ordered_tokens.extend(lang_toks)

    return {tok: idx for idx, tok in enumerate(ordered_tokens)}


@beartype
def broadcast_prefix(
    prefix_token_ids: List[int],
    batch_size: int,
    device: torch.device,
) -> TimeFirst:
    """
    Broadcast a sequence of prefix token IDs into a TimeFirst batch.

    Args:
        prefix_token_ids: List of integer IDs to replicate.
        batch_size: Number of utterances in the batch.
        device: Target torch device.

    Returns:
        A TimeFirst batch containing the broadcasted prefix and uniform lengths.
    """
    p_tensor = torch.tensor(prefix_token_ids, device=device)

    return TimeFirst(
        data=repeat(p_tensor, "p -> p b", b=batch_size),
        lens=torch.full(
            (batch_size,), p_tensor.size(0), device=device, dtype=torch.int32
        ),
    )
