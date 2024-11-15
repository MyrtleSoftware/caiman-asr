#! /usr/bin/env python3

import torch
from beartype import beartype


@beartype
def get_topk_logits(logits: torch.Tensor, vecs_in_pkt: int = 8, vec_size: int = 32):
    """
    Reduce the logits to match the behaviour of caiman-asr solution.

    The caiman-asr accelerated solution does not support a full argmax
    operation, instead it preforms a fuzzy version of this operation.

    The logits are divided into 8 vectors, each of size 32, forming a packet.
    From each packet, the top 32 values are selected from the 8 vectors:
    [max(vec[i] for vec in packet) for i in range(32)].
    These max values are then sent to the CPU, where the argmax operation
    is performed.

    The numbers 8 and 32 are hardcoded in the board, so they are also
    hardcoded here.

    This function effectively broadcasts each batch's minimum value to the
    values that are not the maximum in each vector.

    Args:
        logits: the logits tensor
        vecs_in_pkt: number of vectors in a packet
        vec_size: size of each vector

    Returns:
        the reduced logits tensor
    """

    assert (
        logits.is_contiguous()
    ), "Logits tensor must be contiguous, use '.contiguous()'"

    B, H = logits.shape

    assert H % (vecs_in_pkt * vec_size) == 0, (
        f"Error logits tensor with dimension {H} not divisible"
        f"by the product of {vecs_in_pkt=} and {vec_size=}. Please "
        "choose different arguments."
    )

    reshape_logits = logits.view(B, -1, vecs_in_pkt, vec_size).to(device=logits.device)
    _, n_packets, _, _ = reshape_logits.shape

    # Get max values and indices across the vectors in each packet.
    max_vals, max_indices = reshape_logits.max(dim=2, keepdim=True)

    # Get correct indices of the max values across vectors.
    reshape_indices = (
        torch.arange(0, logits.size(1), 1, device=logits.device).view(
            1, n_packets, vecs_in_pkt, vec_size
        )
    ).expand(B, -1, -1, -1)

    indices = torch.gather(reshape_indices, 2, max_indices).squeeze(2)

    flat_max_vals = max_vals.view(B, -1)
    flat_indices = indices.view(B, -1)

    # Keep minimum to mask reduced values, need contiguous
    # for scatter_ to work.
    _min = torch.min(flat_max_vals, dim=1, keepdim=True).values
    _min = _min.expand(B, H).contiguous()

    # Generate tensor with logits
    scatter_max_vals = _min.scatter_(1, flat_indices, flat_max_vals)

    return scatter_max_vals
