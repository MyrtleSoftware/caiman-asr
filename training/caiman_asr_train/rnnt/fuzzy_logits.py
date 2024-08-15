#! /usr/bin/env python3

import torch
from beartype import beartype


@beartype
def get_topk_logits(logits: torch.Tensor, vecs_in_pkt: int = 8, vec_size: int = 32):
    """
    Reduce the logits to match the behavior of caiman-asr solution.

    The caiman-asr accelerated solution does not support a full argmax
    operation, instead it preforms a fuzzy version of this operation.

    The logits are divided into 8 vectors, each of size 32, forming a packet.
    From each packet, the top 32 values are selected from the 8 vectors:
    [max(vec[i] for vec in packet) for i in range(32)].
    These max values are then sent to the CPU, where the argmax operation
    is performed.

    The numbers 8 and 32 are hardcoded in the board, so they are also
    hardcoded here.

    Args:
        logits: the logits tensor
        vecs_in_pkt: number of vectors in a packet
        vec_size: size of each vector

    Returns:
        the reduced logits tensor
    """
    reshape_logits = logits.view(-1, vecs_in_pkt, vec_size).to(device=logits.device)
    n_packets, _, _ = reshape_logits.shape

    # get max values and indices across the vectors in each packet
    max_vals, max_indices = reshape_logits.max(dim=1)

    # get correct indices of the max values across vectors
    reshape_indices = (
        torch.arange(0, logits.size(1), 1)
        .view(n_packets, vecs_in_pkt, vec_size)
        .to(device=logits.device)
    )
    indices = torch.gather(reshape_indices, 1, max_indices.unsqueeze(1)).squeeze(1)

    flat_max_vals = max_vals.flatten()
    flat_indices = indices.flatten()

    # keep minimum to mask reduced values
    _min = torch.min(flat_max_vals).item()

    # generate tensor with logits
    _input = _min * torch.ones_like(logits[0])
    scatter_max_vals = _input.scatter_(0, flat_indices, flat_max_vals)

    return scatter_max_vals.unsqueeze(0)
