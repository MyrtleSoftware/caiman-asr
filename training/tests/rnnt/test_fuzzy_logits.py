import pytest
import torch

from caiman_asr_train.rnnt.decoder import get_topk_logits


@pytest.mark.parametrize(
    "logits, vecs_in_pkt, vec_size, expected",
    [
        (
            torch.tensor([[0.5, 0.3, 0.2, 0.7, 0.1, 0.9]]),
            2,
            3,
            torch.tensor([[0.3, 0.3, 0.3, 0.7, 0.3, 0.9]]),
        ),
        (
            torch.tensor([[0.5, 0.3, 0.2, 0.7, 0.2, 0.9, 0.1, 0.1, 0.1]]),
            3,
            3,
            torch.tensor([[0.3, 0.3, 0.3, 0.7, 0.3, 0.9, 0.3, 0.3, 0.3]]),
        ),
        (
            torch.tensor(
                [
                    [
                        [0.75, 0.13, 0.85],
                        [0.68, 0.49, 0.79],
                        [0.93, 0.74, 0.44],
                        [0.41, 0.69, 0.19],
                    ],
                    [
                        [0.09, 0.01, 0.82],
                        [0.23, 0.41, 0.35],
                        [0.60, 0.47, 0.42],
                        [0.09, 0.94, 0.88],
                    ],
                    [
                        [0.42, 0.97, 0.01],
                        [9.80, 9.61, 0.00],
                        [0.63, 0.48, 0.00],
                        [0.37, 0.64, 0.00],
                    ],
                ]
            )
            .view(-1)
            .unsqueeze(0),
            4,
            3,
            torch.tensor(
                [
                    [
                        [0.01, 0.01, 0.85],
                        [0.01, 0.01, 0.01],
                        [0.93, 0.74, 0.01],
                        [0.01, 0.01, 0.01],
                    ],
                    [
                        [0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.01],
                        [0.60, 0.01, 0.01],
                        [0.01, 0.94, 0.88],
                    ],
                    [
                        [0.01, 0.01, 0.01],
                        [9.80, 9.61, 0.01],
                        [0.01, 0.01, 0.01],
                        [0.01, 0.01, 0.01],
                    ],
                ]
            )
            .view(-1)
            .unsqueeze(0),
        ),
    ],
)
def test_get_topk_logits(logits, vecs_in_pkt, vec_size, expected):
    logits = get_topk_logits(logits, vecs_in_pkt, vec_size)
    assert torch.allclose(logits, expected)


def legacy_get_topk_logits(
    logits: torch.Tensor, vecs_in_pkt: int = 8, vec_size: int = 32
):
    """
    This is the old unbatched version here to check the
    correctness of the new batched version.
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


@pytest.mark.parametrize("batch", [1, 2, 7])
@pytest.mark.parametrize("elements", [1024, 2048])
def test_batched(batch, elements):
    logits = torch.randn(batch, elements)

    batched = get_topk_logits(logits)

    for i in range(batch):
        assert torch.allclose(batched[None, i], legacy_get_topk_logits(logits[None, i]))
