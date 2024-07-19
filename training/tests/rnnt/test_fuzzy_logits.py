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
