from dataclasses import dataclass

import torch
from beartype import beartype
from beartype.typing import Tuple, Union

from caiman_asr_train.core.pytree import register_dataclass_as_pytree

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@beartype
@dataclass
class _Batch:
    data: torch.Tensor
    lens: torch.Tensor

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data, self.lens

    def pin_memory(self) -> Self:
        return type(self)(self.data.pin_memory(), self.lens.pin_memory())

    def to(
        self, device: Union[str, torch.device, int], non_blocking: bool = False
    ) -> Self:
        return type(self)(
            self.data.to(device=device, non_blocking=non_blocking),
            self.lens.to(device=device, non_blocking=non_blocking),
        )

    def contiguous(self) -> Self:
        return type(self)(self.data.contiguous(), self.lens.contiguous())


@register_dataclass_as_pytree
@beartype
@dataclass
class BatchFirst(_Batch):
    """
    A batch of right-padded tensors with their lengths.

    Members:
        data: (B, T, *)
        lens: (B,)
    """

    def sub_batch(self, lo: int, hi: int, tight: bool = True) -> "BatchFirst":
        if not tight:
            return BatchFirst(self.data[lo:hi], self.lens[lo:hi])

        return BatchFirst(self.data[lo:hi, : self.lens[lo:hi].max()], self.lens[lo:hi])

    @property
    def T(self) -> "TimeFirst":
        return TimeFirst(self.data.transpose(0, 1), self.lens)

    def __post_init__(self):
        assert self.data.shape[0] == self.lens.shape[0], "Batch size mismatch"


@register_dataclass_as_pytree
@beartype
@dataclass
class TimeFirst(_Batch):
    """
    A batch of right-padded tensors with their lengths.

    Members:
        data: (T, B, *)
        lens: (B,)
    """

    def sub_batch(self, lo: int, hi: int, tight: bool = True) -> "TimeFirst":
        if not tight:
            return TimeFirst(self.data[:, lo:hi], self.lens[lo:hi])

        return TimeFirst(self.data[: self.lens[lo:hi].max(), lo:hi], self.lens[lo:hi])

    @property
    def T(self) -> BatchFirst:
        return BatchFirst(self.data.transpose(0, 1), self.lens)

    def __post_init__(self):
        assert self.data.shape[1] == self.lens.shape[0], "Batch size mismatch"
