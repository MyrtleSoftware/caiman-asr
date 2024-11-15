import pytest
import torch
from jaxtyping import TypeCheckError

from caiman_asr_train.utils.type import FT, IT, bearjax


@bearjax
def fun(x: FT[" n"], y: IT[" m"], z: int) -> FT[" n"]:
    return x


@bearjax
def bad_impl(x: FT[" n"]) -> FT["n n"]:
    return x


def test_type_alias():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1, 2, 3], dtype=torch.int32)

    # Test with correct types
    assert torch.all(fun(x, y, 1) == x)

    # Test with bad dtype
    with pytest.raises(TypeCheckError):
        fun(x, x, 1)

    # Test wrong rank
    with pytest.raises(TypeCheckError):
        fun(x[1], y, 2)

    # Test non-tensor
    with pytest.raises(TypeCheckError):
        fun(x, y, 1.0)

    # Test bad return type
    with pytest.raises(TypeCheckError):
        bad_impl(x)
