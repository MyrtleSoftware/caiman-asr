import typing

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor as TorchTensor


def bearjax(*args, **kwargs):
    """
    An alias for `jaxtyped` with `beartype` as the typechecker.
    """
    return jaxtyped(typechecker=beartype)(*args, **kwargs)


@bearjax
def jaxtype_factory(name: str, jax_dtype: type, array_type: type) -> type:
    """
    Utility for creating jaxtyping aliases.

    Adapted from: https://github.com/patrick-kidger/jaxtyping/issues/61
    """

    class _BaseArray:
        """
        jaxtyping alias for `array_type` with dtype `jax_dtype`.
        """

        def __new__(cls, *args, **kwargs):
            raise TypeError("Type FArray cannot be instantiated.")

        def __init_subclass__(cls, *args, **kwargs):
            raise TypeError(f"Cannot subclass {cls.__name__}")

        @typing._tp_cache
        def __class_getitem__(cls, params):
            if isinstance(params, str):
                return array_type[jax_dtype, params]
            else:
                raise Exception(
                    f"Unexpected type for params:\n{type(params)=}\n{params=}"
                )

    _BaseArray.__name__ = name

    _BaseArray.__doc__ = _BaseArray.__doc__.format(
        jax_dtype=repr(jax_dtype),
        array_type=repr(array_type),
    )

    return _BaseArray


# This makes linters happy
class FT(TorchTensor):
    @typing._tp_cache
    def __class_getitem__(cls, params):
        raise NotImplementedError()


# This makes linters happy
class IT(TorchTensor):
    @typing._tp_cache
    def __class_getitem__(cls, params):
        raise NotImplementedError()


FT = jaxtype_factory("FloatTensor", TorchTensor, Float)  # noqa

IT = jaxtype_factory("IntTensor", TorchTensor, Int)  # noqa
