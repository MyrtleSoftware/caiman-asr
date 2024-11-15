from itertools import repeat

from beartype import beartype
from beartype.typing import Callable, Iterable, List, Sized, TypeVar

T = TypeVar("T")


@beartype
def _true(_) -> bool:
    return True


@beartype
def flat(nested: Iterable[Iterable[T]], *, _if: Callable[[T], bool] = _true) -> List[T]:
    """
    Flatten a nested iterable with an optional filter.
    """
    return [x for xs in nested for x in xs if _if(x)]


U = TypeVar("U")


@beartype
def repeat_like(rep: Iterable[U], *, _as: Iterable[Sized]) -> List[U]:
    """
    Repeat each element of `rep` for the length of
    the corresponding sequence in `_as`.
    """
    return flat(repeat(x, n) for x, n in zip(rep, map(len, _as), strict=True))
