from itertools import repeat, starmap

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


V = TypeVar("V")
R = TypeVar("R")


@beartype
def lmap(fun: Callable[[T], R], xs: Iterable[T]) -> List[R]:
    """
    An alias for `list(map(fun, xs))`.
    """
    return list(map(fun, xs))


@beartype
def lstarmap(fun: Callable, x: Iterable) -> List:
    """
    An alias for `list(starmap(fun, x))`.
    """
    return list(starmap(fun, x))


@beartype
def starmap_zip(fun: Callable, *xs: Iterable) -> Iterable:
    """
    An alias for `starmap(fun, zip(*xs))`.
    """
    return starmap(fun, zip(*xs))


@beartype
def lstarmap_zip(fun: Callable, *xs: Iterable) -> List:
    """
    An alias for `list(starmap_zip(fun,*xs))`.
    """
    return list(starmap_zip(fun, *xs))
