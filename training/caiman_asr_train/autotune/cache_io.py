"""
Process-safe cache I/O, shared by the autotune per-config bench cache (``sweep.py``) and
the tree-model bundle (``tree_model.core.train`` / ``tree_model.core.prune``).

Everything lives under ONE cache root (``_cache_root``); callers pass paths RELATIVE to it
and never compute the root themselves -- so the bench timings, the trained bundle, and
tilelang's own compiled binaries all sit under the same dir and persist (or are wiped)
together. Both helpers honor the cache-disable env levers, so a disabled cache is a clean
no-op on write and a clean miss on read.
"""

import contextlib
import os
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, TypeVar

from beartype import beartype

T = TypeVar("T")


@beartype
def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


_DISABLE_ENV = ("TILELANG_DISABLE_CACHE", "CAIMAN_AUTOTUNE_DISABLE_CACHE")


@beartype
def _cache_disabled() -> bool:
    """
    Honor tilelang's global disable (so test harnesses that already set it get a clean
    slate) plus a dedicated lever for the caiman caches alone.
    """
    return any(map(_truthy_env, _DISABLE_ENV))


@beartype
def _cache_root() -> Path:
    """
    The tilelang cache root (``$TILELANG_CACHE_DIR`` or ``~/.tilelang/cache``).
    """
    return Path(os.environ.get("TILELANG_CACHE_DIR", "~/.tilelang/cache")).expanduser()


@beartype
def cache_write(relpath: str | os.PathLike, write: Callable[[str], None]) -> None:
    """
    Atomically publish a cache file at ``_cache_root() / relpath``:
    ``write(tmp)``. A no-op if the cache is disabled.
    """
    if _cache_disabled():
        return

    path = os.fspath(_cache_root() / relpath)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )

    os.close(fd)

    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


@beartype
def cache_read(relpath: str | os.PathLike, load: Callable[[BinaryIO], T]) -> T | None:
    """
    Read a cache file at ``_cache_root() / relpath`` via ``load(handle)``: this opens the
    file (binary) and hands ``load`` an open handle, so ``load`` is a pure deserializer
    (``json.load`` / ``joblib.load``). Returns ``None`` if the cache is disabled or absent.

    A cache is an optimization, so an unreadable entry is a miss, not a crash. Since
    ``load`` only decodes bytes, any ``Exception`` (corrupt file, stale pickle,
    cross-version estimator) means the bytes didn't decode -- warn, purge the entry so it
    rebuilds, and return ``None``. Only safe because ``load`` carries no application logic
    whose bug this could mask. ``BaseException`` (interrupts) still propagates.
    """
    if _cache_disabled():
        return None

    path = os.fspath(_cache_root() / relpath)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as handle:
            return load(handle)
    except Exception as exc:
        warnings.warn(
            f"ignoring unreadable cache entry {path!r} "
            f"({type(exc).__name__}: {exc}); treating it as a cache miss and removing it "
            f"so it is rebuilt",
            stacklevel=2,
        )
        with contextlib.suppress(OSError):
            os.remove(path)
        return None
