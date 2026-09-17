from pathlib import Path

import pytest

from caiman_asr_train.autotune import cache_io


def _use_tmp_bench_cache(monkeypatch, tmp_path):
    """
    Isolate the autotuner's on-disk bench cache in ``tmp_path`` with caching ENABLED
    (clear the two disable-cache env knobs).
    """
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CAIMAN_LSTM_AUTOTUNE_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("TILELANG_DISABLE_CACHE", raising=False)


def test_cache_read_purges_unreadable_entry(tmp_path, monkeypatch):
    # An undecodable cached entry (stale pickle, truncated file, cross-version estimator)
    # must degrade to a miss and be purged so it rebuilds, never crash the caller.
    _use_tmp_bench_cache(monkeypatch, tmp_path)
    rel = "robust/probe.bin"
    cache_io.cache_write(rel, lambda p: Path(p).write_text("not a real bundle"))

    def _stale_pickle(_handle):
        raise ModuleNotFoundError("No module named 'caiman_asr_train.tilelang_lstm'")

    with pytest.warns(UserWarning, match="unreadable cache entry"):
        assert cache_io.cache_read(rel, _stale_pickle) is None
    # Purged: next read is a clean miss (file gone, loader not called).
    assert cache_io.cache_read(rel, _stale_pickle) is None
    assert not (tmp_path / rel).exists()


def test_cache_read_treats_any_exception_as_a_miss(tmp_path, monkeypatch):
    # load is a pure deserializer, so any Exception it raises means the bytes are
    # unreadable: cache_read purges + misses. A KeyboardInterrupt is BaseException, not
    # Exception, so it still propagates and leaves the entry intact.
    _use_tmp_bench_cache(monkeypatch, tmp_path)
    rel = "robust/broken.bin"
    cache_io.cache_write(rel, lambda p: Path(p).write_text("some cached bytes"))

    def _raises(_handle):
        raise KeyError("decode blew up")

    with pytest.warns(UserWarning, match="unreadable cache entry"):
        assert cache_io.cache_read(rel, _raises) is None
    assert not (tmp_path / rel).exists()  # purged -> rebuilds cleanly

    cache_io.cache_write(rel, lambda p: Path(p).write_text("some cached bytes"))

    def _interrupt(_handle):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cache_io.cache_read(rel, _interrupt)
    assert (tmp_path / rel).exists()  # an interrupt never deletes the entry
