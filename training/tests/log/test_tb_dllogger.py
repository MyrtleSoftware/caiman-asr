#!/usr/bin/env python3
from caiman_asr_train.log.tb_dllogger import TBLogger


def make_logger(tmp_path, interval=1):
    return TBLogger(
        enabled=True, log_dir=str(tmp_path), name="train", interval=interval
    )


def test_log_value_writes_scalar(tmp_path):
    logger = make_logger(tmp_path)
    logger.log_value(1, "loss", 2.0)
    assert "loss" not in logger.cache, "cache entry must be flushed on write"


def test_log_value_survives_writer_failure(tmp_path):
    """A transient I/O failure inside TensorBoard's event writer killed a real
    training run mid-flight; log_value must drop the point instead of raising,
    and must not grow the cache without bound while the writer is broken."""
    logger = make_logger(tmp_path)

    def broken_add_scalar(*args, **kwargs):
        raise OSError("disk unhappy")

    logger.summary_writer.add_scalar = broken_add_scalar

    for step in range(5):
        logger.log_value(step, "loss", 1.0)  # must not raise
        assert "loss" not in logger.cache, "failed writes must still clear the cache"


def test_log_value_respects_interval(tmp_path):
    logger = make_logger(tmp_path, interval=3)
    written = []
    logger.summary_writer.add_scalar = lambda key, val, step: written.append(
        (key, val, step)
    )

    logger.log_value(1, "loss", 1.0)
    logger.log_value(2, "loss", 2.0)
    assert written == []
    logger.log_value(3, "loss", 3.0)
    assert written == [("loss", 2.0, 3)], "mean over the interval, one write"
