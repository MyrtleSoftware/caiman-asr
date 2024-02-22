import time

import numpy as np
import pytest
from beartype.typing import List

from rnnt_train.common.data.dali.noise import babble_batch, blend, duplicate_then_blend


def are_allclose(a: List[np.ndarray], b: List[np.ndarray]) -> bool:
    """
    Return True if a and b are allclose, False otherwise.
    """
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if not np.allclose(a[i], b[i]):
            return False
    return True


def test_blend(speech_len: float = 100):
    speech = np.arange(speech_len, dtype=float)
    noise = np.ones_like(speech)

    # check value doesn't change when applying function twice
    target_snr = 400
    start_ratio = np.array([0.1], dtype=np.float32)
    blended1 = blend(speech, noise, target_snr, start_ratio)
    blended2 = blend(speech, noise, target_snr, start_ratio)

    assert np.allclose(blended1, blended2)

    # ..and that the function is a no-op at high snr values
    assert np.allclose(blended1, speech)
    # and that this is still the case when passing a numpy array for target_snr
    high_snr = np.array([400], dtype=np.float32)
    blended1 = blend(speech, noise, high_snr, start_ratio)
    assert np.allclose(blended1, speech)

    # check value doesn't change when applying function twice @lower noise
    target_snr = 10
    blended1 = blend(speech, noise, target_snr, start_ratio)
    blended2 = blend(speech, noise, target_snr, start_ratio)
    assert np.allclose(blended1, blended2)

    # ..and that the function isn't a no-op at low snr values
    assert not np.allclose(blended1, speech)

    # check that value doesn't change when using python version
    blended_py = blend.py_func(speech, noise, target_snr, start_ratio)
    assert np.allclose(blended1, blended_py)


def test_short_noise():
    """Only duplicate_then_blend can handle noise shorter than speech"""
    speech_len = 100
    speech = np.arange(speech_len, dtype=float)
    noise = np.ones((speech_len - 1,))

    target_snr = 10
    start_ratio = np.array([0.1], dtype=np.float32)
    with pytest.raises(ValueError):
        _ = blend(speech, noise, target_snr, start_ratio)
    _ = duplicate_then_blend(speech, noise, target_snr, start_ratio)


def test_long_noise():
    """blend and duplicate_then_blend give the same result for noise longer than speech"""
    speech_len = 100
    speech = np.arange(speech_len, dtype=float)
    noise = np.ones((speech_len + 1,))

    target_snr = 10
    start_ratio = np.array([0.1], dtype=np.float32)
    blended1 = blend(speech, noise, target_snr, start_ratio)
    blended2 = duplicate_then_blend(speech, noise, target_snr, start_ratio)
    assert np.allclose(blended1, blended2)


def test_time_blend(
    speech_len_high: float = 1000, n_trials: int = 200, expected_min_speedup: int = 3
):
    """
    The original function (pre jit compilation) had the following characteristics:
        original: 1.0659337043762207e-4 +/- 1.7639974672490224e-05

    Compilation on a DGX-1 showed:
        post-compile: 1.1928081512451172e-05 +/- 2.3266172050936466e-06

    i.e. a 10x speedup.

    NOTE: On other machines a smaller speedup is recorded, so in order
    to avoid this test being flaky, a significantly lower expected_min_speedup
    than 10 is used.

    Note that compiling with both parallel=True and fastmath=True may slow
    the code down and leave the speedup unchanged respectively.
    """
    np.random.seed(0)
    speechs = [
        np.arange(np.random.randint(10, speech_len_high), dtype=float)
        for _ in range(n_trials + 1)
    ]
    noises = [np.ones_like(speech) for speech in speechs]
    start_ratio = np.array([0.1], dtype=np.float32)

    _ = blend(speechs[0], noises[0], 10, start_ratio)

    def time_blend(fn, n_trials):
        times = []
        for idx in range(n_trials):
            speech, noise = speechs[idx + 1], noises[idx + 1]
            start = time.time()
            fn(speech, noise, 10, start_ratio)
            end = time.time()
            times.append(end - start)
        return times

    times = time_blend(blend, n_trials)

    times_py = time_blend(blend.py_func, n_trials)

    average = np.mean(times)
    std = np.std(times)
    average_py = np.mean(times_py)
    print(f"Time: {average} +/- {std}")
    assert average * expected_min_speedup < average_py


def test_babble_batch(max_speech_len: float = 10, batch_size: int = 3):
    assert batch_size > 1, "This test is for batch_size>1"

    # init array of speech with random lengths
    i = 0
    speech = []
    speech_values = np.arange(batch_size * max_speech_len, dtype=np.float32)
    for _ in range(batch_size):
        speech_len = np.random.randint(2, max_speech_len)
        speech.append(speech_values[i : i + speech_len])
        i += speech_len

    # check result doesn't change when applying function twice
    target_snrs = np.ones(batch_size, dtype=np.float32).reshape(-1, 1) * 400
    start_ratios = np.ones_like(target_snrs, dtype=np.float32) * 0.1

    blended1 = babble_batch(speech, target_snrs, start_ratios)
    blended2 = babble_batch(speech, target_snrs, start_ratios)

    assert are_allclose(blended1, blended2)

    # ..and that the function is a no-op at high snr values
    assert are_allclose(blended1, speech)

    # check value doesn't change when applying function twice @lower noise
    target_snrs = np.ones(batch_size, dtype=np.float32).reshape(-1, 1) * 10
    blended1 = babble_batch(speech, target_snrs, start_ratios)
    blended2 = babble_batch(speech, target_snrs, start_ratios)
    assert are_allclose(blended1, blended2)

    # ..and that the function isn't a no-op at low snr values
    assert not are_allclose(blended1, speech)

    # check that value doesn't change when using python version
    blended_py = babble_batch.py_func(speech, target_snrs, start_ratios)
    assert are_allclose(blended1, blended_py)


def test_babble_batch_raises(speech_len: float = 10):
    # test that babble_batch raise ValueError when given batch_size=1
    batch_size = 1
    speech = [np.arange(speech_len, dtype=np.float32)]
    target_snrs = np.ones(batch_size, dtype=np.float32).reshape(-1, 1) * 10
    start_ratios = np.ones_like(target_snrs, dtype=np.float32) * 0.1

    with pytest.raises(ValueError):
        babble_batch(speech, target_snrs, start_ratios)
