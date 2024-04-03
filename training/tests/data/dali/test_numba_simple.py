import numpy as np
import nvidia.dali.types as types
from numba import jit
from nvidia import dali
from nvidia.dali.plugin.numba.experimental import NumbaFunction


@jit(nopython=True)
def simple_numba(speech: np.ndarray) -> np.ndarray:
    """
    Simple numba function.

    This is a simplified version of blend
    """
    speech_MS = (speech * speech).mean()
    merged = speech + 3
    merged_MS = (merged * merged).mean()
    result = merged * np.sqrt(speech_MS / merged_MS)
    return result


def simple_numba_dali_api(speech_out: np.ndarray, speech_in: np.ndarray) -> np.ndarray:
    """
    The dali numba api requires:
        that the input and output tensors are passed as function args
        that the parent function (here simple_numba_dali_api) isn't jitted
        ...but that the child function (here simple_numba) is jitted with nopython=True
    """
    speech_out[:] = simple_numba(speech_in)
    # It is necessary to copy output into speech_out with [:].
    # The following line (instead of the one above) results in all zeros being returned
    # speech_out = simple_numba(speech_in)
    return speech_out


def test_dali_numba_sample(test_data_dir):
    """
    Simple test to document the (slightly strange) dali Numba api
    """
    audio_file_list = [
        "gov_DOT_uscourts_DOT_ca9_DOT_04-56618_DOT_2006-02-16_DOT_mp3_00027.flac",
        "duplicate_clip.flac",
        "TestNoiseDataset/data/noise_file.wav",
    ]
    audio_file_list = [str(test_data_dir / file) for file in audio_file_list]

    pipe = dali.pipeline.Pipeline(batch_size=3, num_threads=1, device_id=None)
    reader = dali.ops.FileReader(file_root=".", files=audio_file_list, device="cpu")
    decoder = dali.ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True)
    numba_augment = NumbaFunction(
        run_fn=simple_numba_dali_api,
        in_types=[types.FLOAT],
        ins_ndim=[1],
        outs_ndim=[1],
        out_types=[types.FLOAT],
        device="cpu",
    )
    pad = dali.ops.Pad(device="cpu", fill_value=0)
    with pipe:
        audio, labels = reader()
        audio, sr = decoder(audio)
        augmented_audio = numba_augment(audio)
        audio = pad(audio)
        augmented_audio = pad(augmented_audio)

        pipe.set_outputs(augmented_audio, audio, labels)

    pipe.build()
    outputs = pipe.run()

    aug, audio, labs = outputs
    augmented = aug.as_array()
    audio = audio.as_array()

    assert augmented.shape == audio.shape
    assert not np.allclose(
        augmented, audio
    ), "augmented audio should be different to audio"

    # for non-copy failure case:
    assert not np.allclose(
        augmented, np.zeros_like(augmented)
    ), "Should not have all zeros after applying function"
