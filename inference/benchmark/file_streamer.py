import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pause
from beartype import beartype
from beartype.typing import Generator, Optional
from timer import Timer
from utils import bold_yellow

from caiman_asr_train.data.make_datasets.pretty_print import pretty_path


@contextmanager
def temporary_filename(suffix=None):
    """Context that introduces a temporary file.

    Creates a temporary file, yields its name, and upon context exit, deletes it.
    (In contrast, tempfile.NamedTemporaryFile() provides a 'file' object and
    deletes the file as soon as that file object is closed, so the temporary file
    cannot be safely re-opened by another library or process.)

    Args:
      suffix: desired filename extension (e.g. '.mp4').

    Yields:
      The name of the temporary file.
    """

    tmp_name = None

    try:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_name = f.name
        f.close()
        yield tmp_name
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


@beartype
class FileStreamer:
    def __init__(
        self,
        audio_fpath: str,
        data_dir: Path,
        play_audio: bool,
        timer: Optional[Timer] = None,
        chunk_duration: float = 0.1,
    ) -> None:
        full_audio_path = data_dir / audio_fpath
        assert full_audio_path.is_file(), f"{full_audio_path} does not exist"

        self.sampling_rate = 16000
        self.chunk_duration = chunk_duration
        self.audio_fpath = full_audio_path
        self.play_audio = play_audio
        if play_audio:
            if os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", "No") == "Yes":
                raise OSError(
                    "Cannot play audio in a Docker container. "
                    "Try a virtual environment instead"
                )
            self.aplay_path = shutil.which("aplay")
            if self.aplay_path is None:
                raise FileNotFoundError(
                    "aplay not found in PATH. Try sudo apt install alsa-utils"
                )

        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

    def __enter__(self: object) -> object:
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        pass

    @contextmanager
    def open_as_wav(self, input_path):
        if str(input_path).endswith(".wav"):
            # Short-circuit if already in WAV format
            with open(input_path, "rb") as wav:
                yield wav

            return

        with temporary_filename(suffix=".wav") as output_path:
            subprocess.run(
                [
                    "sox",
                    input_path,
                    "-c",
                    "1",
                    "-r",
                    str(self.sampling_rate),
                    output_path,
                ],
            )

            with open(output_path, "rb") as wav:
                yield wav

    def stream(self) -> Generator[bytes, None, None]:
        """
        Mimics a stream of audio data by reading it chunk by chunk from a file.

        Returns: A generator that yields chunks of audio data.
        """
        # default chunk duration is 300ms, but MicrophoneStream uses only 100ms
        self.chunk_idx = 0
        chunk_size = int(self.sampling_rate * self.chunk_duration) * 2
        num_warnings = 0

        with self.open_as_wav(self.audio_fpath) as audio_file:
            if self.play_audio:
                subprocess.Popen([self.aplay_path, self.audio_fpath])

            audio_bytes = audio_file.read()
            audio_stream = BytesIO(audio_bytes)
            session_start = self.timer.datetime()
            self.start_msg = f"session_start;{session_start}"
            while True:
                # Read a chunk from a file
                data = audio_stream.read(chunk_size)

                # Pad final frame, send, and terminate
                if len(data) < chunk_size:
                    data += bytes(chunk_size - len(data))
                    yield data
                    break

                # Simulate chunk buffering
                send_dtime = self.timer.increment_datetime(
                    session_start,
                    (self.chunk_idx + 1) * self.chunk_duration,
                )
                now = self.timer.now()
                if now > send_dtime:
                    # This can happen if the calling code is too slow,
                    # e.g. if ws.recv() blocks for too long in transcribe_caiman.py
                    num_warnings += 1
                    if num_warnings == 1:
                        delay = self.timer.datetime_diff(now, send_dtime) * 1000
                        print(
                            bold_yellow(
                                "Warning: The client's file streamer"
                                f" for {pretty_path(self.audio_fpath)} "
                                f"is running {delay} ms behind realtime for this frame. "
                                "Perhaps there was a compute latency spike? "
                                "This could artificially increase latencies"
                            )
                        )
                pause.until(send_dtime)

                self.chunk_idx += 1

                # Send chunk
                yield data
        if num_warnings > 1:
            print(
                bold_yellow(
                    f"{pretty_path(self.audio_fpath)} had {num_warnings - 1} hidden "
                    "warnings about slow file streaming"
                )
            )

    def close_trans(self):
        self.end_msg = f"session_end;{self.timer.datetime()}"
