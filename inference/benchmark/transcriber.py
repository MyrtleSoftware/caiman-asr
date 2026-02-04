#!/usr/bin/env python3


import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

from beartype import beartype
from beartype.typing import List
from ctm import CTM
from file_streamer import FileStreamer
from timer import Timer
from tqdm import tqdm
from utils import bold_yellow, ftrans_is_valid


@beartype
class Transcriber(ABC):
    """
    Transcriber base class to interface and benchmark ASR providers.

    Args:
        limit_to: Optional. Limit the number of files to transcribe.
        trans_dir: Path to the directory to save the transcriptions.
        retries: Number of times to retry transcription.
        data_dir: Path to the directory containing the data.
        play_audio: Whether to play the audio file. For debugging purposes.
        suffix: Suffix for the transcription file.
        model_id: Optional. Model ID for the transcription.
    """

    def __init__(
        self,
        limit_to: int | None,
        trans_dir: Path,
        retries: int,
        data_dir: Path,
        play_audio: bool,
        suffix: str,
        model_id: str | None = None,
    ):
        self.timer = Timer()
        self.limit_to = limit_to
        self.trans_dir = trans_dir
        self.retries = retries
        self.data_dir = data_dir
        self.play_audio = play_audio
        self.trans_dir.mkdir(exist_ok=True, parents=True)
        self.suffix = suffix
        self.model_id = model_id

    @abstractmethod
    def attempt_to_transcribe(self, file: str, ftrans: Path, duration: float) -> None:
        pass

    @abstractmethod
    def trans_to_ctm(self, manifest: List[dict]) -> CTM:
        pass

    def transcribe(self, manifest: List[dict], force: bool) -> None:
        manifest_ = manifest[: self.limit_to] if self.limit_to is not None else manifest
        for item in tqdm(manifest_, desc="Transcribing files"):
            file = item["files"][0]["fname"]
            ftrans = self.trans_dir / f"{Path(file).stem}.{self.suffix}"

            if not force:
                if ftrans_is_valid(ftrans):
                    continue
            sr = item["files"][0].get("sample_rate", 16000)
            assert (
                sr == 16000
            ), f"Sample rate != 16000 for {file=}. Downstream code assumes it is. {sr=}"
            retries_so_far = 0
            while retries_so_far < self.retries:
                found_error = False
                try:
                    self.attempt_to_transcribe(file, ftrans, item["original_duration"])
                except Exception:
                    print("Encountered an error during transcription attempt")
                    traceback.print_exc()
                    found_error = True

                # Check for retries
                if found_error or not ftrans_is_valid(ftrans):
                    retries_so_far += 1
                    print(
                        bold_yellow(
                            f"Warning: file {file} failed transcription attempt "
                            f"{retries_so_far}/{self.retries}"
                        )
                    )
                    if retries_so_far == self.retries:
                        # Delete file if invalid
                        # so it doesn't crash the latency calculation step
                        if ftrans.is_file():
                            ftrans.unlink()
                        raise Exception(
                            f"Failed to transcribe {file} after {self.retries} attempts"
                        )
                    else:
                        time.sleep(0.1)
                else:
                    break

    def write_trans_to_file(
        self, ftrans: Path, file_streamer: FileStreamer, messages: List[str]
    ) -> None:
        with open(ftrans, "w") as ftrans_fh:
            ftrans_fh.write(f"{file_streamer.start_msg}\n")
            for trans in messages:
                ftrans_fh.write(f"{trans}\n")
            ftrans_fh.write(f"{file_streamer.end_msg}\n")
