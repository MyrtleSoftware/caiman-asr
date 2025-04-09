#!/usr/bin/env python3


import time
import traceback
from pathlib import Path

from beartype import beartype
from beartype.typing import List
from utils import bold_yellow, ftrans_is_valid


@beartype
class Transcriber:
    def transcribe(self, manifest: List[dict], force: bool) -> None:
        manifest_ = manifest[: self.limit_to] if self.limit_to is not None else manifest
        for i, item in enumerate(manifest_):
            print(f"Transcribing {i+1}/{len(manifest_)}...")
            file = item["files"][0]["fname"]
            ftrans = self.trans_dir / f"{Path(file).stem}.{self.suffix}"

            if not force:
                if ftrans_is_valid(ftrans):
                    continue

            retries_so_far = 0
            while retries_so_far < self.retries:
                found_error = False
                try:
                    self.attempt_to_transcribe(file, ftrans)
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
