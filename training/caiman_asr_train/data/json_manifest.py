"""
JSON manifest data schema.

Example #1: required fields:
[
  {
    "transcript": "Okay, here's your phone.",
    "files": [
      {
        "fname": "audio_split/S21_P46_56.70_58.36.wav"
      }
    ],
    "original_duration": 1.6599999999999966,
  },
...
]

Example #2: optional fields:

[
  {
    "transcript": "peter told his troubles to mister shimerda he was unable to ...",
    "files": [
      {
        "channels": 1,
        "sample_rate": 16000.0,
        "bitdepth": 16,
        "bitrate": 256000.0,
        "duration": 15.07,
        "num_samples": 241120,
        "encoding": "Signed Integer PCM",
        "silent": false,
        "fname": "dev-clean/2035/147961/2035-147961-0000.flac",
        "speed": 1
      }
    ],
    "original_duration": 15.07,
    "original_num_samples": 241120
  },

    ...
"""

from beartype.typing import Optional
from pydantic import BaseModel, conlist


class File(BaseModel):
    sample_rate: Optional[float] = None
    fname: str
    channels: Optional[int] = None
    bitdepth: Optional[int] = None
    bitrate: Optional[float] = None
    duration: Optional[float] = None
    num_samples: Optional[int] = None
    encoding: Optional[str] = None
    silent: Optional[bool] = None
    speed: Optional[float] = None


class JSONEntry(BaseModel):
    transcript: str
    files: conlist(File, min_length=1, max_length=1)
    original_duration: float
    original_num_samples: Optional[int] = None
    whisper: dict = {}
    parakeet: dict = {}
    original: dict = {}


class JSONDataset(BaseModel):
    entries: list[JSONEntry]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> JSONEntry:
        return self.entries[idx]
