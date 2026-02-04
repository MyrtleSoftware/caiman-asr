from enum import Enum, auto

from beartype import beartype
from beartype.typing import Dict, Optional

from caiman_asr_train.train_utils.distributed import print_once

_DEFAULT_TRANSCRIPT = "transcript"

_DEFAULT_SELECT = {
    "use_transcripts": {
        "to_use": [_DEFAULT_TRANSCRIPT],
        "on_missing": "raise_error",
    }
}


class OnMissing(Enum):
    raise_error = auto()
    use_default = auto()
    skip = auto()


@beartype
class SelectTranscript:
    def __init__(self, config: Dict = _DEFAULT_SELECT):
        """
        Build a functor that selects the transcript to be used.

        Defaults to raising an error if the transcript is not found.
        """

        if "use_transcripts" not in config or "to_use" not in config["use_transcripts"]:
            print_once(
                "Field 'use_transcripts' not in config, will use the default transcripts"
            )

            self.key = [_DEFAULT_TRANSCRIPT]
            self.err = OnMissing.raise_error

        else:
            self.key = config["use_transcripts"]["to_use"]

            if not isinstance(self.key, list):
                self.key = [self.key]

            match config["use_transcripts"].get("on_missing", "raise_error"):
                case "raise_error":
                    self.err = OnMissing.raise_error
                case "use_default":
                    self.err = OnMissing.use_default
                case "skip":
                    self.err = OnMissing.skip
                case _:
                    raise ValueError(
                        f"Invalid 'on_missing': {config['use_transcripts']['on_missing']}"
                    )

        if config is not _DEFAULT_SELECT:
            print_once(f"Will use transcript(s): {self.key}, else {self.err}")

    def select(self, manifest_entry: dict) -> Optional[tuple[str, str]]:
        """
        Select the transcript to be used.

        Returns a tuple of:
            (selected transcript, selected transcript key)

        Only returns None if no transcript is found and the error handling is set to skip.
        """
        for k in self.key:
            if k in manifest_entry:
                return manifest_entry[k], k

        match self.err:
            case OnMissing.raise_error:
                raise KeyError(
                    f"No transcript matching {self.key} found in {manifest_entry}"
                )
            case OnMissing.use_default:
                return manifest_entry[_DEFAULT_TRANSCRIPT], _DEFAULT_TRANSCRIPT
            case OnMissing.skip:
                return None

    def __call__(self, manifest_entry: dict) -> Optional[str]:
        """
        Select the transcript to be used.

        Only returns None if no transcript is found and the error handling is set to skip.
        """
        match self.select(manifest_entry):
            case None:
                return None
            case (transcript, _):
                return transcript
