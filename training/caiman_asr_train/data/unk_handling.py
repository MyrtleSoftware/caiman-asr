#!/usr/bin/env python3

from enum import Enum

from beartype import beartype

from caiman_asr_train.train_utils.distributed import warn_once


class UnkHandling(Enum):
    FAIL = "FAIL"
    WARN = "WARN"


@beartype
def check_tokenized_transcript(
    tokens: list[int], transcript: str, unk_handling: UnkHandling
) -> None:
    """Raise an error if there's an unk in the transcript,
    except only warn if unk_handling is set to WARN."""
    has_unk = 0 in tokens
    if not has_unk:
        return
    message = f"<unk> found during tokenization (OOV?)\n{transcript}"
    match unk_handling:
        case UnkHandling.FAIL:
            raise ValueError(
                message + "\nSee the 'Changing the character set' docs "
                "for how to disable error if this is expected"
            )
        case UnkHandling.WARN:
            warn_once(message)
        case _:
            raise ValueError(f"Invalid enum value: {unk_handling}")


@beartype
def maybe_filter_transcripts(
    transcripts: list[list[int]], unk_handling: UnkHandling
) -> list[list[int]]:
    """When the user sets unk_handling to WARN,
    filter out transcripts with unks
    so that KenLM training doesn't crash"""
    match unk_handling:
        case UnkHandling.FAIL:
            return transcripts
        case UnkHandling.WARN:
            # The user will have previously seen warnings about the unks
            # during tokenization. Filter the unks out here so the ngram
            # training doesn't fail
            return [transcript for transcript in transcripts if 0 not in transcript]
        case _:
            raise ValueError(
                f"Invalid value: {unk_handling} "
                "in the '/path/to/sentencepiece.yaml' configuration file."
            )
