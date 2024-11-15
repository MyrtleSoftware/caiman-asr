from dataclasses import dataclass

from beartype import beartype
from beartype.typing import List, TypeAlias, Union


@beartype
@dataclass
class Silence:
    """
    Indicates that the utterance was terminated due to a silence.
    """

    final_time: float


@beartype
@dataclass
class EOS:
    """
    Indicates that the utterance was terminated due to an end-of-sentence token.
    """

    final_time: float


@beartype
@dataclass
class Never:
    """
    Indicates that the utterance was not terminated during the evaluation.
    """

    pass


Termination: TypeAlias = Union[Silence, Never, EOS]


@beartype
@dataclass
class PerWordTimestamp:
    """Word and corresponding model timestamps."""

    word: str
    start_frame: int
    end_frame: int


@beartype
@dataclass
class SequenceTimestamp:
    """List of PerWordTimestamps for a sentence."""

    seqs: List[PerWordTimestamp]
    eos: Termination


@beartype
def group_timestamps(
    subwords_list: List[List[str]],
    timestamps_list: List[List[int]],
    sentences: List[str],
    last_emit_time: List[Termination],
) -> List[SequenceTimestamp]:
    """Return word-level timestamps.

    This function matches tokens/subwords to words and accordingly
    combines the token-level timestamps to produce word-level timestamps.
    Token-level timestamps have a single time index for each token,
    whilst word-level timestamps have a start and end time index.

    Parameters
    ----------
    subwords_list:
        List of lists of predicted subwords (detokenized tokens).
    timestamps_list:
        List of lists of single timestamps corresponding to each token/subword.
    sentences:
        List of detokenized predicted sentences.
    last_emit_time:
        Time of EOS/final emit for each sentence.

    Returns
    -------
    A list of SequenceTimestamps for each sentence.
    """
    assert (
        len(sentences)
        == len(subwords_list)
        == len(timestamps_list)
        == len(last_emit_time)
    )
    results = []
    for i, sentence in enumerate(sentences):
        subwords, timestamps = subwords_list[i], timestamps_list[i]
        # Initialize variables
        new_timestamps = []
        index = 0
        sentence_words = sentence.split()

        for word in sentence_words:
            word_tokens = []
            # Gather all subwords that could make up this word
            while index < len(subwords) and "".join(word_tokens).strip() != word:
                # Trim leading spaces
                if word_tokens or subwords[index].strip() != "":
                    word_tokens.append(subwords[index])

                index += 1

            word_span = timestamps[index - len(word_tokens) : index]
            start_time = min(word_span)
            end_time = max(word_span)

            new_timestamps.append(PerWordTimestamp(word, start_time, end_time))

        results.append(SequenceTimestamp(new_timestamps, last_emit_time[i]))
    return results
