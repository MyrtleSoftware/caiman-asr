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
    """
    Word and corresponding model timestamps.

    The start and end frame are inclusive i.e. if a word is a single token then
    `start_frame == end_frame`.

    """

    word: str
    start_frame: int  # inclusive
    end_frame: int  # also inclusive


@beartype
@dataclass
class WordTimestamps:
    word: str
    start_time: float
    end_time: float


@beartype
def frame_to_time(
    stamp: PerWordTimestamp,
    frame_width: float,
    initial_pad: float,
    head_offset: float = 0,
    tail_offset: float = 0,
) -> WordTimestamps:
    """
    Convert frame timestamps to time timestamps.
    """

    assert initial_pad < frame_width, "Assumptions invalid"

    # The user gets frame `n` after `(n + 1) * frame_width - initial_pad` of
    # input however the word starts one frame width earlier, the first frame
    # is `frame_width - initial_pad` long and the following are all `frame_width`
    if stamp.start_frame == 0:
        start_time = 0.0
    else:
        start_time = stamp.start_frame * frame_width - initial_pad

    # The end_time needs a +1 for the same reason as above.
    end_time = (stamp.end_frame + 1) * frame_width - initial_pad

    # This results in the correct word end point as follows:
    #
    # User receives frame 0 containing "<blank>" 60ms - 15ms after start of recording.
    # Uses receives frame 1 containing "<blank>" 120ms - 15ms after start of recording.
    # User receives frame 2 containing "cat" 180ms - 15ms after start of recording.
    #
    # Hence:
    #   Start of cat = 2 * 60 - 15 = 120ms - 15ms
    #   End of cat (i.e model outputs cat) = 3 * 60 - 15 = 180ms - 15ms
    #   Duration of cat = 180ms - 120ms = 60ms

    return WordTimestamps(
        word=stamp.word,
        start_time=max(0.0, start_time - head_offset),
        end_time=max(0.0, end_time - tail_offset),
    )


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
