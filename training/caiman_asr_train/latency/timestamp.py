from dataclasses import dataclass

from beartype import beartype
from beartype.typing import List


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


@beartype
def group_timestamps(
    subwords_list: List[List[str]],
    timestamps_list: List[List[int]],
    sentences: List[str],
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

    Returns
    -------
    A list of SequenceTimestamps for each sentence.
    """
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
            while index < len(subwords) and "".join(word_tokens) != word:
                word_tokens.append(subwords[index])
                index += 1

            # Start time is the timestamp of the first token
            start_time = timestamps[index - len(word_tokens)]
            # End time is the timestamp of the last token
            end_time = timestamps[index - 1]
            new_timestamps.append(PerWordTimestamp(word, start_time, end_time))

        results.append(SequenceTimestamp(new_timestamps))
    return results
