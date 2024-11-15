#!/usr/bin/env python3

from dataclasses import dataclass

from beartype import beartype


@beartype
@dataclass
class ServerResponse:
    text: str
    timestamp: float
    is_partial: bool


@beartype
def fuse_timestamps(responses: list[ServerResponse]) -> list[tuple[str, float]]:
    char_timestamps, partials = [], []

    for response in responses:
        if response.is_partial:
            partials.append((response.text, response.timestamp))

        else:
            trans, tstamp = response.text, response.timestamp

            for idx, char in enumerate(trans):
                char_tstamp = tstamp
                for partial_word, curr_tstamp in partials[::-1]:
                    if idx > len(partial_word) - 1:
                        continue
                    elif partial_word[idx] == char:
                        char_tstamp = curr_tstamp
                    else:
                        break
                char_timestamps.append((char, char_tstamp))

            new_partials = []
            for partial_word, tstamp in partials:
                if len(partial_word) > len(trans):
                    new_partials.append((partial_word[len(trans) :], tstamp))
            partials = new_partials

    return char_timestamps


@beartype
def get_word_timestamps(responses: list[ServerResponse]) -> list[tuple[str, float]]:
    # Fuse partials/finals into a single list
    char_timestamps = fuse_timestamps(responses)

    # Split into words
    prev_tstamp, word, words = 0.0, "", []
    for char, tstamp in char_timestamps:
        if char == " ":
            if word:
                words.append((word, prev_tstamp))
            word = ""
            prev_tstamp = 0.0
        else:
            word += char
            prev_tstamp = max(prev_tstamp, tstamp)

    if word:
        words.append((word, prev_tstamp))

    return words
