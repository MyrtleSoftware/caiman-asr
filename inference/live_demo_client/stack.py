from enum import Enum

from beartype import beartype
from beartype.typing import List, Optional


class Style(Enum):
    FINAL = "\033[92m"  # green
    PARTIAL = "\033[0;31m"  # red


_END = "\033[0m"


def fprint(msg: str, sty: Optional[Style] = None):
    if sty is None:
        print(msg, end="", flush=True)
    else:
        print(f"{sty.value}{msg}{_END}", end="", flush=True)


class PipeStack:
    @beartype
    def __init__(self, cols: int = 80) -> None:
        """
        A stack of strings that can be pushed and popped from the terminal.
        This class abstracts line wrapping and deletion.
        """

        self._stack: List[List[str]] = []
        # Wrap early so we always know how
        # many new lines there have been
        self._cols: int = cols
        self._curs: List[int] = [0]

        fprint("\n")

    @property
    @beartype
    def _top(self) -> int:
        return self._curs[-1]

    @_top.setter
    @beartype
    def _top(self, value: int) -> None:
        self._curs[-1] = value

    @beartype
    def _back_line(self) -> None:
        assert self._curs.pop() == 0
        # Move cursor up one line
        fprint("\033[F")
        # Move cursor to right
        fprint(f"\033[{self._top}C")

    def _down_line(self):
        self._curs.append(0)
        fprint("\n")

    @beartype
    def _del(self, n: int) -> None:
        if n == 0:
            return

        if self._top == 0:
            self._back_line()

        assert self._top >= n
        self._top = self._top - n

        # Overwrite the last n characters
        fprint("\b" * n)
        fprint(" " * n)
        fprint("\b" * n)

    @beartype
    def _words(self, msg: str) -> List[str]:
        """
        Split into words, the words have spaces in front of them,
        the first words may not have a space if it is part of a
        multi-token word.
        """

        if msg == "":
            return []

        words = [f" {m}" for m in msg.split(" ") if m != ""]

        if words and not msg.startswith(" "):
            words[0] = words[0][1:]

        return words

    @beartype
    def _push(self, word: str, sty: Optional[Style] = None) -> str:
        assert len(word) < self._cols, "Message is too long for one line"

        if word.startswith(" ") and self._top + len(word) > self._cols:
            self._down_line()

        if word.startswith(" ") and self._top == 0:
            word = word[1:]

        self._top = self._top + len(word)
        fprint(word, sty)

        return word

    @beartype
    def push(self, msg: str, sty: Optional[Style] = None) -> None:
        """
        Push a message to the stack, it will be displayed on the terminal.
        """
        self._stack.append([self._push(word, sty) for word in self._words(msg)])

    @beartype
    def pop(self) -> None:
        """
        Pop the last message from the stack and delete it from the terminal.
        """
        words = self._stack.pop()

        while words:
            self._del(len(words.pop()))
