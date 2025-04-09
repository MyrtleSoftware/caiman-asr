from dataclasses import dataclass

from beartype.typing import List, Optional


@dataclass
class CtmItem:
    """
    A single line/entry in a CTM file
    """

    file: str
    speaker: int
    text: str
    start: float
    duration: float
    confidence: Optional[float] = 0.0

    def __str__(self) -> str:
        string = (
            f"{self.file} {self.speaker} {self.start} "
            f"{self.duration} {self.text} {self.confidence}"
        )
        return string

    @staticmethod
    def from_string(string: str):
        """
        String format: `file_name speaker_id start_time duration word confidence`
        """
        parts = string.strip().split(" ")
        return CtmItem(
            file=parts[0],
            speaker=int(parts[1]),
            start=float(parts[2]),
            duration=float(parts[3]),
            text=parts[4],
            confidence=float(parts[5]),
        )


class CTM:
    """
    A collection of all entries in a a CTM file
    """

    def __init__(self) -> None:
        self.items: List[CtmItem] = []

    def add_item(self, item: CtmItem) -> None:
        self.items.append(item)

    def add_items(self, items: List[CtmItem]) -> None:
        self.items += items

    def to_file(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            for item in self.items:
                file.write(str(item) + "\n")

    @staticmethod
    def from_file(file_path: str):
        ctm = CTM()
        with open(file_path, "r") as file:
            for line in file:
                item = CtmItem.from_string(line)
                ctm.add_item(item)
        return ctm

    def __str__(self):
        return "\n".join([str(item) for item in self])

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.items):
            result = self.items[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    def get_transcripts(self):
        """
        Return transcript for all items. Useful to calculate WER. The return
        format is dict:
        {
            file1: 'string',
            file2: 'string',
            ...
        }
        """
        if not len(self.items):
            return {}

        trans = {}
        curr_file, curr_trans = "", ""
        for idx, item in enumerate(self):
            if idx == 0:
                curr_file = item.file
                curr_trans = item.text
                continue

            if item.file == curr_file:
                curr_trans += f" {item.text}"
            else:
                trans[curr_file] = curr_trans
                curr_file = item.file
                curr_trans = item.text
        trans[curr_file] = curr_trans

        return trans

    def get_timestamps(self):
        """
        Return a list of word and timestamps for all files. Useful to calculate
        latencies. The return format is :
        {
            file1: [(w1, start1, dur1), (w2, start2, dur2), ...],
            file2: [(w1, start1, dur1), (w2, start2, dur2), ...],
            ...
        }
        """
        if not len(self.items):
            return {}

        out = {}
        curr_file, curr_tstamps = "", []
        for idx, item in enumerate(self):
            if idx == 0:
                curr_file = item.file
                curr_tstamps = [(item.text, float(item.start), float(item.duration))]
                continue

            if item.file == curr_file:
                curr_tstamps += [(item.text, float(item.start), float(item.duration))]
            else:
                out[curr_file] = curr_tstamps
                curr_file = item.file
                curr_tstamps = [(item.text, float(item.start), float(item.duration))]
        out[curr_file] = curr_tstamps

        return out
