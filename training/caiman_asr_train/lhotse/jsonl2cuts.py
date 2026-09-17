import json

import soundfile
from beartype import beartype
from beartype.typing import Generator
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from lhotse.cut import Cut
from tqdm import tqdm

from caiman_asr_train.lhotse.config import Manifest


@beartype
def jsonl_to_cutset(
    jsonl_file: Manifest,
    verbose: bool = True,
) -> CutSet:
    """
    Convert a Granary JSONL file to a Lhotse CutSet.

    The resulting cutset will store relative paths (just like the Granary JSONL manifest).
    This will read the metadata of each audio file to populate the Recording info.

    Arguments:
        jsonl_file: The JSONL manifest file.
        verbose: Whether to display a progress bar during conversion.
    """
    iter = JsonlIter(jsonl_file)

    if verbose:
        iter = tqdm(iter, desc="Converting Granary JSONL to CutSet")

    return CutSet.from_cuts(iter)


@beartype
class JsonlIter:
    """
    An iterator that converts a Granary JSONL manifest to Lhotse Cuts.

    Arguments:
        jsonl_file: The Granary JSONL manifest file.
    """

    def __init__(self, jsonl_file: Manifest):
        self.jsonl_path = jsonl_file.path
        self.data_dir = jsonl_file.data_dir

        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"JSONL file {jsonl_file} not found.")

    def __iter__(self) -> Generator[Cut]:

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                utt = json.loads(line)
                r_path = utt["audio_filepath"]
                a_path = self.data_dir / r_path

                meta = soundfile.info(a_path)

                cut = Recording(
                    id=f"{r_path}-{self.jsonl_path.name}",
                    sources=[
                        AudioSource(
                            type="file",
                            channels=list(range(meta.channels)),
                            source=r_path,
                        )
                    ],
                    sampling_rate=int(meta.samplerate),
                    num_samples=meta.frames,
                    duration=meta.duration,
                ).to_cut()

                # Help the type checker
                assert cut.recording is not None

                seg = SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.recording.id,
                    start=0,
                    duration=cut.duration,
                    text=utt["text"],
                )

                cut.supervisions.append(seg)

                yield cut

    def __len__(self) -> int:
        return len(self.jsonl_path)
