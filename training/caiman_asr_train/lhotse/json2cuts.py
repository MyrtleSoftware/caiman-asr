from collections import defaultdict

import orjson
import soundfile
from beartype import beartype
from beartype.typing import Generator, Optional
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment
from lhotse.cut import Cut
from tqdm import tqdm

from caiman_asr_train.lhotse.config import Manifest
from caiman_asr_train.lhotse.warn import lhotse_warn


@beartype
def json_to_cutset(
    json_file: Manifest,
    text_keys: Optional[list[str]],
    verbose: bool = True,
) -> CutSet:
    """
    Convert a legacy JSON file to a Lhotse CutSet.

    The resulting cutset will store relative paths (just like the legacy JSON).
    This will read the metadata of each audio file to populate the Recording info.

    Arguments:
        json_file: The legacy JSON manifest file.
        verbose: Whether to display a progress bar during conversion.
        text_keys: List of additional json keys to use as transcripts, these
            will be attached to the supervision as custom fields.
    """
    iter = LegacyJsonIter(json_file, text_keys=text_keys)

    if verbose:
        iter = tqdm(iter, desc="Converting legacy JSON to CutSet")

    return CutSet.from_cuts(iter)


@beartype
class LegacyJsonIter:
    def __init__(self, json_file: Manifest, text_keys: Optional[list[str]]):
        """
        An iterator that converts a legacy JSON manifest to Lhotse Cuts.

        This must eagerly loads the entire JSON into memory but lazily does
        the conversion to Cuts.

        Arguments:
            json_file: The legacy JSON manifest file.
            text_keys: List of additional json keys to use as transcripts,
                stored in the supervision's custom field.
        """
        self.json_path = json_file.path
        self.data_dir = json_file.data_dir
        self.text_keys = text_keys if text_keys is not None else []

        if not self.json_path.is_file():
            raise FileNotFoundError(f"JSON file {json_file} not found.")

        # Reads into memory, legacy json is not streamable
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.legacy_manifest = orjson.loads(f.read())

    def __iter__(self) -> Generator[Cut]:
        # Track how many times to warn
        missing = defaultdict(int)

        for utt in self.legacy_manifest:
            r_path = utt["files"][0]["fname"]
            a_path = self.data_dir / r_path

            meta = soundfile.info(a_path)

            cut = Recording(
                id=f"{r_path}-{self.json_path.name}",
                sources=[
                    AudioSource(
                        type="file", channels=list(range(meta.channels)), source=r_path
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
                text=utt["transcript"],
            )

            for key in self.text_keys:
                if key in utt:
                    seg = seg.with_custom(key, utt[key])
                elif missing[key] == 0:
                    lhotse_warn(
                        f"Legacy json missing {key=}, further warnings will be suppressed"
                    )
                    missing[key] += 1

            cut.supervisions.append(seg)

            yield cut

    def __len__(self) -> int:
        return len(self.legacy_manifest)
