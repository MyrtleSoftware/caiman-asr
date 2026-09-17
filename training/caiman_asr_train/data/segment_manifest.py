from beartype import beartype
from beartype.typing import Callable, Dict, List, Optional, Tuple
from lhotse import CutSet, SupervisionSegment
from tqdm import tqdm
from wtpsplit import SaT

from caiman_asr_train.data.text.is_tag import is_tag


@beartype
class ManifestSegmenter:
    """EOS segmenter for endpointing Lhotse Manifests"""

    def __init__(self, eos_token, use_cuda):
        assert is_tag(eos_token), "EOS token must be a tag."
        self.eos_token = eos_token

        # This is their largest model.
        self.sat = SaT("sat-12l-sm")
        if use_cuda:
            # This moves the model to GPU in place.
            self.sat.half().to("cuda")

        self.eos_for = self.make_eos_for()

    @beartype
    def make_eos_for(self) -> Callable[[str], str]:
        """
        Return a function that appends an EOS token to a segment.
        """
        stripped = self.eos_token.strip()

        @beartype
        def eos_for(seg: str) -> str:
            """
            Add an EOS token to the end of a segment.
            """
            if seg.endswith(" "):
                return f"{stripped} "
            else:
                return f" {stripped}"

        return eos_for

    @beartype
    def build_transcript(
        self, splits: List[str], rep_splits: List[str]
    ) -> Tuple[int, str]:
        """
        Analyze the segmenter's outputs and add EOS tokens to the transcript.
        """

        # This counts how many times the segmenter
        # agreed with itself on the first half of the repeated
        # transcript. This could be 0---the segmenter sees future context
        eos_count = 0

        for a, b in zip(splits, rep_splits):
            # Need to strip because we added white space into rep_splits.
            if a.strip() == b.strip():
                eos_count += 1

        out = []

        if eos_count == 0 and len(splits) > 1:
            # The segmenter didn't agree with itself at all.
            # Empirically, this happens when the transcript
            # was cut off in the middle of a sentence. So
            # fall back to trusting the non-repeated segmentation,
            # and don't put an EOS at the end
            for a in splits[:-1]:
                out.append(a)
                out.append(self.eos_for(a))
                eos_count += 1
            out.append(splits[-1])
        else:
            # Only add EOS where the two segmenters agree.
            # If the repeated segmentation put an EOS at the
            # end of the 1st transcript, include that EOS
            for a, b in zip(splits, rep_splits):
                if a.strip() == b.strip():
                    out.append(a)
                    out.append(self.eos_for(a))
                else:
                    out.append(a)

        return eos_count, "".join(out).strip()

    @beartype
    def _get_transcripts_from_cutset(
        self, cutset: CutSet, transcript_field: Optional[str] = None
    ):
        transcripts = []
        for cut in cutset:
            for sup in cut.supervisions:
                if transcript_field:
                    transcript = _custom_transcript(sup, transcript_field)
                else:
                    transcript = sup.text
                transcripts.append(transcript)
        return transcripts

    @beartype
    def add_eos_to_cutset(
        self, cutset_id: str, cutset: CutSet, transcript_field: Optional[str] = None
    ) -> Dict[str, str]:
        """
        This function performs a text transformation that segments
        the transcript using an SAT model, adds EOS tokens to the end of each
        segment, and rejoins the modified segments into a single transcript.

        This function handles the edge case where SAT model always designates the
        end of string as the end of a segment. For example the SAT model may segment
        `"I like cake! I like"` into `["I like cake!", "I like"]`. In this case, we
        would like to add an EOS token to the first segment, but not the second.
        """

        text_transforms = {}

        # Skip empty transcripts
        single = self._get_transcripts_from_cutset(cutset, transcript_field)

        # Also segment the repeated transcript
        # to detect if there's an EOS at the end
        repeat = [" ".join([x, x]) for x in single]

        split_single = self.sat.split(single)
        split_repeat = self.sat.split(repeat)

        for s, r, og_s in ztqdm(
            split_single, split_repeat, single, total=len(single), desc=cutset_id
        ):
            ms = merge_split_words(s)
            mr = merge_split_words(r)

            _, out = self.build_transcript(ms, mr)

            text_transforms[og_s] = out

        return text_transforms


def ztqdm(*args, **kwargs):
    """Zipped tqdm"""
    return tqdm(zip(*args), **kwargs)


@beartype
def _custom_transcript(sup: SupervisionSegment, transcript_field: str) -> str:
    if sup.has_custom(transcript_field):
        transcript = sup.custom[transcript_field]
    else:
        raise ValueError(f"{transcript_field} not found in SupervisionSegment")
    return transcript


@beartype
def merge_split_words(splits: List[str]) -> List[str]:
    """
    Fix segments that were split mid-word.

    >>> merge_split_words(["hello ", "wor", "ld"])
    ['hello ', 'world']
    """

    head = splits[:1]
    tail = splits[1:][::-1]

    while tail:
        next = tail.pop()

        if head[-1].endswith(" ") or next.startswith(" "):
            head.append(next)
        else:
            head[-1] += next

    return head
