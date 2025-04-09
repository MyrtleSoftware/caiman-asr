from beartype import beartype
from beartype.typing import Callable, Dict, List, Tuple
from more_itertools import partition
from tqdm import tqdm
from wtpsplit import SaT

from caiman_asr_train.data.text.is_tag import is_tag


@beartype
def add_eos_to_manifest(
    manifest: List[Dict], eos_token: str, use_cuda: bool = True
) -> List[Dict]:
    """
    This function performs a manifest->manifest transformation that segments
    the transcript using an SAT model, adds EOS tokens to the end of each
    segment, and rejoins the modified segments into a single transcript.

    This function handles the edge case where SAT model always designates the
    end of string as the end of a segment. For example the SAT model may segment
    `"I like cake! I like"` into `["I like cake!", "I like"]`. In this case, we
    would like to add an EOS token to the first segment, but not the second.
    """

    assert is_tag(eos_token), "EOS token must be a tag."

    # This is their largest model.
    sat = SaT("sat-12l-sm")

    if use_cuda:
        # This moves the model to GPU in place.
        sat.half().to("cuda")

    single = [x["transcript"].strip() for x in manifest]
    # Also segment the repeated transcript
    # to detect if there's an EOS at the end
    repeat = [" ".join([x, x]) for x in single]

    split_single = sat.split(single)
    split_repeat = sat.split(repeat)

    eos_for = make_eos_for(eos_token)

    for s, r, m in ztqdm(split_single, split_repeat, manifest, total=len(manifest)):
        ms = merge_split_words(s)
        mr = merge_split_words(r)

        n, out = build_transcript(ms, mr, eos_for)

        m["transcript"] = out
        m["eos_count"] = n

    return manifest


def ztqdm(*args, **kwargs):
    return tqdm(zip(*args), **kwargs)


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


@beartype
def make_eos_for(eos_token: str) -> Callable[[str], str]:
    """
    Return a function that appends an EOS token to a segment.
    """
    stripped = eos_token.strip()

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
    splits: List[str], rep_splits: List[str], eos_for: Callable[[str], str]
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
            out.append(eos_for(a))
            eos_count += 1
        out.append(splits[-1])
    else:
        # Only add EOS where the two segmenters agree.
        # If the repeated segmentation put an EOS at the
        # end of the 1st transcript, include that EOS
        for a, b in zip(splits, rep_splits):
            if a.strip() == b.strip():
                out.append(a)
                out.append(eos_for(a))
            else:
                out.append(a)

    return eos_count, "".join(out).strip()


@beartype
def add_eos_to_manifest_avoid_empty(
    manifest: list[dict], eos_token: str, use_cuda: bool
) -> list[dict]:
    # wtpsplit will crash on empty transcripts,
    # so just pass those through the script
    has_transcript, only_has_whitespace = partition(
        lambda utt: utt["transcript"].strip() == "", manifest
    )
    has_transcript_eos = add_eos_to_manifest(list(has_transcript), eos_token, use_cuda)
    return has_transcript_eos + list(only_has_whitespace)
