import difflib

from beartype import beartype
from beartype.typing import Dict, List
from ctm import CTM
from levenshtein_rs import levenshtein_list as levenshtein

from caiman_asr_train.evaluate.metrics import standardize_wer
from caiman_asr_train.latency.measure_latency_lite import compute_latency_metrics


@beartype
def calculate_wer(ref_ctm: CTM, hyp_ctm: CTM) -> Dict:
    """
    This function calculates the WER between two CTM object. It is calculated as
    the sum of addition, deletions, and substitutions of words in the hypotheses
    list entries compared to the respective entries in the references list, divided
    by the total number of words in the the references list.

    Arguments
    ----------
    :ref_ctm: a reference CTM object, use CTM.from_file(fpath) to load a ctm file
    :hyp_ctm: a hypothess CTM object, use CTM.from_file(fpath) to load a ctm file

    Returns
    -------
    :wer: word error rate metric
    :scores: the number of additions, deletions, substitutions in all the entries of the
        hypotheses list
    :words: the number of words in the references list

    Raises
    ------
    ValueError: if the number of references is greater than the number of hypotheses
    """
    refs = ref_ctm.get_transcripts()
    hyps = hyp_ctm.get_transcripts()

    scores, words = 0, 0
    for ref_id, ref_txt in refs.items():
        if ref_id not in hyps.keys():
            continue

        hyp_txt = hyps[ref_id]
        hyp_txt = standardize_wer(hyp_txt)
        hyp_txt = hyp_txt.strip().split(" ")

        ref_txt = standardize_wer(ref_txt)
        ref_txt = ref_txt.strip().split(" ")

        words += len(ref_txt)
        scores += levenshtein(ref_txt, hyp_txt)

    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")

    rounded_wer = round(wer * 100, 4)
    print(f"wer: {rounded_wer}%, errors: {scores}, words: {words}")
    return {"WER %": rounded_wer, "Errors": scores, "Total words": words}


@beartype
def calculate_latency(
    ref_ctm: CTM,
    hyp_ctm: CTM,
    include_subs: bool = False,
    percentiles: List[int] = [90, 99],
) -> tuple[dict, dict]:
    """
    Calculate word-latency between in reference and hypothesis CTMs.
    The latency is defined as difference in seconds between the end
    of a reference word and start of the hypothesis word. This is really
    useful only for streaming ASRs when the reference was obtained offline
    whereas the hypothesis word timestamp is really an instant when the ASR
    returned the token. It is thus expected hypothesis words have
    duration = 0.

    Arguments:
    :ref_ctm: reference CTM object, use CTM.from_file(fpath) to load a ctm file
    :hyp_ctm: hypothesis CTM object, use CTM.from_file(fpath) to load a ctm file

    Raises
    ------
    ValueError: if the number of references is greater than the number of hypotheses
    """

    latencies, _, utt_stats = _align_transcripts(ref_ctm, hyp_ctm, include_subs)
    # Use frame-width zero because we are measuring UPL not EL
    latency_metrics = compute_latency_metrics(
        latencies=latencies,
        sil_latency=[],
        eos_latency=[],
        frame_width=0.0,
        percentiles=percentiles,
    )

    for k, v in latency_metrics.items():
        print(f"{k} | {v:.3f} s |")
    return latency_metrics, utt_stats


@beartype
def _align_transcripts(
    ref_ctm: CTM, hyp_ctm: CTM, include_subs: bool = False
) -> tuple[list, list, dict]:
    """
    Aligns the words from ground truth and predicted transcripts and calculates the
    emission latencies.
    """

    # Get transcript word-by-word and timestamps
    ref_tstamps = ref_ctm.get_timestamps()
    hyp_tstamps = hyp_ctm.get_timestamps()

    # Init vars
    ref_utts = len(ref_tstamps)
    done_utts, done_words = 0, 0
    latencies, end_times = [], []

    # Process each utterance separately
    for fname, ref_tstamp in ref_tstamps.items():
        if fname not in hyp_tstamps:
            continue
        hyp_tstamp = hyp_tstamps[fname]

        ref_words = [t[0] for t in ref_tstamp]
        hyp_words = [t[0] for t in hyp_tstamp]

        # Initialize the SequenceMatcher for each file
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # opcodes are instructions ('equal', 'replace', 'delete', 'insert')
            # for transforming ground_truth into predicted.
            # Each opcode (tag, i1, i2, j1, j2) specifies the operation and
            # start/end indices in both sequences.

            # Process according to the tag and include_subs flag
            if tag in ("equal", "replace") and (tag != "replace" or include_subs):
                latencies += [
                    (hyp_tstamp[j][1] + hyp_tstamp[j][2])
                    - (ref_tstamp[i][1] + ref_tstamp[i][2])
                    for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
                ]

                end_times += [
                    (ref_tstamp[i][1] + ref_tstamp[i][2]) for i in range(i1, i2)
                ]

                done_words += i2 - i1

        done_utts += 1

    assert done_utts > 0, (
        "Cannot find any server predictions\n"
        "Did you accidentally pass --skip_transcription?"
    )

    print(
        f"CTMs aligned; total utts: {ref_utts},"
        f" processed utts: {done_utts},"
        f" processed words: {done_words}"
    )
    return (
        latencies,
        end_times,
        {
            "Total utts": ref_utts,
            "Processed utts": done_utts,
        },
    )
