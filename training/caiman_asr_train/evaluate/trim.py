from dataclasses import dataclass
from itertools import count

import torch
from beartype import beartype
from beartype.typing import List, Optional, Tuple

from caiman_asr_train.evaluate.state_resets.timestamp import (
    Timestamp,
    user_perceived_time,
)
from caiman_asr_train.latency.timestamp import EOS, Never, Silence, Termination


@beartype
@dataclass
class EOSTrimConfig:
    """
    Information for EOS trimming.
    """

    eos_idx: int
    eos_is_terminal: bool
    blank_idx: int


@beartype
def trim_predictions(
    pred: List[List[int]],
    timestamps: List[List[Timestamp]],
    probs: List[List[float]],
    pre_enc_width: float,
    post_enc_width: float,
    feat_lens: torch.Tensor,
    eos_vad_threshold: float,
    eos_info: Optional[EOSTrimConfig] = None,
) -> Tuple[
    List[List[int]],
    List[List[Timestamp]],
    List[List[float]],
    List[Termination],
]:
    """
    Trim predictions to remove tokens after n-consecutive blanks.

    Trim predictions to remove EOS token and any tokens after it.

    Also computes the frame/time at which the model stopped emitting tokens.
    """

    o_pred = []
    o_timestamps = []
    o_probs = []
    o_emit = []

    for y, t, p, worst in zip(pred, timestamps, probs, feat_lens.tolist()):
        # Options: no eos, eos missing, eos in prediction.

        assert len(y) == len(t) == len(p), f"Got {len(y)} {len(t)} {len(p)}"

        # First check for silence termination.

        proc_end = worst * pre_enc_width

        if not t:
            # Special cases: no tokens implies all blanks
            # if worst > consecutive_blanks then we would have
            # terminated with silence.

            o_pred.append(y)
            o_timestamps.append(t)
            o_probs.append(p)

            if proc_end > eos_vad_threshold:
                o_emit.append(Silence(eos_vad_threshold))
            else:
                o_emit.append(Never())

            continue

        term = Never()

        # It doesn't matter that the code checks for silence before EOS.
        # Since each check truncates tokens after the silence/EOS,
        # the reported Termination will always be the earliest one,
        # just as in the streaming case

        if eos_vad_threshold != float("inf"):
            last_tok = (user_perceived_time(t[-1]) + 1) * post_enc_width
            sil_frames = round(eos_vad_threshold / post_enc_width)

            # Look for n-consecutive blanks at end.
            if proc_end - last_tok > eos_vad_threshold:
                term = Silence(last_tok + eos_vad_threshold)

            # Look for n-consecutive blanks in middle.
            # Don't check for n consecutive blanks before the first non-blank,
            # as the speaker shouldn't be cut off before they say anything

            for idx, t_prev, t_idx in zip(count(1), t[:-1], t[1:]):
                t_prev = user_perceived_time(t_prev)
                t_idx = user_perceived_time(t_idx)

                if t_idx - t_prev > sil_frames:
                    frames = t_prev + 1 + sil_frames

                    y = y[:idx]
                    t = t[:idx]
                    p = p[:idx]

                    term = Silence(frames * post_enc_width)
                    break

        if eos_info is not None:
            if eos_info.eos_is_terminal:
                # Trim at the first EOS token and report EOS.
                idx = find(eos_info.eos_idx, y)
            else:
                # If the last non-blank is EOS then report it as EOS
                # termination but don't trim the prediction at earlier
                # EOS tokens.

                idx = None

                # Get the index of the first EOS token whose successors are only blanks/EOS,
                # if such an EOS token exists
                for i in range(len(y) - 1, -1, -1):
                    match y[i]:
                        case eos_info.eos_idx:
                            idx = i
                        case eos_info.blank_idx:
                            continue
                        case _:
                            break

            if idx is not None:
                # Plus one because zero indexed.
                term = EOS((user_perceived_time(t[idx]) + 1) * post_enc_width)

                assert y[idx] == eos_info.eos_idx

                # Trim after EOS.
                idx += 1

                y = y[:idx]
                t = t[:idx]
                p = p[:idx]

        o_pred.append(y)
        o_timestamps.append(t)
        o_probs.append(p)
        o_emit.append(term)

    return o_pred, o_timestamps, o_probs, o_emit


@beartype
def find(val: int, lst: List[int]) -> Optional[int]:
    """
    Return first list.index() or None if not present.
    """
    try:
        return lst.index(val)
    except ValueError:
        return None
