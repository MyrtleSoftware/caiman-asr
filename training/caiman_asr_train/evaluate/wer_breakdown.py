#!/usr/bin/env python3

from beartype import beartype

from caiman_asr_train.evaluate.distributed_utils import multigpu_wer
from caiman_asr_train.train_utils.distributed import print_once


@beartype
def print_wer_breakdown(
    hypotheses: list[str], references: list[str], breakdown_chars: str
) -> None:
    def get_wer(transformation):
        return multigpu_wer(
            [transformation(h) for h in hypotheses],
            [transformation(r) for r in references],
            standardize=False,
        )

    # Handle lowercasing separately from punctuation
    results = [("case", get_wer(lambda x: x.lower()))]
    for p in breakdown_chars:
        results.append((f"'{p}'", get_wer(lambda x: x.replace(p, ""))))
    results.append(
        (f"'{breakdown_chars}'", get_wer(lambda x: remove_all(x, breakdown_chars)))
    )

    unstd_wer = get_wer(lambda x: x)

    print_once("")
    print_once("WER % (relative improvement %)")
    print_once("-" * 30)
    print_once(
        f"Unstandardized: {unstd_wer*100:5.3f}% "
        f"({relative_improvement_percent(unstd_wer, unstd_wer):5.3f}%)"
    )
    for p, wer in results:
        print_once(
            f"Ignore {p}: {wer*100:5.3f}% "
            f"({relative_improvement_percent(unstd_wer, wer):5.3f}%)"
        )


@beartype
def remove_all(string: str, to_removestr) -> str:
    return "".join(c for c in string if c not in to_removestr)


@beartype
def relative_improvement_percent(original, better) -> float:
    return 100 * (original - better) / original
