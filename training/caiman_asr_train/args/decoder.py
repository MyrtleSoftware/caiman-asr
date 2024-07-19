from argparse import ArgumentParser

from beartype import beartype


@beartype
def add_decoder_args(parser: ArgumentParser) -> None:
    decoder = parser.add_argument_group("decoder setup")
    decoder.add_argument(
        "--decoder",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="decoder to use",
    )
    decoder.add_argument(
        "--beam_width", default=4, type=int, help="beam width when using beam decoder"
    )
    decoder.add_argument(
        "--temperature",
        default=1.4,
        type=float,
        help=(
            "Softmax temperature to use during decoding. Increasing this above 1.0 should "
            "increase diversity of beam hypotheses. Not used for greedy decoding as it "
            "doesn't change the greedy ordering."
        ),
    )
    decoder.add_argument(
        "--beam_prune_score_thresh",
        default=0.4,
        type=float,
        help=(
            "Pruning threshold for adaptive beam search. At the start of each timestep, "
            "hypotheses with a score per token that is `beam_prune_score_thresh` less "
            "than the best hypothesis' score will be pruned."
            "Reducing the value will make the pruning more aggressive and setting it "
            "< 0 will disable it. Note "
            "that it is necessary to set --beam_prune_topk_thresh to a value > 0 "
            "in addition to setting this parameter > 0 in order to see significant "
            "pruning."
        ),
    )
    decoder.add_argument(
        "--beam_prune_topk_thresh",
        default=1.5,
        type=float,
        help=(
            "Pruning threshold for the hypothesis expansion step. Tokens with a "
            "logprob score `beam_prune_topk_thresh` less than the most likely token "
            "will not be considered."
            "Reducing the value will make the pruning more aggressive and setting it "
            "< 0 will disable it. Note "
            "that it is necessary to set --beam_prune_score_thresh to a value > 0 "
            "in addition to setting this parameter > 0 in order to see significant "
            "pruning."
        ),
    )
    decoder.add_argument(
        "--fuzzy_topk_logits",
        action="store_true",
        default=False,
        help=(
            "Reduce the logits tensor to match the "
            "tensor that the hardware-accelerated solution is based on."
        ),
    )
    add_ngram_args(parser)


@beartype
def add_ngram_args(parser: ArgumentParser) -> None:
    ngram = parser.add_argument_group("ngram setup")
    ngram.add_argument(
        "--override_ngram_path",
        default=None,
        type=str,
        help="Path to KenLM n-gram file (.arpa or .binary).",
    )
    ngram.add_argument(
        "--skip_ngram",
        default=False,
        action="store_true",
        help="Disable n-gram shallow fusion with beam search decoding.",
    )
