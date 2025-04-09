from argparse import Namespace

from beartype import beartype

from caiman_asr_train.args.shared import check_shared_args
from caiman_asr_train.args.train import train_arg_parser, verify_train_args


@beartype
def stats_generation_parse_args() -> Namespace:
    parser = train_arg_parser()
    parser.add_argument(
        "--dump_mel_stats_batch_size",
        type=int,
        default=128,
        help="Batch size for dumping mel stats. Will override the global batch size",
    )
    args = parser.parse_args()
    args = update_args_stats_generation(args, args.dump_mel_stats_batch_size)

    args = verify_train_args(args)
    check_shared_args(args)

    return args


def update_args_stats_generation(args: Namespace, batch_size) -> Namespace:
    """
    This function overrides training arg defaults for stats generation.
    """
    args.prob_background_noise = 0.0
    args.prob_babble_noise = 0.0
    args.prob_train_narrowband = 0.0

    args.global_batch_size = batch_size
    args.num_gpus = 1
    args.num_buckets = 1

    # to avoid the DaliDataLoader.drop_last condition removing some utterances:
    args.grad_accumulation_batches = 1

    return args
