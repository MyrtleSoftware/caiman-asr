from argparse import Namespace

from beartype import beartype

from caiman_asr_train.args.shared import check_shared_args
from caiman_asr_train.args.train import train_arg_parser, verify_train_args


@beartype
def stats_generation_parse_args() -> Namespace:
    parser = train_arg_parser()
    parser.add_argument(
        "--mel_duration_sample",
        type=float,
        default=float("inf"),
        help="Limit the sample of features used to compute stats to this many seconds.",
    )
    parser.add_argument(
        "--mel_batch_duration",
        type=float,
        default=3600.0,
        help="Duration (in seconds) of features to process in a single batch when "
        "computing mel stats.",
    )

    args = parser.parse_args()

    # This arg is required by conf_build but then ignored.
    args.batch_duration = args.mel_batch_duration

    args = verify_train_args(args)
    check_shared_args(args, check_transducer=False)

    return args
