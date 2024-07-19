import ruamel.yaml

from caiman_asr_train.args.val import val_arg_parser
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun
from caiman_asr_train.val import build_objects, run_validate


def update_scale_factor(yaml_path: str, scale_factor: float):
    yaml = ruamel.yaml.YAML()
    with open(yaml_path, "r") as file:
        config = yaml.load(file)

    config["ngram"]["scale_factor"] = scale_factor

    with open(yaml_path, "w") as file:
        yaml.dump(config, file)


def check_beam_args(args):
    if args.decoder != "beam":
        if args.local_rank == 0:
            print(
                (
                    "WARNING: The intended use of this script is to sweep across n-gram "
                    "scale factor, which requires beam decoding. Setting --decoder to beam."
                )
            )
        args.decoder = "beam"


def main(args):
    best_scale_factor = None
    best_wer = float("inf")

    for i, scale_factor in enumerate(args.scale_factors):
        if i > 0:
            args.skip_init = True

        update_scale_factor(args.model_config, scale_factor)
        val_objects, profilers = build_objects(args)
        results = run_validate(args, val_objects)
        if results["wer"] < best_wer:
            best_wer = results["wer"]
            best_scale_factor = scale_factor

    print_once(
        (
            f"Best WER: {best_wer*100:.3f}%. "
            f"Best scale factor: {best_scale_factor}. "
            f"Updating `scale_factor` in model config YAML to {best_scale_factor}"
        )
    )
    update_scale_factor(args.model_config, best_scale_factor)


if __name__ == "__main__":
    parser = val_arg_parser()
    parser.add_argument(
        "--scale_factors",
        type=float,
        default=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25],
        nargs="+",
    )
    args = parser.parse_args()
    if not args.cpu:
        maybe_restart_with_torchrun(
            args.num_gpus,
            args.called_by_torchrun,
            "/workspace/training/caiman_asr_train/lm/sweep_scale_factor.py",
        )
    check_beam_args(args)
    main(args)
