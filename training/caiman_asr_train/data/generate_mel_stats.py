import os
from argparse import Namespace

import numpy as np
import torch

from caiman_asr_train.args.norm_stats_generation import stats_generation_parse_args
from caiman_asr_train.data.build_dataloader import build_dali_loader
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.data.webdataset import LengthUnknownError
from caiman_asr_train.rnnt import config
from caiman_asr_train.setup.dali import build_dali_yaml_config
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


class TokenizerResultsIgnored(Tokenizer):
    """
    For convenience: a Tokenizer that returns a fixed value to be ignored.
    """

    def __init__(self):
        self.charset = list("abc")

    def tokenize(self, transcript):
        return [1]


def generate_stats(args: Namespace):
    """
    Record training data log mel stats and save them to disk.
    """

    assert args.train_manifest_ratios is None
    assert args.relative_train_manifest_ratios is None
    assert args.canary_exponent is None

    cfg = config.load(args.model_config)
    (dataset_kw, features_kw, _, _) = config.input(cfg, "train")
    update_config_mel_stats(dataset_kw, features_kw)
    user_symbols = list(get_all_user_tokens(cfg).values())
    dali_yaml_config = build_dali_yaml_config(
        config_data=dataset_kw, config_features=features_kw, user_symbols=user_symbols
    )
    train_loader = build_dali_loader(
        args,
        "train",
        batch_size=args.global_batch_size,
        dali_yaml_config=dali_yaml_config,
        tokenizer=TokenizerResultsIgnored(),
        world_size=1,
        mel_feat_normalizer=None,
        cpu=args.dali_train_device == "cpu",
    )

    meldim = features_kw["n_filt"]
    melsum = torch.zeros(meldim, dtype=torch.float64)
    melss = torch.zeros(meldim, dtype=torch.float64)
    meln = torch.zeros(1, dtype=torch.float64)

    try:
        total_loader_len = f"{len(train_loader):<10}"
    except LengthUnknownError:
        total_loader_len = "unk"

    for i, batch in enumerate(train_loader):
        print(
            f"({train_loader.pipeline_type} evaluation: {i:>10}/{total_loader_len}",
            end="\r",
        )
        logmel, logmel_lens, _, _, _, _ = batch
        melsum = melsum.to(logmel.device)
        melss = melss.to(logmel.device)
        meln = meln.to(logmel.device)

        # NOTE: no need to exclude padding since pad is 0 and we are summing
        melsum += torch.sum(logmel, (0, 2))
        melss += torch.sum(logmel * logmel, (0, 2))
        meln += torch.sum(logmel_lens)

    melmeans = melsum / meln
    melvars = melss / meln - melmeans * melmeans
    # calculated as doubles for precision; convert to float32s for use
    melmeans = melmeans.type(torch.FloatTensor)
    melvars = melvars.type(torch.FloatTensor)
    meln = meln.type(torch.FloatTensor)

    # test all variance values are positive
    z = np.zeros_like(melvars)
    np.testing.assert_array_less(
        z, melvars, "\nERROR : All variances should be positive\n"
    )

    output_dir = args.output_dir
    print(f"\nSaving generated mel stats to {output_dir}")
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # save as PyTorch tensors
    torch.save(melmeans, f"{output_dir}/melmeans.pt", pickle_protocol=5)
    torch.save(melvars, f"{output_dir}/melvars.pt", pickle_protocol=5)
    torch.save(meln, f"{output_dir}/meln.pt", pickle_protocol=5)


def update_config_mel_stats(dataset_kw, features_kw):
    # turn off dither to have deterministic stats
    features_kw["dither"] = 0.0
    # turn off augmentations
    dataset_kw["speed_perturbation"] = None
    dataset_kw["trim_silence"] = False

    # don't require transcript normalization as this would require a built
    # tokenizer instead of the TokenizerResultsIgnored object
    dataset_kw["normalize_transcripts"] = False


if __name__ == "__main__":
    args = stats_generation_parse_args()
    generate_stats(args)
