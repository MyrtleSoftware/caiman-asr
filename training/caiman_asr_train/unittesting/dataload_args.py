from argparse import Namespace
from pathlib import Path

import yaml
from beartype import beartype


@beartype
def _make_test_config(test_data_dir: Path, path: str) -> dict:
    with open(path, "r") as f:
        template = yaml.safe_load(f)

    # Replace placeholders for testing equivalents

    template["use_transcripts"]["to_use"] = None
    template["use_transcripts"]["on_missing"] = "raise_error"
    template["input_train"]["audio_dataset"]["max_duration"] = float("inf")
    template["input_train"]["audio_dataset"]["max_transcript_len"] = None
    template["input_train"]["audio_dataset"]["min_tps"] = 0.0
    template["input_train"]["audio_dataset"]["max_tps"] = float("inf")
    template["langs"] = ["en"]

    template["tokenizer"]["sentpiece_model"] = str(
        test_data_dir / "librispeech62.eos.pnc-tag.model"
    )

    for x in ["train", "val"]:
        template[f"input_{x}"]["filterbank_features"]["stats_path"] = str(test_data_dir)

    return template


def gen_dataload_args(test_data_dir, tmpdir) -> Namespace:
    """
    Produce a Namespace object with the default arguments for data loading tests.

    The default is the JSON format.
    """

    test_yaml = _make_test_config(test_data_dir, "configs/testing-1023sp.yaml")
    test_config = tmpdir / "testing-1023sp.yaml"

    with open(test_config, "w") as f:
        yaml.dump(test_yaml, f)

    return Namespace(
        val_manifests=[str(test_data_dir / "peoples-speech-short.cuts.jsonl")],
        val_dataset_dir=str(test_data_dir),
        val_batch_size=1,
        val_final_padding_secs=0.0,
        model_config=str(test_config),
        train_dataset_yaml=str(test_data_dir / "peoples-speech-short.yaml"),
        seed=1,
        prob_background_noise=0.0,
        conditional=False,
        sampling_mode="1D-bucketing-dyn",
        batch_duration=3600.0,
        num_dur_buckets=20,
        prob_room_aug=0.0,
        loader_workers_per_gpu=None,
        turn_off_initial_padding=False,
        inspect_audio=False,
        inspect_memory_usage=None,
        norm_over_utterance=False,
        norm_use_global_stats=False,
        norm_ramp_start=3600,
        norm_ramp_end=3600 * 2,
        eoi_drop_fraction=0.5,
        feature_dump_dir=None,
        val_standardizer_lang="en",
    )
