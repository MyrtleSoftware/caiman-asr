import copy
from argparse import Namespace
from pathlib import Path

from caiman_asr_train.data.decide_on_loader import DataSource


def gen_dataload_args(test_data_dir) -> Namespace:
    """
    Produce a Namespace object with the default arguments for data loading tests.

    The default is the JSON format.
    """
    manifest_fp = str(test_data_dir / "peoples-speech-short.json")
    return Namespace(
        grad_accumulation_batches=1,
        val_manifests=[manifest_fp],
        train_manifests=[manifest_fp],
        local_rank=0,
        num_buckets=6,
        dataset_dir=str(test_data_dir),
        dali_train_device="cpu",
        dali_val_device="cpu",
        prob_background_noise=0.0,
        prob_babble_noise=0.0,
        train_tar_files=None,
        val_tar_files=None,
        read_from_tar=False,
        seed=1,
        turn_off_initial_padding=True,
        inspect_audio=False,
        prob_val_narrowband=0.0,
        prob_train_narrowband=0.0,
        output_dir=Path("/results"),
        n_utterances_only=None,
        noise_dataset=None,
        use_noise_audio_folder=False,
        noise_config=None,
        val_from_dir=False,
        val_audio_dir=None,
        val_txt_dir=None,
        dali_processes_per_cpu=1,
        use_hugging_face=False,
        hugging_face_val_dataset=None,
        hugging_face_val_config=None,
        hugging_face_val_split="validation",
        hugging_face_val_transcript_key="text",
        norm_ramp_start_step=None,
        norm_ramp_end_step=None,
        norm_starting_ratio=0.25,
        norm_over_utterance=False,
        norm_use_global_stats=False,
        warmup_steps=1000,
        hold_steps=1000,
        half_life_steps=1000,
        val_final_padding_secs=0.0,
        train_manifest_ratios=None,
        relative_train_manifest_ratios=None,
        global_batch_size=1024,
        canary_exponent=None,
    )


def update_dataload_args(args: Namespace, format: DataSource) -> Namespace:
    """
    Util for updating the args from gen_dataload_args() depending on the DataSource.
    """
    args = copy.deepcopy(args)
    if format == DataSource.JSON:
        # assumes the default is json
        pass
    elif format == DataSource.TARFILE:
        # use all tar files in test_data_dir
        args.val_from_dir = False
        args.read_from_tar = True
        args.use_hugging_face = False
        args.train_tar_files = args.val_tar_files = ["webdataset-eg.tar"]
    elif format == DataSource.HUGGINGFACE:
        args.val_from_dir = False
        args.read_from_tar = False
        args.use_hugging_face = True
        args.hugging_face_val_dataset = "distil-whisper/librispeech_asr_dummy"
        args.hugging_face_val_split = "validation[:10]"
    else:
        raise ValueError(f"Unknown data source format: {format}")

    args.model_config = "configs/testing-1023sp_run.yaml"

    return args
