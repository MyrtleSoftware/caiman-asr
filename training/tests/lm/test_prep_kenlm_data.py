import os
from argparse import Namespace

from caiman_asr_train.lm.prep_kenlm_data import main


def load_text_file_to_list(file_path):
    return open(file_path).read().splitlines()


def test_prep_kenlm_data(
    test_data_dir,
    tmp_path,
    dataload_args,
):
    args = Namespace(
        model_config=dataload_args.model_config,
        train_dataset_yaml=dataload_args.train_dataset_yaml,
        max_utterances=None,
        output_path=str(tmp_path / "kenlm_input.txt"),
        batch_size=1,
        inspect_memory_usage=None,
        cpu=True,
    )

    main(args)

    with open(args.output_path) as f:
        produced_lines = f.read().splitlines()

    loaded_txt = load_text_file_to_list(
        os.path.join(str(test_data_dir), "ngram/ps-short_ls29.txt")
    )

    for p in produced_lines:
        assert p in loaded_txt
