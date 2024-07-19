import os

import pytest

from caiman_asr_train.lm.prep_kenlm_data import (
    extract_transcripts_from_manifests,
    extract_transcripts_from_tars,
)
from caiman_asr_train.rnnt import config
from caiman_asr_train.setup.text_normalization import normalize_config_from_full_yaml


def load_text_file_to_list(file_path):
    return open(file_path).read().splitlines()


@pytest.mark.parametrize(
    "dataset, saved_txt_file, read_from_tar",
    [
        ("peoples-speech-short.json", "ngram/ps-short_ls29.txt", False),
        ("webdataset-eg.tar", "ngram/webd-eg_ls29.txt", True),
    ],
)
def test_prep_kenlm_data(
    test_data_dir,
    tokenizer,
    tmp_path,
    mini_config_fp,
    dataset,
    saved_txt_file,
    read_from_tar,
):
    cfg = config.load(mini_config_fp)
    labels = cfg["tokenizer"]["labels"]
    normalize_config = normalize_config_from_full_yaml(cfg)

    saved_txt_filepath = os.path.join(str(test_data_dir), saved_txt_file)
    loaded_txt = load_text_file_to_list(saved_txt_filepath)

    if read_from_tar:
        transcripts = extract_transcripts_from_tars(
            [dataset], str(test_data_dir), tokenizer, normalize_config, labels
        )
    else:
        transcripts = extract_transcripts_from_manifests(
            [dataset], str(test_data_dir), tokenizer, normalize_config, labels
        )

    token_sentences = [
        " ".join(tokenizer.sentpiece.id_to_piece(token) for token in sentence)
        for sentence in transcripts
    ]
    assert token_sentences == loaded_txt
