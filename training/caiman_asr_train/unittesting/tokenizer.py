import yaml

from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.config import get_tokenizer_conf


def _update_langs(tmp_path, cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Update the LANGS key
    cfg["langs"] = ["en"]

    # save the config to a temporary file
    named_cfg_path = str(tmp_path / "test_config.yaml")
    with open(named_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    return named_cfg_path


def load_tokenizer_kw(tmp_path, config_fp, test_data_dir):
    mini_cfg_fp = _update_langs(tmp_path, config_fp)
    cfg = config.load(mini_cfg_fp)
    tokenizer_kw = get_tokenizer_conf(cfg)
    # replace the spm with the one in the test data dir
    tokenizer_kw["sentpiece_model"] = str(test_data_dir / "librispeech29.model")
    tokenizer_kw["label_smoothing_kwargs"]["dict_path"] = str(
        test_data_dir / "words.txt"
    )
    return tokenizer_kw
