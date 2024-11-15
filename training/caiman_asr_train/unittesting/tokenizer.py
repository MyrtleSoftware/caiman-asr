from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.config import get_tokenizer_conf


def load_tokenizer_kw(config_fp, test_data_dir):
    cfg = config.load(config_fp)
    tokenizer_kw = get_tokenizer_conf(cfg)
    # replace the spm with the one in the test data dir
    tokenizer_kw["sentpiece_model"] = str(test_data_dir / "librispeech29.model")
    return tokenizer_kw
