from caiman_asr_train.rnnt import config


def load_tokenizer_kw(config_fp, test_data_dir):
    cfg = config.load(config_fp)
    tokenizer_kw = config.tokenizer(cfg)
    # replace the spm with the one in the test data dir
    tokenizer_kw["sentpiece_model"] = str(test_data_dir / "librispeech29.model")
    return tokenizer_kw
