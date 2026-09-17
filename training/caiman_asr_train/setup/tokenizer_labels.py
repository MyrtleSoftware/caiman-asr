from caiman_asr_train.data.text.allowed_charset import get_allowed_charset


def set_tokenizer_labels(cfg: dict) -> dict:
    """Set tokenizer labels

    Labels are not specified in the config file but generated from language ID.
    This function adds the "labels" field to the tokenizer config based on the
    "langs" field if "labels" is not already specified.
    The "labels" field is checked for backward compatibility and is mutually
    exclusive with "langs".

    Parameters
    ----------
    cfg : dict
        Config dictionary
    """
    assert ("labels" in cfg["tokenizer"]) ^ (
        "langs" in cfg
    ), "tokenizer.labels is mutually exclusive with cfg.langs"

    if "labels" not in cfg["tokenizer"]:
        cfg["tokenizer"]["labels"] = get_allowed_charset(langs=cfg["langs"])

    assert all(
        char not in cfg["tokenizer"]["labels"] for char in "<>"
    ), "Tokenizer labels should not include angle brackets"

    return cfg
