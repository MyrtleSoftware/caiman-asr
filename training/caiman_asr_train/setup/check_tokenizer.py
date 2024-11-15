#!/usr/bin/env python3

from beartype import beartype

from caiman_asr_train.train_utils.distributed import warn_once


@beartype
def check_tokenizer_kw(
    checkpoint_tokenizer_kw: dict | None, config_tokenizer_kw: dict
) -> None:
    if checkpoint_tokenizer_kw is None:
        warn_once("Cannot check if you are using the correct tokenizer")
        return
    ckpt_kw = get_relevant_kw(checkpoint_tokenizer_kw)
    cfg_kw = get_relevant_kw(config_tokenizer_kw)
    if ckpt_kw != cfg_kw:
        raise ValueError(
            f"The checkpoint's previous tokenizer keywords were {ckpt_kw}\n"
            f"But the config file says {cfg_kw}"
        )


@beartype
def get_relevant_kw(tokenizer_kw: dict) -> dict:
    """Other information like 'sampling' is allowed to change"""
    return {
        "sentpiece_model": tokenizer_kw["sentpiece_model"],
        "labels": tokenizer_kw["labels"],
    }
