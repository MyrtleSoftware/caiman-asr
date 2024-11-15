#!/usr/bin/env python3

from beartype import beartype
from beartype.typing import Dict

from caiman_asr_train.data.text.is_tag import is_tag


@beartype
def get_all_user_tokens(config: Dict) -> Dict[str, str]:
    """
    Fetch and validate all user tokens/symbols from the config.
    """

    if "user_tokens" not in config:
        return {}

    if config["user_tokens"] is None:
        return {}

    assert isinstance(config["user_tokens"], dict), "User tokens must be a dictionary."

    out = {}

    for k, v in config["user_tokens"].items():
        assert isinstance(k, str), f"User token key {k} must be a string."

        if v is None:
            continue

        assert isinstance(v, str), f"User token value {v} must be a string."
        assert is_tag(v), f"User token {v} is not in valid form."

        out[k] = v

    return out
