#!/usr/bin/env python3

from beartype import beartype
from beartype.typing import Dict, List, Tuple

from caiman_asr_train.data.text.is_tag import is_tag


@beartype
def get_all_user_tokens(config: Dict) -> Tuple[Dict[str, str], List[str]]:
    """
    Fetch and validate all user tokens/symbols from the config.
    """

    langs = config.get("langs", [])
    out = {f"lang_{lang}": f"<lang_{lang}>" for lang in langs} if len(langs) > 1 else {}

    user_tokens = config.get("user_tokens") or {}

    # required for kenlm data-prep
    if not user_tokens:
        return {}, []

    assert isinstance(user_tokens, dict), "User tokens must be a dictionary."

    if "lang_detect" in user_tokens and user_tokens["lang_detect"] is None:
        user_tokens.pop("lang_detect")

    for k, v in user_tokens.items():
        assert isinstance(k, str), f"User token key {k} must be a string."

        if v is None:
            continue

        assert isinstance(v, str), f"User token value {v} must be a string."
        assert is_tag(v), f"User token {v} is not in valid form."

        out[k] = v

    return out, list(out.values())
