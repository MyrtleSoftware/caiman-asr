from beartype import beartype
from beartype.typing import Dict, Optional, Union

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


@beartype
def get_user_token(
    tok: str, config: Dict, tokenizer: Optional[Tokenizer]
) -> Optional[Union[int, str]]:
    """
    Fetch and validate a user token from the config. If a tokenizer is
    provided, return the index of the token in the tokenizer's vocab.
    """

    all_toks = get_all_user_tokens(config)

    if tok not in all_toks:
        return None

    sym = config["user_tokens"][tok]

    if tokenizer is None:
        return sym

    # The first token is a space, so we need to skip it
    try:
        meta_idx = tokenizer.tokenize(sym)[1]
    except Exception:
        raise ValueError(
            f"Failed to tokenize user token {tok}:{sym}, "
            "maybe you intended to pass --eos_decoding none"
            " and remove EOS from the config?"
        )

    return meta_idx
