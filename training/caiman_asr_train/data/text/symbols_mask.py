import random
import re
import string

from beartype import beartype
from beartype.typing import Dict, Optional


@beartype
def mask_user_symbols(text: str, symbols_mask: Optional[Dict[str, str]] = None) -> str:
    """
    :text is a string of word separated by " "
    :symbols_mask is a dict of form {rnd_string: word_to_mask}
    """
    if symbols_mask is not None:
        for masker, maskee in symbols_mask.items():
            text = re.sub(re.escape(maskee), masker, text)
    return text


@beartype
def unmask_user_symbols(
    text: str, symbols_mask: Optional[Dict[str, str]] = None
) -> str:
    """
    :text is a string of word separated by " "
    :symbols_mask is a dict of form {rnd_string: word_to_mask}
    """
    if symbols_mask is not None:
        for masker, maskee in symbols_mask.items():
            text = re.sub(masker, maskee, text, flags=re.IGNORECASE)
    return text


@beartype
def generate_symbols_mask(
    user_symbols: Optional[list[str]] = None,
) -> Optional[dict[str, str]]:
    """
    Generate a mask to hide `user_symbols`` from normalization
    """
    if user_symbols:
        symbols_mask = {
            "".join(random.choice(string.ascii_lowercase) for _ in range(20)): sym
            for sym in user_symbols
        }
        assert len(symbols_mask) == len(user_symbols)
        return symbols_mask
    return None
