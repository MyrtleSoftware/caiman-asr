import random
from functools import lru_cache

import numpy as np
from beartype import beartype
from beartype.typing import Optional
from lhotse import SupervisionSegment

from caiman_asr_train.data.text.normalizers import MemoizedLowercase, language_normalize
from caiman_asr_train.data.text.symbols_mask import generate_symbols_mask
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig


@beartype
class MemoizedNormalize:
    def __init__(
        self,
        config: NormalizeConfig,
        charset: list[str],
        max_cache_size: Optional[int] = None,
    ):
        """
        A caching configurable normalizer.

        This class is required because applying lru_cache directly to a method
        is a footgun: https://rednafi.com/python/lru_cache_on_methods/

        Args:
            config: NormalizeConfig object containing normalization settings.
            charset: List of allowed characters for normalization.
            max_cache_size: Maximum size of the LRU cache. If None unbounded.
        """

        self.config = config
        self.charset = charset
        self.mask = generate_symbols_mask(config.user_symbols)
        self.normalize = lru_cache(maxsize=max_cache_size)(self._normalize)

    def _normalize(self, text: str, lang_code: str) -> str:
        text = language_normalize(text, self.charset, self.config, lang_code, self.mask)
        return text

    def __call__(self, text: str, lang_code: str) -> str:
        return self.normalize(text, lang_code)


@beartype
class Tokenize:
    def __init__(self, tokenizer: Tokenizer):
        """
        Simple wrapper around a tokenizer that adds a "tokens" custom field.
        """
        self.tokenizer = tokenizer

    def __call__(self, seg: SupervisionSegment) -> SupervisionSegment:
        return seg.with_custom(
            "tokens", np.asarray(self.tokenizer.tokenize(_get_text(seg)))
        )


@beartype
def _get_user_token(user_symbol: str, tokenizer: Tokenizer) -> int:
    toks = tokenizer.tokenize(user_symbol)
    # Space token at the start
    assert len(toks) == 2
    # Remove space token
    toks = toks[1:]
    # Sanity
    assert len(toks) == 1
    return toks[0]


@beartype
class PrefixTokenize:
    def __init__(
        self,
        tokenizer: Tokenizer,
        user_symbols: dict[str, str],
        pnc_prob: float,
        max_cache_size: Optional[int] = None,
        languages: Optional[dict[str, str]] = None,
        lang_mask_prob: Optional[float] = None,
    ):
        """
        Wrapper around a tokenizer that adds a "tokens" custom field, this
        wrapper prefixes the transcript with either a "pnc" or "nopnc" token
        on-the-fly before tokenization. When selecting the "nopnc" prefix the
        input text is lowercased first, the result of this lowercase
        normalization is cached.
        Note that the text does not change, only the tokens change depending
        on the prefix.

        Args:
            tokenizer: The tokenizer to use.
            user_symbols: A dictionary containing at least "pnc" and "nopnc" keys.
            pnc_prob: Probability of using the "pnc" prefix.
            max_cache_size: Maximum size of the LRU cache. If None unbounded.
            languages: A dictionary from language codes to language tokens. If supplied,
                the language token will be included in the prefix.
            lang_mask_prob: Probability of masking the language token with <lang_detect>.

        """
        if "pnc" not in user_symbols or "nopnc" not in user_symbols:
            raise ValueError(
                "'pnc' or 'nopnc' missing from user_symbols_dict. "
                "Add them to `user_symbols` in the model config (.yaml), "
                "and ensure your tokenizer has the corresponding tokens "
                "(you may need to train a new tokenizer)."
            )

        self.pnc_tok = _get_user_token(user_symbols["pnc"], tokenizer)
        self.nopnc_tok = _get_user_token(user_symbols["nopnc"], tokenizer)
        if lang_mask_prob is not None and lang_mask_prob > 0:
            assert user_symbols.get("lang_detect") is not None
            self.lang_detect_tok = _get_user_token(
                user_symbols["lang_detect"], tokenizer
            )

        # Sanity
        assert self.pnc_tok != self.nopnc_tok

        self.tokenizer = tokenizer
        self.pnc_prob = pnc_prob
        self.lang_mask_prob = lang_mask_prob

        if languages is None:
            self.lang_toks = None
        else:
            self.lang_toks = {
                k: _get_user_token(v, tokenizer) for k, v in languages.items()
            }

        self.lowercase = MemoizedLowercase(
            list(user_symbols.values()), max_cache_size=max_cache_size
        )

    def __call__(self, seg: SupervisionSegment) -> SupervisionSegment:

        assert (
            seg.language is not None
        ), "SupervisionSegment must have a language for PrefixTokenize"
        pnc_text = _get_text(seg)
        nopnc_text = self.lowercase(pnc_text, seg.language)

        if nopnc_text is None:
            # Implies already lowercase and clean -> use nopnc prefix
            prefix_tok = self.nopnc_tok
            text = pnc_text
        elif random.random() < self.pnc_prob:
            prefix_tok = self.pnc_tok
            text = pnc_text
        else:
            prefix_tok = self.nopnc_tok
            text = nopnc_text

        if self.lang_toks is None:
            prefix = [prefix_tok]
        elif seg.language is None:
            raise ValueError("No language to prefix, is `append_language` enabled?")
        elif (lang_tok := self.lang_toks.get(seg.language)) is None:
            raise ValueError(f"{seg.language=} is not in {list(self.lang_toks.keys())}")
        elif self.lang_mask_prob is not None and random.random() < self.lang_mask_prob:
            prefix = [prefix_tok, self.lang_detect_tok]
        else:
            prefix = [prefix_tok, lang_tok]

        tokens = self.tokenizer.tokenize(text)

        seg = seg.with_custom("prefix", np.asarray(prefix))
        seg = seg.with_custom("tokens", np.asarray(tokens))
        return seg


@beartype
def _get_text(seg: SupervisionSegment) -> str:
    """
    Get the text to process from a SupervisionSegment.
    The preferred entry is "normalized_text",
    or the default "text" field if the former is not available.
    Raises an error if neither is available.
    """

    if seg.has_custom("normalized_text"):
        return seg.normalized_text
    elif seg.text is not None:
        return seg.text
    else:
        raise ValueError("No text to process")
