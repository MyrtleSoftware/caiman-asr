import json

from beartype import beartype
from cerberus import Validator

from caiman_asr_train.keywords.trie import Keywords


@beartype
def _patch_spaces(word: str) -> str:
    return word.replace(" ", "▁")


@beartype
def load_keywords(path: str) -> Keywords[str]:
    """
    Load a list of keywords from a json file.

    Expected format: dict[str, Number]
    """
    with open(path, "r") as f:
        keywords = json.load(f)

    schema = {
        "keywords": {
            "type": "dict",
            "keysrules": {"type": "string"},
            "valuesrules": {"type": "number"},
        }
    }

    v = Validator(require_all=True)

    if not v.validate(keywords, schema):
        raise ValueError(f"Schema not matched: {v.errors}")

    keywords = [(_patch_spaces(k), float(v)) for k, v in keywords["keywords"].items()]

    return Keywords(keywords)
