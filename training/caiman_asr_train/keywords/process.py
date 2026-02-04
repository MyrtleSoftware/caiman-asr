import json

from beartype import beartype
from beartype.typing import List
from cerberus import Validator

from caiman_asr_train.keywords.trie import Keywords
from caiman_asr_train.train_utils.distributed import print_once


@beartype
def _patch_spaces(word: str) -> str:
    return word.replace(" ", "▁")


@beartype
def load_and_validate_keywords_json(path: str) -> List[str]:
    """
    Load keywords from a json file and validate the schema.
    """
    with open(path, "r") as f:
        keywords = json.load(f)

    schema = {
        "keywords": {
            "type": "list",
            "schema": {"type": "string"},
        }
    }

    v = Validator(require_all=True)

    if not v.validate(keywords, schema):
        raise ValueError(f"Schema not matched: {v.errors}")

    return keywords["keywords"]


@beartype
def load_keywords(
    path: str, c_0: float, beta: float, context_score: float
) -> Keywords[str]:
    """
    Load a list of keywords from a json file.

    Expected format: dict[str, Number]
    """
    keywords = load_and_validate_keywords_json(path)

    keywords = [_patch_spaces(k) for k in keywords]

    return Keywords(keywords, c_0, beta, context_score)


@beartype
def return_list_of_keywords(path: str) -> List[str]:
    """
    Load a list of keywords from a json file,
    filtering out those that contain more than one word.

    This is used for the keyword boosting metrics.
    """
    keywords = load_and_validate_keywords_json(path)

    filtered_keywords = []
    for k in keywords:
        if len(k.split()) > 1:
            print_once(
                f"Warning: Keyword '{k}' contains more than one word "
                "and will be excluded from metrics."
            )
        else:
            filtered_keywords.append(k.strip())

    return filtered_keywords
