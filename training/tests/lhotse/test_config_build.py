from pathlib import Path

import pytest

from caiman_asr_train.lhotse.config_build import (
    _parse_dataset_yaml,
    _select_train_langs,
)

MULTILINGUAL_YAML = Path(__file__).parent.parent / "test_data" / "multilingual.yaml"


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({}, None),
        ({"langs": ["en", "fr"]}, ["en", "fr"]),
        ({"langs": ["en", "fr"], "ft_langs": ["fr"]}, ["fr"]),
        ({"langs": ["en", "fr"], "ft_langs": []}, ["en", "fr"]),
        # ft_langs takes precedence over langs even if langs is absent.
        ({"ft_langs": ["fr"]}, ["fr"]),
    ],
)
def test_select_train_langs(cfg, expected):
    assert _select_train_langs(cfg) == expected


def test_parse_dataset_yaml_defaults_to_all_manifest_langs():
    manifests = _parse_dataset_yaml(MULTILINGUAL_YAML)
    assert set(manifests.manifests.keys()) == {"en", "fr"}


def test_parse_dataset_yaml_filters_by_langs():
    manifests = _parse_dataset_yaml(MULTILINGUAL_YAML, langs=["en"])
    assert set(manifests.manifests.keys()) == {"en"}


def test_parse_dataset_yaml_rejects_unknown_lang():
    with pytest.raises(AssertionError, match="Requested languages not found"):
        _parse_dataset_yaml(MULTILINGUAL_YAML, langs=["de"])
