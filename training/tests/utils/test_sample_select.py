import pytest

from caiman_asr_train.utils.select_transcripts import SelectTranscript


@pytest.fixture
def manifest_entry():
    return {
        "transcript": "A",
        "alt-1": "B",
        "alt-2": "C",
    }


def test_default_select(manifest_entry):
    s = SelectTranscript()
    assert s(manifest_entry) == "A"

    with pytest.raises(KeyError):
        s({})


@pytest.mark.parametrize("mode", ["raise_error", "use_default", "skip"])
def test_ok_select(manifest_entry, mode):
    config = {
        "use_transcripts": {
            "to_use": "transcript",
            "on_missing": mode,
        }
    }

    s = SelectTranscript(config)
    assert s(manifest_entry) == "A"


@pytest.mark.parametrize("mode", ["raise_error", "use_default", "skip"])
def test_fallback_select(manifest_entry, mode):
    config = {
        "use_transcripts": {
            "to_use": ["fish", "alt-2"],
            "on_missing": mode,
        }
    }

    s = SelectTranscript(config)
    assert s(manifest_entry) == "C"


@pytest.mark.parametrize("mode", ["raise_error", "use_default", "skip"])
def test_invalid_select(manifest_entry, mode):
    config = {
        "use_transcripts": {
            "to_use": "fish",
            "on_missing": mode,
        }
    }

    s = SelectTranscript(config)

    match mode:
        case "raise_error":
            with pytest.raises(KeyError):
                s(manifest_entry)
        case "use_default":
            assert s(manifest_entry) == "A"
        case "skip":
            assert s(manifest_entry) is None
