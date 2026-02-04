#!/usr/bin/env python3

import pytest

from caiman_asr_train.data.text.preprocess import generate_symbols_mask
from caiman_asr_train.data.text.standardize_manifest import (
    standardize_manifest_transcripts,
)


@pytest.fixture
def mock_manifest():
    """Create a mock manifest with various transcript fields."""
    return [
        {
            "transcript": "Hello, this is a Test.",
            "model_output": "Hello, this is a Test too!",
            "special": "Another Test with <EOS> token.",
        },
        {
            "transcript": "Dr. Smith visited the cafe.",
            "model_output": "Professor Jones went to university.",
            "special": "Mr. <EOS> Johnson works at <EOS> Company.",
        },
        {
            "transcript": "ald. jones and ald Smith joined Lt. col Hanson.",
            "special": "ald. jones and Ald Smith joined Lt. Col Hanson. <star> <EOS>",
        },
        {
            "special": (
                "hmmm? Capitalise the next word. ummm.. hi. <EOS> "
                "And ummm what about this? Hmm. Yes that's, umm. <EOS> <star> "
                "No. Um, um, well. <EOS>"
            )
        },
        {"special": "MHMM <EOS> Hi. umm. I am a dr. Hmm. Ummm! <EOS> Mmmhmm <star>"},
    ]


def test_basic_standardization(mock_manifest):
    """Test basic standardization with default settings."""
    result = standardize_manifest_transcripts(
        mock_manifest, transcript_fields=["transcript"], num_workers=1
    )

    assert result[0]["transcript-standardized"] == "hello this is a test"
    assert result[1]["transcript-standardized"] == "doctor smith visited the cafe"
    assert (
        result[2]["transcript-standardized"]
        == "alderman jones and alderman smith joined lieutenant colonel hanson"
    )
    assert "transcript-standardized" not in result[3]


def test_multiple_fields(mock_manifest):
    """Test standardization of multiple transcript fields."""
    result = standardize_manifest_transcripts(
        mock_manifest, transcript_fields=["transcript", "model_output"], num_workers=1
    )

    assert result[0]["transcript-standardized"] == "hello this is a test"
    assert result[0]["model_output-standardized"] == "hello this is a test too"
    assert result[1]["transcript-standardized"] == "doctor smith visited the cafe"
    assert (
        result[1]["model_output-standardized"] == "professor jones went to university"
    )
    assert (
        result[2]["transcript-standardized"]
        == "alderman jones and alderman smith joined lieutenant colonel hanson"
    )

    assert "special-standardized" not in result[0]
    assert "special-standardized" not in result[1]
    assert "special-standardized" not in result[2]
    assert "model_output-standardized" not in result[2]
    assert "model_output-standardized" not in result[3]


@pytest.mark.parametrize(
    (
        "preserve_case, preserve_punctuation, expected_transcript0, "
        "expected_model_output0, expected_transcript1, expected_model_output1"
    ),
    [
        (
            True,
            False,
            "Hello this is a Test",
            "Hello this is a Test too",
            "Doctor Smith visited the cafe",
            "Professor Jones went to university",
        ),
        (
            False,
            True,
            "hello, this is a test.",
            "hello, this is a test too!",
            "doctor smith visited the cafe.",
            "professor jones went to university.",
        ),
        (
            True,
            True,
            "Hello, this is a Test.",
            "Hello, this is a Test too!",
            "Doctor Smith visited the cafe.",
            "Professor Jones went to university.",
        ),
    ],
)
def test_preserve_options(
    mock_manifest,
    preserve_case,
    preserve_punctuation,
    expected_transcript0,
    expected_model_output0,
    expected_transcript1,
    expected_model_output1,
):
    """Test case and punctuation preservation options."""
    result = standardize_manifest_transcripts(
        mock_manifest,
        transcript_fields=["transcript", "model_output"],
        num_workers=1,
        preserve_case=preserve_case,
        preserve_punctuation=preserve_punctuation,
    )

    assert result[0]["transcript-standardized"] == expected_transcript0
    assert result[0]["model_output-standardized"] == expected_model_output0
    assert result[1]["transcript-standardized"] == expected_transcript1
    assert result[1]["model_output-standardized"] == expected_model_output1


def test_user_symbols_handling(mock_manifest):
    """Test handling of user symbols with and without symbols_mask."""
    # Without symbols_mask - <EOS> will be removed
    result = standardize_manifest_transcripts(
        mock_manifest, transcript_fields=["special"], num_workers=1
    )

    assert result[0]["special-standardized"] == "another test with token"
    assert result[1]["special-standardized"] == "mister johnson works at company"
    assert (
        result[2]["special-standardized"]
        == "alderman jones and alderman smith joined lieutenant colonel hanson"
    )
    assert result[3]["special-standardized"] == (
        "capitalize the next word hi " "and what about this yes that is " "no well"
    )
    assert result[4]["special-standardized"] == "hi i am a doctor"

    # With symbols_mask - <EOS> will be preserved
    symbols_mask = generate_symbols_mask(["<EOS>"])
    result = standardize_manifest_transcripts(
        mock_manifest,
        transcript_fields=["special"],
        num_workers=1,
        symbols_mask=symbols_mask,
    )

    assert result[0]["special-standardized"] == "another test with <EOS> token"
    assert (
        result[1]["special-standardized"]
        == "mister <EOS> johnson works at <EOS> company"
    )
    assert (
        result[2]["special-standardized"]
        == "alderman jones and alderman smith joined lieutenant colonel hanson <EOS>"
    )
    assert result[3]["special-standardized"] == (
        "capitalize the next word hi <EOS> "
        "and what about this yes that is <EOS> "
        "no well <EOS>"
    )
    assert result[4]["special-standardized"] == "<EOS> hi i am a doctor <EOS>"

    # + <star> token
    symbols_mask = generate_symbols_mask(["<EOS>", "<star>"])
    result = standardize_manifest_transcripts(
        mock_manifest,
        transcript_fields=["special"],
        num_workers=1,
        symbols_mask=symbols_mask,
    )

    assert result[0]["special-standardized"] == "another test with <EOS> token"
    assert (
        result[1]["special-standardized"]
        == "mister <EOS> johnson works at <EOS> company"
    )
    assert (
        result[2]["special-standardized"]
        == "alderman jones and alderman smith joined lieutenant colonel hanson <star> <EOS>"
    )
    assert result[3]["special-standardized"] == (
        "capitalize the next word hi <EOS> "
        "and what about this yes that is <EOS> <star> "
        "no well <EOS>"
    )
    assert result[4]["special-standardized"] == "<EOS> hi i am a doctor <EOS> <star>"

    # + preserve casing + punctuation
    result = standardize_manifest_transcripts(
        mock_manifest,
        transcript_fields=["special"],
        num_workers=1,
        preserve_case=True,
        preserve_punctuation=True,
        symbols_mask=symbols_mask,
    )

    assert result[0]["special-standardized"] == "Another Test with <EOS> token."
    assert (
        result[1]["special-standardized"]
        == "Mister <EOS> Johnson works at <EOS> Company."
    )
    assert result[2]["special-standardized"] == (
        "alderman jones and Alderman Smith joined "
        "Lieutenant Colonel Hanson. <star> <EOS>"
    )
    assert result[3]["special-standardized"] == (
        "Capitalize the next word. Hi. <EOS> "
        "And what about this? Yes that is. <EOS> <star> "
        "No. Well. <EOS>"
    )
    assert result[4]["special-standardized"] == "<EOS> Hi. I am a doctor. <EOS> <star>"
