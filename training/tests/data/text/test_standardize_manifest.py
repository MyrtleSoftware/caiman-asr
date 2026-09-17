#!/usr/bin/env python3

import pytest
import yaml
from beartype.typing import Dict, Optional
from lhotse import CutSet, MonoCut, SupervisionSegment

from caiman_asr_train.data.text.standardize_manifest import (
    LhotseManifestStandardizer,
)


@pytest.fixture
def make_inputs(tmp_path, tmp_path_factory):
    def _make_inputs(text: str, custom: Optional[Dict[str, str]] = None):
        sup = SupervisionSegment(
            id="sup1",
            recording_id="rec1",
            start=0.0,
            duration=1.0,
            text=text,
        )
        if custom:
            for key in custom:
                sup = sup.with_custom(key, custom[key])
        cut = MonoCut(id="cut1", start=0, duration=1.0, channel=0, supervisions=[sup])

        path = tmp_path / f"{cut.id}.jsonl"
        CutSet.from_cuts([cut]).to_file(path)
        output_dir = tmp_path_factory.mktemp("output")
        out_path = output_dir / f"{cut.id}.jsonl"
        return path, out_path

    return _make_inputs


@pytest.fixture
def model_config(tmp_path):
    def _model_config(user_tokens: Dict[str, str] = {}):
        config = {"user_tokens": user_tokens}
        path = tmp_path / "test_model_config.yaml"
        with open(path, "w") as tmp_f:
            yaml.dump(config, tmp_f)
        return str(path)

    return _model_config


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Hello, this is a Test.", "hello this is a test"),
        ("Dr. Smith visited the cafe.", "doctor smith visited the cafe"),
        (
            "ald. jones and ald Smith joined Lt. col Hanson.",
            "alderman jones and alderman smith joined lieutenant colonel hanson",
        ),
    ],
)
def test_basic_standardization(input_text, expected, make_inputs, model_config):
    path, out_path = make_inputs(input_text)

    standardizer = LhotseManifestStandardizer(
        model_config=model_config(),
        num_workers=1,
        lang_code="en",
    )

    standardizer.process_cutset(str(path), str(out_path.parent))

    processed = CutSet.from_jsonl(str(out_path))
    sup = next(iter(processed)).supervisions[0]
    assert sup.custom["transcript-standardized"] == expected


custom_transcripts = [
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
]

expected_outputs = [
    {
        "transcript-standardized": "hello this is a test",
        "model_output-standardized": "hello this is a test too",
    },
    {
        "transcript-standardized": "doctor smith visited the cafe",
        "model_output-standardized": "professor jones went to university",
    },
    {
        "transcript-standardized": "alderman jones and alderman smith "
        "joined lieutenant colonel hanson"
    },
]


test_inputs = zip(expected_outputs, custom_transcripts)


@pytest.mark.parametrize("expected, custom", test_inputs)
def test_multiple_fields(expected, custom, make_inputs, model_config):
    """Test standardization of multiple transcript fields."""

    path, out_path = make_inputs("Test", custom)

    standardizer = LhotseManifestStandardizer(
        model_config=model_config(),
        num_workers=1,
        lang_code="en",
    )
    transcript_fields = ["transcript", "model_output"]
    standardizer.process_cutset(
        str(path), str(out_path.parent), transcript_fields=transcript_fields
    )

    processed = CutSet.from_jsonl(str(out_path))
    sup = next(iter(processed)).supervisions[0]

    for transcript_field in transcript_fields:
        standardized_field = f"{transcript_field}-standardized"
        if standardized_field in expected:
            assert sup.custom[standardized_field] == expected[standardized_field]

    assert "special-standardized" not in sup.custom


sentences = [
    "Hello, this is a Test.",
    "Hello, this is a Test too!",
    "Dr. Smith visited the cafe.",
    "Professor Jones went to university.",
]

options = [
    (True, False),
    (False, True),
    (True, True),
]


@pytest.mark.parametrize("input_text", sentences)
@pytest.mark.parametrize("preserve_case, preserve_punctuation", options)
def test_preserve_options(
    input_text, preserve_case, preserve_punctuation, make_inputs, model_config
):
    def mock_standardize(text, preserve_case, preserve_punctuation):
        t = text

        # expansion
        if "Dr." in t:
            t = t.replace("Dr.", "Doctor")

        # case handling
        if not preserve_case:
            t = t.lower()

        # punctuation handling
        if not preserve_punctuation:
            t = "".join(c for c in t if c.isalnum() or c.isspace())

        return t.strip()

    path, out_path = make_inputs(input_text)

    standardizer = LhotseManifestStandardizer(
        model_config=model_config(),
        num_workers=1,
        preserve_case=preserve_case,
        preserve_punctuation=preserve_punctuation,
        lang_code="en",
    )

    standardizer.process_cutset(str(path), str(out_path.parent))

    processed = CutSet.from_jsonl(str(out_path))
    sup = next(iter(processed)).supervisions[0]

    assert sup.custom["transcript-standardized"] == mock_standardize(
        input_text, preserve_case, preserve_punctuation
    )


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Another Test with <EOS> token.", "another test with token"),
        (
            "Mr. <EOS> Johnson works at <EOS> Company.",
            "mister johnson works at company",
        ),
        (
            "ald. jones and Ald Smith joined Lt. Col Hanson. <star> <EOS>",
            "alderman jones and alderman smith joined lieutenant colonel hanson",
        ),
        (
            (
                "hmmm? Capitalise the next word. ummm.. hi. <EOS> "
                "And ummm what about this? Hmm. Yes that's, umm. <EOS> <star> "
                "No. Um, um, well. <EOS>"
            ),
            (
                "capitalize the next word hi "
                "and what about this yes that is "
                "no well"
            ),
        ),
        (
            "MHMM <EOS> Hi. umm. I am a dr. Hmm. Ummm! <EOS> Mmmhmm <star>",
            "hi i am a doctor",
        ),
    ],
)
def test_no_user_symbol(input_text, expected, make_inputs, model_config):
    path, out_path = make_inputs(input_text)

    standardizer = LhotseManifestStandardizer(
        model_config=model_config(),
        num_workers=1,
        expand_contractions=True,
        lang_code="en",
    )

    standardizer.process_cutset(str(path), str(out_path.parent))

    processed = CutSet.from_jsonl(str(out_path))
    sup = next(iter(processed)).supervisions[0]

    assert sup.custom["transcript-standardized"] == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Another Test with <EOS> token.", "another test with <EOS> token"),
        (
            "Mr. <EOS> Johnson works at <EOS> Company.",
            "mister <EOS> johnson works at <EOS> company",
        ),
        (
            "ald. jones and Ald Smith joined Lt. Col Hanson. <star> <EOS>",
            "alderman jones and alderman smith joined lieutenant colonel hanson <EOS>",
        ),
        (
            (
                "hmmm? Capitalise the next word. ummm.. hi. <EOS> "
                "And ummm what about this? Hmm. Yes that's, umm. <EOS> <star> "
                "No. Um, um, well. <EOS>"
            ),
            (
                "capitalize the next word hi <EOS> "
                "and what about this yes that is <EOS> "
                "no well <EOS>"
            ),
        ),
        (
            "MHMM <EOS> Hi. umm. I am a dr. Hmm. Ummm! <EOS> Mmmhmm <star>",
            "<EOS> hi i am a doctor <EOS>",
        ),
    ],
)
def test_single_user_symbol(input_text, expected, make_inputs, model_config):
    path, out_path = make_inputs(input_text)

    standardizer = LhotseManifestStandardizer(
        model_config=model_config({"eos": "<EOS>"}),
        num_workers=1,
        expand_contractions=True,
        lang_code="en",
    )

    standardizer.process_cutset(str(path), str(out_path.parent))

    processed = CutSet.from_jsonl(str(out_path))
    sup = next(iter(processed)).supervisions[0]

    assert sup.custom["transcript-standardized"] == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Another Test with <EOS> token.", "another test with <EOS> token"),
        (
            "Mr. <EOS> Johnson works at <EOS> Company.",
            "mister <EOS> johnson works at <EOS> company",
        ),
        (
            ("ald. jones and Ald Smith" "joined Lt. Col Hanson. <star> <EOS>"),
            (
                "alderman jones and alderman smith"
                "joined lieutenant colonel hanson <star> <EOS>"
            ),
        ),
        (
            (
                "hmmm? Capitalise the next word. ummm.. hi. <EOS> "
                "And ummm what about this? Hmm. Yes that's, umm. <EOS> <star> "
                "No. Um, um, well. <EOS>"
            ),
            (
                "capitalize the next word hi <EOS> "
                "and what about this yes that is <EOS> <star> "
                "no well <EOS>"
            ),
        ),
        (
            "MHMM <EOS> Hi. umm. I am a dr. Hmm. Ummm! <EOS> Mmmhmm <star>",
            "<EOS> hi i am a doctor <EOS> <star>",
        ),
    ],
)
def test_multiple_user_symbols(input_text, expected, make_inputs, model_config):
    path, out_path = make_inputs(input_text)

    standardizer = LhotseManifestStandardizer(
        model_config=model_config({"eos": "<EOS>", "star": "<star>"}),
        num_workers=1,
        expand_contractions=True,
        lang_code="en",
    )

    standardizer.process_cutset(str(path), str(out_path.parent))

    processed = CutSet.from_jsonl(str(out_path))
    sup = next(iter(processed)).supervisions[0]

    assert sup.custom["transcript-standardized"] == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Another Test with <EOS> token.", "Another Test with <EOS> token."),
        (
            "Mr. <EOS> Johnson works at <EOS> Company.",
            "Mister <EOS> Johnson works at <EOS> Company.",
        ),
        (
            ("ald. jones and Ald Smith" " joined Lt. Col Hanson. <star> <EOS>"),
            (
                "alderman jones and Alderman Smith"
                " joined Lieutenant Colonel Hanson. <star> <EOS>"
            ),
        ),
        (
            (
                "hmmm? Capitalise the next word. ummm.. hi. <EOS> "
                "And ummm what about this? Hmm. Yes that's, umm. <EOS> <star> "
                "No. Um, um, well. <EOS>"
            ),
            (
                "Capitalize the next word. Hi. <EOS> "
                "And what about this? Yes that is. <EOS> <star> "
                "No. Well. <EOS>"
            ),
        ),
        (
            "MHMM <EOS> Hi. umm. I am a dr. Hmm. Ummm! <EOS> Mmmhmm <star>",
            "<EOS> Hi. I am a doctor. <EOS> <star>",
        ),
    ],
)
def test_multiple_user_symbols_preserve_options(
    input_text, expected, make_inputs, model_config
):
    path, out_path = make_inputs(input_text)

    standardizer = LhotseManifestStandardizer(
        model_config=model_config({"eos": "<EOS>", "star": "<star>"}),
        num_workers=1,
        preserve_case=True,
        preserve_punctuation=True,
        expand_contractions=True,
        lang_code="en",
    )

    standardizer.process_cutset(str(path), str(out_path.parent))

    processed = CutSet.from_jsonl(str(out_path))
    sup = next(iter(processed)).supervisions[0]

    assert sup.custom["transcript-standardized"] == expected
