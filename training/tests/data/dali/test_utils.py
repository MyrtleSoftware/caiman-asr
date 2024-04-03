from caiman_asr_train.data.dali.utils import _parse_json


def test_parse_json(test_data_dir):
    assert test_data_dir is not None
    out_files, transcripts = _parse_json(
        str(test_data_dir / "peoples-speech-short.json")
    )
    assert len(out_files) == 2
    assert len(transcripts) == 2
    assert "duplicate_clip.flac" in out_files.keys()
    assert (
        "gov_DOT_uscourts_DOT_ca9_DOT_04-56618_DOT_2006-02-16_DOT_mp3_00027.flac"
        in out_files.keys()
    )
    assert {"label": 0, "duration": 8.89} in out_files.values()
