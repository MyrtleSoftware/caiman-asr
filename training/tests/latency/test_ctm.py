#!/usr/bin/env python3

import os
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

from caiman_asr_train.latency import ctm
from caiman_asr_train.latency.measure_latency import Latencies
from caiman_asr_train.latency.timestamp import EOS, PerWordTimestamp, SequenceTimestamp


def _make_sequence(words, final_time):
    return SequenceTimestamp(
        [PerWordTimestamp(word, start, end) for word, start, end in words],
        EOS(final_time),
    )


def test_to_ctm(tmp_path):
    output_fp = tmp_path / "out.ctm"
    seq = _make_sequence([("hello", 0, 1), ("world", 2, 3)], final_time=0.4)

    ctm.to_ctm(seq, str(output_fp), "audio.wav", frame_width=0.1, initial_pad=0.01)

    assert output_fp.read_text().splitlines() == [
        "audio.wav 1 0.000 0.190 hello ",
        "audio.wav 1 0.190 0.200 world ",
    ]


def test_dump_ctm(tmp_path):
    output_fp = tmp_path / "dump.ctm"
    sequences = [
        _make_sequence([("first", 0, 0)], final_time=0.1),
        _make_sequence([("second", 1, 2)], final_time=0.3),
    ]
    flist = ["sample1.wav", "sample2.wav"]

    termination_map = ctm.dump_ctm(
        flist, sequences, str(output_fp), frame_width=0.2, initial_pad=0.05
    )

    assert termination_map == {
        "sample1.wav": sequences[0].eos,
        "sample2.wav": sequences[1].eos,
    }
    assert output_fp.read_text().count("\n") == 2


@pytest.mark.parametrize(
    "path,is_absolute",
    [
        ("/var/tmp/audio.wav", True),
        ("rel/audio.wav", False),
    ],
)
def test_get_abs_path(tmp_path, path, is_absolute):
    directory = str(tmp_path / "base")
    expected = path if is_absolute else os.path.join(directory, path)

    assert ctm.get_abs_path(directory, path) == expected


def test_get_abs_manifest_paths(tmp_path):
    data_dir = tmp_path / "manifest"
    manifest_paths = ["first.json", "second.json", "/tmp/manifest/third.json"]
    abs_paths = ctm.get_abs_manifest_paths(str(data_dir), manifest_paths)

    assert abs_paths == [
        os.path.join(str(data_dir), "first.json"),
        os.path.join(str(data_dir), "second.json"),
        "/tmp/manifest/third.json",
    ]


def test_get_abs_tar_paths(tmp_path):
    data_dir = tmp_path / "data"
    tar_paths = ["a.tar", "b.tar", "/tmp/data/c.tar"]
    abs_paths = ctm.get_abs_tar_paths(str(data_dir), tar_paths)

    assert abs_paths == [
        "/tmp/data/c.tar",
        os.path.join(str(data_dir), "a.tar"),
        os.path.join(str(data_dir), "b.tar"),
    ]


def test_get_audio_filenames_from_tar(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path = data_dir / "audio.tar"

    files = {
        "clip_one.wav": "wav data",
        "clip_two.flac": "flac data",
        "notes.txt": "metadata",
    }

    for name, contents in files.items():
        (data_dir / name).write_text(contents)

    with tarfile.open(tar_path, "w") as archive:
        for name in files:
            archive.add(data_dir / name, arcname=name)

    filenames = ctm.get_audio_filenames_from_tar(str(data_dir), [tar_path.name])

    assert filenames == [["clip_one.wav", "clip_two.flac"]]


@pytest.mark.parametrize(
    "val_multiple",
    [False, True],
)
def test_get_reference_manifest_files(tmp_path, val_multiple):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    manifest = dataset_dir / "val.cuts.jsonl.gz"
    manifest.write_text("{}")

    args = Namespace(
        read_from_tar=False,
        val_tar_files=[],
        val_manifests=["val.cuts.jsonl.gz"],
        val_dataset_dir=str(dataset_dir),
        all_val_manifests=["val.cuts.jsonl.gz"],
        all_dataset_dirs=[str(dataset_dir)],
    )

    assert ctm.get_reference_manifest_files(args, val_multiple=val_multiple) == [
        str(manifest)
    ]


def test_manage_ctm_export(monkeypatch, tmp_path):
    args = Namespace(
        model_config="dummy.yaml",
        read_from_tar=False,
        val_tar_files=[],
        val_manifests=["first.manifest", "second.manifest"],
        output_dir=str(tmp_path),
        timestamp="123",
        latency_head_offset=0.0,
        latency_tail_offset=0.0,
        turn_off_initial_padding=False,
        val_standardizer_lang="en",
    )

    sequences = [
        _make_sequence([("hello", 0, 0), ("world", 1, 1)], final_time=0.5),
        _make_sequence([("one", 1, 1), ("two", 2, 2), ("three", 3, 3)], final_time=1.0),
    ]
    flist = ["audio1.wav", "audio2.wav"]
    gt_ctm_files = [str(tmp_path / "ref.ctm")]

    monkeypatch.setattr(ctm, "encoder_output_frame_width", lambda _: 0.06)
    monkeypatch.setattr(ctm, "initial_padding", lambda _: 0.015)
    monkeypatch.setattr(ctm.dist, "is_initialized", lambda: False)

    captured = {}

    def fake_align_ctm_files(
        gt_ctm_fpaths,
        produced_ctm,
        last_emit_time,
        lang_code,
        head_offset,
        tail_offset,
    ):
        captured["produced_ctm"] = produced_ctm
        captured["last_emit_time"] = last_emit_time

        latencies = Latencies(
            raw_latency=[0.1, 0.2],
            fw_latency=[0.15],
            sil_latency=[0.3],
            eos_latency=[0.4],
        )
        return (latencies, "unused", 0.8, 0.9, {"timestamp_metric": 1.0})

    monkeypatch.setattr(ctm, "align_ctm_files", fake_align_ctm_files)
    monkeypatch.setattr(
        ctm, "compute_latency_metrics", lambda *_, **__: {"avg_latency": 0.15}
    )

    (
        latency_metrics,
        timestamp_metrics,
        latencies,
    ) = ctm.manage_ctm_export(args, sequences, gt_ctm_files, flist)

    ctm_path = Path(args.output_dir) / "first-second_123.ctm"
    assert captured["produced_ctm"] == str(ctm_path)
    assert captured["last_emit_time"] == {
        "audio1.wav": sequences[0].eos,
        "audio2.wav": sequences[1].eos,
    }
    assert latency_metrics == {
        "avg_latency": 0.15,
        "token_usage_rate": 0.8,
        "terminal_token_usage_rate": 0.9,
    }
    assert timestamp_metrics == {"timestamp_metric": 1.0}
    assert latencies.raw_latency == [0.1, 0.2]
    assert latencies.fw_latency == [0.15]
    assert latencies.sil_latency == [0.3]
    assert latencies.eos_latency == [0.4]
    assert ctm_path.exists()
