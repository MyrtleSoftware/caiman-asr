#!/usr/bin/env python3


import json
from pathlib import Path
from uuid import uuid4

from beartype import beartype

from caiman_asr_train.utils.fast_json import fast_read_json, fast_write_json


@beartype
def reference_read_json(json_path: str | Path) -> list[dict] | dict:
    with open(json_path, "r") as f:
        return json.load(f)


@beartype
def reference_write_json(python_structure, json_path: str | Path) -> None:
    with open(json_path, "w") as f:
        json.dump(python_structure, f, indent=2, ensure_ascii=False)


@beartype
def is_file_same(path1: str | Path, path2: str | Path) -> bool:
    with open(path1) as f1, open(path2) as f2:
        return f1.read() == f2.read()


def test_same_on_manifest(test_data_dir):
    original_path = str(test_data_dir / "peoples-speech-short.json")

    reference_path = Path("/tmp") / f"{uuid4()}.json"
    reference_manifest = reference_read_json(original_path)
    reference_write_json(reference_manifest, reference_path)

    fast_path = Path("/tmp") / f"{uuid4()}.json"
    fast_manifest = fast_read_json(original_path)
    fast_write_json(fast_manifest, fast_path)

    assert is_file_same(reference_path, fast_path)
    assert reference_manifest == fast_manifest


def test_mandarin():
    """Check that non-ASCII isn't escaped"""
    manifest = {"transcript": "你好"}
    manifest_path = Path("/tmp") / f"{uuid4()}.json"
    fast_write_json(manifest, manifest_path)
    assert manifest == fast_read_json(manifest_path)
