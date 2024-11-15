#!/usr/bin/env python3

from pathlib import Path

import orjson
from beartype import beartype

from caiman_asr_train.data.json_manifest import JSONDataset


@beartype
def fast_read_json(json_path: str | Path) -> list[dict] | dict:
    with open(json_path, "rb") as f:
        return orjson.loads(f.read())


@beartype
def fast_read_json_py(json_path: str | Path) -> JSONDataset:
    data = fast_read_json(json_path)
    return JSONDataset(entries=data)


@beartype
def fast_write_json(python_structure, json_path: str | Path) -> None:
    with open(json_path, "wb") as f:
        f.write(orjson.dumps(python_structure, option=orjson.OPT_INDENT_2))
