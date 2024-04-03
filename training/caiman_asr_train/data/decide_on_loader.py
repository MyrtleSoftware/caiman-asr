#!/usr/bin/env python3
from enum import Enum

from beartype import beartype


class DataSource(Enum):
    JSON = 1
    TARFILE = 2
    HUGGINGFACE = 3


@beartype
def decide_on_loader(
    val_from_dir: bool, read_from_tar: bool, use_hugging_face: bool
) -> DataSource:
    assert (
        sum([val_from_dir, read_from_tar, use_hugging_face]) <= 1
    ), "At most one of val_from_dir, read_from_tar, and hugging_face can be True"
    if read_from_tar:
        return DataSource.TARFILE
    if use_hugging_face:
        return DataSource.HUGGINGFACE
    return DataSource.JSON
