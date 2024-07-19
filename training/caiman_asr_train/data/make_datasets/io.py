#!/usr/bin/env python
# Copyright (c) 2024, Myrtle.ai. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import tarfile
from pathlib import Path
from typing import Union

import requests
from tqdm.auto import tqdm


def download_file(
    url: str, filepath: Union[str, Path], force_download: bool = False
) -> None:
    """
    Download URL to a file. Does not support resume rather creates a
    temp file which is renamed after download finishes.
    """
    filepath = Path(filepath)

    if filepath.is_file():
        if force_download:
            filepath.unlink()
            print(
                f"""
                {filepath} exists but downloading from scratch
                because `force_download` = True.
            """
            )
        else:
            print(
                f"{filepath} exists but skipping download because `force_download` = False."
            )
            return

    temp_filepath = Path(str(filepath) + ".tmp")

    req = requests.get(url, stream=True)
    file_size = int(req.headers["Content-Length"])
    chunk_size = 1024 * 1024  # 1MB
    total_chunks = int(file_size / chunk_size)

    with open(temp_filepath, "wb") as fp:
        content_iterator = req.iter_content(chunk_size=chunk_size)
        for chunk in tqdm(
            content_iterator,
            total=total_chunks,
            unit="MB",
            desc=str(filepath),
            leave=True,
        ):
            fp.write(chunk)

    temp_filepath.rename(filepath)


def md5_checksum(filepath: Union[str, Path], target_hash: str) -> bool:
    """
    Do MD5 checksum.
    """
    filepath = Path(filepath)

    file_hash = hashlib.md5()
    with open(filepath, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest() == target_hash


def extract_tar(filepath: Union[str, Path], data_dir: Union[str, Path]) -> None:
    """
    Extract tar files into a folder.
    """
    filepath = Path(filepath)
    data_dir = Path(data_dir)

    if filepath.suffixes == [".tar", ".gz"]:
        mode = "r:gz"
    elif filepath.suffix == ".tar":
        mode = "r:"
    else:
        raise IOError(f"filepath has unknown extension {filepath}")

    with tarfile.open(filepath, mode) as tar:
        members = tar.getmembers()
        for member in tqdm(iterable=members, total=len(members), leave=True):
            tar.extract(path=str(data_dir), member=member)
