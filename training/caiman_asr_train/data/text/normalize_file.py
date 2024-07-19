#!/usr/bin/env python3

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023, Myrtle Software Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from beartype import beartype

from caiman_asr_train.data.text.normalizers import lowercase_normalize


@beartype
def normalize_list(transcripts, charset: list[str], quiet: bool = False):
    return [lowercase_normalize(t, charset=charset, quiet=quiet) for t in transcripts]


@beartype
def normalize_by_line(text, charset: list[str], quiet: bool = False):
    """Split text at \n, clean each line,
    and then glue together with spaces.
    Hence most of the document will be normalized
    even if one line is unnormalizable
    and needs to be deleted"""
    return " ".join(normalize_list(text.split("\n"), charset=charset, quiet=quiet))
