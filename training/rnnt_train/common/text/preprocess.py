# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from beartype.typing import List, Optional, Union

from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.text.normalize_file import normalize


def norm_and_tokenize(
    transcript: str,
    tokenizer: Optional[Tokenizer],
    normalize_transcripts: bool,
    charset: Optional[List[str]] = None,
) -> Union[List[int], str]:
    """
    Normalizes and optionally tokenizes a transcript.
    """
    if normalize_transcripts:
        charset = tokenizer.charset if tokenizer is not None else charset
        assert (
            charset is not None
        ), "Must either pass tokenizer or charset but both are None"
        transcript_ = normalize(transcript, quiet=False, charset=charset)
        if not transcript_:
            raise ValueError(f"Transcript normalization failed for {transcript=}.")
        transcript = transcript_

    if tokenizer is None:
        return transcript
    return tokenizer.tokenize(transcript)
