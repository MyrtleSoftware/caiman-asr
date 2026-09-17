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
from beartype import beartype
from beartype.typing import List, Optional, Union

from caiman_asr_train.data.text.normalizers import language_normalize
from caiman_asr_train.data.text.symbols_mask import generate_symbols_mask
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig


# only used in myrtle-data
@beartype
def norm_and_tokenize(
    transcript: str,
    normalize_config: NormalizeConfig,
    lang_code: str,
    tokenizer: Optional[Tokenizer] = None,
    charset: Optional[List[str]] = None,
) -> Union[List[int], str]:
    """
    Normalizes and optionally tokenizes a transcript. `Transcript` is a string,
    typically a single utterance.
    """
    # Define a mask to hide `user_symbols`` from normalization
    user_symbols = normalize_config.user_symbols
    symbols_mask = generate_symbols_mask(user_symbols)

    charset = tokenizer.charset if tokenizer is not None else charset

    msg = "Must either pass tokenizer or charset but both are None"

    assert charset is not None, msg

    transcript = language_normalize(
        transcript, charset, normalize_config, lang_code, symbols_mask
    )

    if tokenizer is None:
        return transcript

    return tokenizer.tokenize(transcript)
