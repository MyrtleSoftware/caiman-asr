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
import multiprocessing as mp
import random
import re
import string

import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, List, Optional, Union

from caiman_asr_train.data.text.normalizers import select_and_normalize
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig


def mask_user_symbols(text: str, symbols_mask: Optional[Dict[str]] = None) -> str:
    """
    :text is a string of word separated by " "
    :symbols_mask is a dict of form {rnd_string: word_to_mask}
    """
    if symbols_mask is not None:
        for masker, maskee in symbols_mask.items():
            text = re.sub(re.escape(maskee), masker, text)
    return text


def unmask_user_symbols(text: str, symbols_mask: Optional[Dict[str]] = None) -> str:
    """
    :text is a string of word separated by " "
    :symbols_mask is a dict of form {rnd_string: word_to_mask}
    """
    if symbols_mask is not None:
        for masker, maskee in symbols_mask.items():
            text = re.sub(masker, maskee, text)
    return text


@beartype
def norm_and_tokenize(
    transcript: str,
    normalize_config: NormalizeConfig,
    tokenizer: Optional[Tokenizer] = None,
    charset: Optional[List[str]] = None,
) -> Union[List[int], str]:
    """
    Normalizes and optionally tokenizes a transcript. `Transcript` is a string,
    typically a single utterance.
    """
    # Define a mask to hide `user_symbols`` from normalization
    symbols_mask = None
    user_symbols = normalize_config.user_symbols
    if user_symbols:
        symbols_mask = {
            "".join(random.choice(string.ascii_lowercase) for _ in range(20)): sym
            for sym in user_symbols
        }
        assert len(symbols_mask) == len(user_symbols)

    charset = tokenizer.charset if tokenizer is not None else charset

    msg = "Must either pass tokenizer or charset but both are None"

    assert charset is not None, msg

    transcript = mask_user_symbols(transcript, symbols_mask)
    transcript = select_and_normalize(transcript, charset, normalize_config)
    transcript = unmask_user_symbols(transcript, symbols_mask)

    if tokenizer is None:
        return transcript

    return tokenizer.tokenize(transcript)


@beartype
def norm_and_tokenize_parallel(
    transcripts: List[str],
    normalize_config: NormalizeConfig,
    tokenizer: Optional[Tokenizer] = None,
    charset: Optional[List[str]] = None,
    min_trans_per_process: int = 50,
) -> List[List[int] | str]:
    """
    Parallelized version of `norm_and_tokenize()`.

    Args:
    :transcripts: a list of trascripts to tokenize
    :tokenizer: a tokenizer object
    :normalize_config: Controls how to normalize transcripts prior to tokenization
    :charset: a character set for tokenization in case of a missing tokenizer
    :min_trans_per_process: minimum number of transcripts per process to
        use multiprocessing, otherwise default to single-processing
    """
    # To circumvent overallocating CPUs because each DaliRnntIterator()
    # runs as a separate process if we run multi GPU training
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    num_processes = max(1, int(mp.cpu_count() / world_size))

    trans_num = len(transcripts)
    if min_trans_per_process * num_processes >= trans_num or mp.cpu_count() < 2:
        results = [
            norm_and_tokenize(t, normalize_config, tokenizer, charset)
            for t in transcripts
        ]
    else:
        # spawn processes and collect results
        with mp.get_context("spawn").Pool(processes=num_processes) as pool:
            args = [(t, normalize_config, tokenizer, charset) for t in transcripts]
            results = pool.starmap(norm_and_tokenize, args)

    msg = f"Some transcripts failed to tokenize ({trans_num} vs. {len(results)})"
    assert trans_num == len(results), msg

    return results
