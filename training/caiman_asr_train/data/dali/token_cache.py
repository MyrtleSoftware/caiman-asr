#!/usr/bin/env python3
import random
import string
from dataclasses import dataclass

import torch
from beartype.typing import List, Tuple

from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.text.normalizers import lowercase_normalize
from caiman_asr_train.data.text.preprocess import (
    generate_symbols_mask,
    mask_user_symbols,
    norm_and_tokenize_parallel,
    unmask_user_symbols,
)
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig


def is_clean_lowercase(transcript: str) -> bool:
    allowed_punctuation = "'<>"
    for char in transcript:
        if char in string.punctuation and char not in allowed_punctuation:
            return False
    return transcript.islower()


@dataclass
class TranscriptPair:
    original: str
    lowercased: str
    original_prefix: List[int]
    original_prefix_len: int
    lowercased_prefix: List[int]
    lowercased_prefix_len: int

    def get_pnc(self) -> Tuple[str, List[int], int]:
        return self.original, self.original_prefix, self.original_prefix_len

    def get_nopnc(self) -> Tuple[str, List[int], int]:
        return self.lowercased, self.lowercased_prefix, self.lowercased_prefix_len


class NormalizeCache:
    """
    When using json manifests, this class normalizes the
    transcripts at the start of training for speed.

    When using Webdataset files/Hugging Face, DALI outputs transcripts.
    This class decodes them.

    Args:
        transcripts: in form of `{0: 'trans_1', ...}`
    """

    def __init__(
        self,
        normalize_config: NormalizeConfig,
        tokenizer: Tokenizer,
        device_type: str,
        data_source: DataSource,
        transcripts: dict,
        conditional: bool,
        pnc_prob: float = 0.25,
    ):
        self.tokenizer = tokenizer
        self.device_type = device_type
        self.data_source = data_source
        self.conditional = conditional
        self.pnc_prob = pnc_prob
        self.normalize_config = normalize_config

        if self.data_source is not DataSource.JSON:
            return

        self.raw_transcripts_cache = transcripts

        # Normalize (+ standardize) all transcripts once at start of training,
        # but don't tokenize as we use on the fly tokenization
        # with token sampling and label smoothing.
        self.norm_transcripts = norm_and_tokenize_parallel(
            [transcripts[i] for i in range(len(transcripts))],
            normalize_config,
            charset=self.tokenizer.charset,
        )

        if self.conditional:
            assert {"pnc", "nopnc"} <= normalize_config.user_symbols_dict.keys(), (
                "pnc and nopnc missing from user_symbols_dict. "
                "Add them to `user_symbols` in the model config (.yaml), "
                "and ensure your tokenizer has the corresponding tokens "
                "(you may need to train a new tokenizer)."
            )
            pnc = normalize_config.user_symbols_dict["pnc"]
            nopnc = normalize_config.user_symbols_dict["nopnc"]
            pnc_tokens = tokenizer.tokenize(pnc)
            nopnc_tokens = tokenizer.tokenize(nopnc)

            assert pnc_tokens[0] == nopnc_tokens[0]  # space token
            # Remove space token
            pnc_tokens = pnc_tokens[1:]
            nopnc_tokens = nopnc_tokens[1:]

            pnc_tokens_len = len(pnc_tokens)
            nopnc_tokens_len = len(nopnc_tokens)

            assert pnc_tokens_len == nopnc_tokens_len == 1, (
                "pnc and nopnc tokens must have length 1. "
                f"Got {pnc_tokens_len} and {nopnc_tokens_len} respectively."
            )
            assert pnc_tokens[0] != nopnc_tokens[0], (
                "pnc and nopnc tokens must be different. "
                f"Got {pnc_tokens[0]} and {nopnc_tokens[0]} respectively."
            )

            symbols_mask = generate_symbols_mask(normalize_config.user_symbols)
            lowercase_charset = list(" '" + string.ascii_lowercase)

            # Create transcript pairs: original (pnc) and lowercased (nopnc)
            self.transcript_pairs = []
            for i in range(len(self.norm_transcripts)):
                transcript = self.norm_transcripts[i]
                transcript_masked = mask_user_symbols(transcript, symbols_mask)
                is_already_lowercased = is_clean_lowercase(transcript_masked)

                if is_already_lowercased:
                    # pnc and nopnc are the same
                    pair = TranscriptPair(
                        transcript,
                        transcript,
                        nopnc_tokens,
                        nopnc_tokens_len,
                        nopnc_tokens,
                        nopnc_tokens_len,
                    )
                else:
                    lowercased = lowercase_normalize(
                        transcript_masked, lowercase_charset
                    )
                    lowercased = unmask_user_symbols(lowercased, symbols_mask)
                    pair = TranscriptPair(
                        transcript,
                        lowercased,
                        pnc_tokens,
                        pnc_tokens_len,
                        nopnc_tokens,
                        nopnc_tokens_len,
                    )

                self.transcript_pairs.append(pair)

    def get_transcripts(
        self, data: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        """
        Generate transcripts in format expected by training and evaluation loops
        """
        labels = data["label"]

        if self.data_source is not DataSource.JSON:
            # labels is the actual transcript
            transcripts = labels
            sizes = data["label_lens"].squeeze(1)
            raw_transcripts = [unicode_to_str(seq) for seq in data["raw_transcript"]]
            prefix_lens = [0] * len(transcripts)  # ignore prefix lengths for now
        else:
            # labels refers to an id that can be used to retrieve the transcript
            ids = labels.flatten().numpy()
            if self.conditional:
                # Select version based on probability
                selected_transcripts = []
                prefixes = []
                prefix_lens = []
                for i in ids:
                    pair = self.transcript_pairs[i]
                    if random.random() < self.pnc_prob:
                        transcript, prefix, prefix_len = pair.get_pnc()
                    else:
                        transcript, prefix, prefix_len = pair.get_nopnc()
                    selected_transcripts.append(transcript)
                    prefixes.append(prefix)
                    prefix_lens.append(prefix_len)

                transcripts = [self.tokenizer.tokenize(x) for x in selected_transcripts]
                # concat prefix and transcript, in token space
                transcripts = [
                    prefix + t for prefix, t in zip(prefixes, transcripts, strict=True)
                ]
            else:
                transcripts = [
                    self.tokenizer.tokenize(self.norm_transcripts[i]) for i in ids
                ]
                prefix_lens = [0] * len(transcripts)
            transcripts = [torch.tensor(x, dtype=torch.int64) for x in transcripts]

            # data['label_lens'] is populated with meaningless values and is not used
            sizes = torch.tensor([t.size(0) for t in transcripts], dtype=torch.int32)

            raw_transcripts = [self.raw_transcripts_cache[i] for i in ids]

        # Tensors are padded with 0. In `sentencepiece` it is set to <unk>,
        # because it cannot be disabled, and is absent in the data.
        # Note this is different from the RNN-T blank token (index 1023).
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True)

        # move to gpu only when requested
        if self.device_type == "gpu":
            transcripts = transcripts.cuda()
            sizes = sizes.cuda()

        return transcripts, sizes, raw_transcripts, torch.tensor(prefix_lens)


def unicode_to_str(seq: list[int]) -> str:
    return "".join(chr(i) for i in seq if i != -1)
