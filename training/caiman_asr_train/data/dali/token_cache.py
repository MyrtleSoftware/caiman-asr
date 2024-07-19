#!/usr/bin/env python3
import numpy as np
import torch
from beartype.typing import Tuple

from caiman_asr_train.data.decide_on_loader import DataSource
from caiman_asr_train.data.text.preprocess import norm_and_tokenize_parallel
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig


class TokenCache:
    """When using json manifests, this class tokenizes the
    transcripts at the start of training for speed. DALI outputs indices,
    and this class looks them up in the cache to get transcripts.

    When using Webdataset files/Hugging Face, DALI outputs transcripts.
    This class decodes them."""

    def __init__(
        self,
        normalize_config: NormalizeConfig,
        tokenizer: Tokenizer,
        device_type: str,
        data_source: DataSource,
    ):
        self.normalize_config = normalize_config
        self.tokenizer = tokenizer
        self.device_type = device_type
        self.data_source = data_source

    def pretokenize(self, transcripts: dict) -> None:
        """
        Tokenize transcripts, which are in form of `{0: 'trans_1', ...}` into
        array of `torch.Tensor` where each item in a tensor is `int` encoding
        a token_id. Transcripts and their lengths are stored as attributes of this
        class.
        """
        if self.data_source is not DataSource.JSON:
            return
        # tokenize all transcripts once at start of training
        self.raw_transcripts_cache = transcripts
        transcripts = [transcripts[i] for i in range(len(transcripts))]
        transcripts = norm_and_tokenize_parallel(
            transcripts, self.normalize_config, self.tokenizer
        )
        transcripts = [torch.tensor(t) for t in transcripts]
        self.tr = np.array(transcripts, dtype=object)
        self.t_sizes = torch.tensor([t.size(0) for t in transcripts], dtype=torch.int32)

    def get_transcripts(
        self, data: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Generate transcripts in format expected by training and evaluation loops
        """
        labels = data["label"]

        if self.data_source is not DataSource.JSON:
            # labels is the actual transcript
            transcripts = labels
            sizes = data["label_lens"].squeeze(1)
            raw_transcripts = [unicode_to_str(seq) for seq in data["raw_transcript"]]
        else:
            # labels refers to an id that can be used to retrieve the transcript
            ids = labels.flatten().numpy()
            transcripts = self.tr[ids]
            # data['label_lens'] is populated with meaningless values and is not used
            sizes = self.t_sizes[ids]
            raw_transcripts = [self.raw_transcripts_cache[i] for i in ids]

        # Tensors are padded with 0. In `sentencepiece` it is set to <unk>,
        # because it cannot be disabled, and is absent in the data.
        # Note this is different from the RNN-T blank token (index 1023).
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True)

        # move to gpu only when requested
        if self.device_type == "gpu":
            transcripts = transcripts.cuda()
            sizes = sizes.cuda()

        return transcripts, sizes, raw_transcripts


def unicode_to_str(seq: list[int]) -> str:
    return "".join(chr(i) for i in seq if i != -1)
