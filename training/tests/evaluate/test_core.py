#!/usr/bin/env python3
from argparse import Namespace
from datetime import timedelta

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from caiman_asr_train.evaluate.core import evaluate
from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.rnnt.response import (
    DecodingResponse,
    FrameResponses,
    HypothesisResponse,
)


class FakeModel:
    def __init__(self):
        self.enc_stack_time_factor = 2

    def eval(self):
        pass

    def train(self):
        pass


class FakeLoader:
    def __init__(self):
        self.pipeline_type = "val"
        self.data = [
            "audio",
            torch.tensor([0]),
            "txt",
            "txt_len",
            ["raw_transcripts"],
            ["fnames"],
        ]

    def __iter__(self):
        return self

    def __next__(self):
        if data := self.data:
            self.data = None
            print(f"FakeLoader: {data}")
            return data
        raise StopIteration

    def __len__(self):
        return 1


class FakeDecoder:
    def __init__(self, tokens):
        self.tokens = tokens

    def decode(self, feats, feat_lens):
        return [{i: __class__._response(i, tok) for i, tok in enumerate(self.tokens)}]

    @property
    def eos_index(self):
        return None

    @staticmethod
    def _response(step: int, tok: int) -> FrameResponses:
        return FrameResponses(
            partials=None,
            final=DecodingResponse(
                start_frame_idx=step,
                duration_frames=1,
                is_provisional=False,
                alternatives=[
                    HypothesisResponse(
                        y_seq=[tok],
                        timesteps=[step],
                        token_seq=["<unk>"],
                        confidence=[0.1],
                    )
                ],
            ),
        )


TOKENIZER_NUM_LABELS = 29


@settings(deadline=timedelta(seconds=1))
@given(st.lists(st.integers(min_value=0, max_value=TOKENIZER_NUM_LABELS - 1)))
def test_unk_token(tokenizer, tokens):
    """The decoder should not crash even if an untrained model predicts '⁇'"""
    assert tokenizer.num_labels == TOKENIZER_NUM_LABELS
    args = Namespace(
        breakdown_wer=False,
        breakdown_chars="",
        output_dir="/tmp",
        timestamp="19700101",
        dali_train_device="cuda",
        dali_val_device="cuda",
        sr_segment=0,
        calculate_emission_latency=False,
        model_config="/workspace/training/configs/testing-1023sp.yaml",
        eos_vad_threshold=float("inf"),
        eos_is_terminal=False,
    )

    evaluate(
        epoch=0,
        step=0,
        val_loader=FakeLoader(),
        val_feat_proc=lambda x: x,
        detokenize=tokenizer.detokenize,
        ema_model=FakeModel(),
        loss_fn=None,
        decoder=FakeDecoder(tokens),
        cfg=None,
        args=args,
        skip_logging=True,
        error_rate=ErrorRate.WORD,
    )
