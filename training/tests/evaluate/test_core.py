#!/usr/bin/env python3
from argparse import Namespace
from dataclasses import dataclass
from datetime import timedelta

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from caiman_asr_train.evaluate.core import evaluate
from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.lhotse.config_build import build_from_legacy_sources
from caiman_asr_train.lhotse.pipeline import LhotsePipeline
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.joint_loss import JointLoss
from caiman_asr_train.rnnt.response import (
    DecodingResponse,
    FrameResponses,
    HypothesisResponse,
)


@dataclass
class FakeConfig:
    enc_stack_time_factor = 2


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()

    def eval(self):
        pass

    def train(self):
        pass


class FakeJointLoss(JointLoss):
    """``calculate_loss`` is False here, so this must never be called."""

    def forward(self, f, g, t, loss_mods):
        raise AssertionError("joint_loss must not run when calculate_loss is False")


class FakeDecoder:
    def __init__(self, tokens):
        self.tokens = tokens

    def decode(self, feats):
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


def maybe_suppress():
    if hasattr(HealthCheck, "function_scoped_fixture"):
        return dict(suppress_health_check=[HealthCheck.function_scoped_fixture])
    else:
        return dict()


@settings(deadline=timedelta(seconds=1), **maybe_suppress())
@given(st.lists(st.integers(min_value=0, max_value=TOKENIZER_NUM_LABELS - 1)))
def test_unk_token(tokenizer, dataload_args, tokens):
    """The decoder should not crash even if an untrained model predicts '⁇'"""
    assert tokenizer.num_labels == TOKENIZER_NUM_LABELS
    args = Namespace(
        val_dataset_dir=dataload_args.val_dataset_dir,
        punctuation_marks=[",", ".", "?"],
        punctuation_normalization=True,
        output_dir="/tmp",
        timestamp="19700101",
        loader_train_device="gpu",
        loader_val_device="gpu",
        sr_segment=0,
        calculate_emission_latency=False,
        calculate_per=False,
        model_config=dataload_args.model_config,
        eos_vad_threshold=float("inf"),
        eos_is_terminal=False,
        latency_head_offset=0.0,
        latency_tail_offset=0.0,
        turn_off_initial_padding=True,
        decoder="greedy",
        keywords_path=None,
        boost_keywords=False,
        keyword_boost_c0=0.3,
        keyword_boost_beta=0.9,
        keyword_boost_context_score=1.0,
        val_standardizer_lang="en",
    )

    cfg = config.load(dataload_args.model_config)

    conf, *_ = build_from_legacy_sources(
        dataload_args,
        cfg,
        "val",
        use_cuda=True,
    )

    evaluate(
        epoch=0,
        step=0,
        val_loader=LhotsePipeline(conf, tokenizer=tokenizer),
        detokenize=tokenizer.detokenize,
        ema_model=FakeModel(),
        joint_loss=FakeJointLoss(),
        decoder=FakeDecoder(tokens),
        cfg=cfg,
        args=args,
        skip_logging=True,
        error_rate=ErrorRate.WORD,
    )
