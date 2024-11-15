import unittest

import pytest
from pydantic import ValidationError

from caiman_asr_train.export.config_schema import (
    FilterbankFeaturesSchema,
    FrameSplicingSchema,
    InputValSchema,
    ModelSchema,
    RNNTInferenceConfigSchema,
    TokenizerSchema,
)


class TestConfigSchema(unittest.TestCase):
    def test_load(self):
        cfg_data = {
            "grad_noise_scheduler": {
                "decay_const": 0.55,
                "noise_level": 0.0,
                "start_step": 2000,
            },
            "input_train": {
                "audio_dataset": {
                    "max_duration": 20.0,
                    "max_transcript_len": 450,
                    "normalize_transcripts": False,
                    "sample_rate": 16000,
                    "speed_perturbation": {
                        "max_rate": 1.15,
                        "min_rate": 0.85,
                        "p": 1.0,
                    },
                    "trim_silence": True,
                },
                "filterbank_features": {
                    "dither": 1e-05,
                    "n_fft": 512,
                    "n_filt": 80,
                    "normalize": "per_feature",
                    "sample_rate": 16000,
                    "window": "hann",
                    "window_size": 0.025,
                    "window_stride": 0.01,
                },
                "frame_splicing": {"frame_stacking": 3, "frame_subsampling": 3},
                "spec_augment": {
                    "freq_masks": 2,
                    "max_freq": 20,
                    "max_time": 0.03,
                    "min_freq": 0,
                    "min_time": 0,
                    "time_masks": 10,
                },
            },
            "input_val": {
                "audio_dataset": {
                    "normalize_transcripts": False,
                    "sample_rate": 16000,
                    "trim_silence": False,
                },
                "filterbank_features": {
                    "dither": 1e-05,
                    "n_fft": 512,
                    "n_filt": 80,
                    "normalize": "per_feature",
                    "sample_rate": 16000,
                    "window": "hann",
                    "window_size": 0.025,
                    "window_stride": 0.01,
                    "stats_path": "/datasets/stats/librispeech-winsz0.025",
                },
                "frame_splicing": {"frame_stacking": 3, "frame_subsampling": 3},
            },
            "rnnt": {
                "custom_lstm": True,
                "enc_batch_norm": False,
                "enc_dropout": 0.1,
                "enc_freeze": False,
                "enc_n_hid": 1536,
                "enc_post_rnn_layers": 6,
                "enc_pre_rnn_layers": 2,
                "enc_rw_dropout": 0.0,
                "enc_stack_time_factor": 2,
                "forget_gate_bias": 1.0,
                "in_feats": 240,
                "joint_apex_relu_dropout": True,
                "joint_apex_transducer": "pack",
                "joint_dropout": 0.3,
                "joint_n_hid": 1024,
                "joint_net_lr_factor": 0.243,
                "pred_batch_norm": False,
                "pred_dropout": 0.3,
                "pred_n_hid": 768,
                "pred_rnn_layers": 2,
                "pred_rw_dropout": 0.0,
                "quantize": False,
            },
            "tokenizer": {
                "labels": ["a", "b"],
                "sentpiece_model": "/datasets/sentencepieces/50k-ls+cv+mls+ps17407.model",
            },
            "ngram": {
                "ngram_path": "/datasets/ngrams/NGRAM_SUBDIR",
                "scale_factor": 0.05,
            },
        }
        inference_config = RNNTInferenceConfigSchema(**cfg_data)

        self.assertIsInstance(inference_config, RNNTInferenceConfigSchema)
        self.assertIsInstance(inference_config.input_val, InputValSchema)
        self.assertIsInstance(inference_config.rnnt, ModelSchema)
        self.assertIsInstance(inference_config.tokenizer, TokenizerSchema)
        self.assertIsInstance(
            inference_config.input_val.filterbank_features, FilterbankFeaturesSchema
        )
        self.assertIsInstance(
            inference_config.input_val.frame_splicing, FrameSplicingSchema
        )


def test_raises_error_with_new_field():
    with pytest.raises(
        ValidationError,
        match="1 validation error for TokenizerSchema"
        "\nnew_field\n  Extra inputs are not permitted",
    ):
        TokenizerSchema(
            **{
                "labels": ["a", "b"],
                "sentpiece_model": "/datasets/sentencepieces/50k-ls+cv+mls+ps17407.model",
                "new_field": "new_value",
            }
        )


def test_raises_error_with_missing_field():
    with pytest.raises(
        ValidationError,
        match="1 validation error for TokenizerSchema"
        "\nsentpiece_model\n  Field required",
    ):
        TokenizerSchema(
            **{"labels": [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i"]}
        )
