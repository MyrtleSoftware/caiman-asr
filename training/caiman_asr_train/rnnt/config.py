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

import copy
import inspect
from pathlib import Path

import yaml
from beartype import beartype
from beartype.typing import Dict
from pydantic import BaseModel, ConfigDict, confloat

from caiman_asr_train.data import features
from caiman_asr_train.data.dali.pipeline import PipelineParams, SpeedPerturbationParams
from caiman_asr_train.data.unk_handling import UnkHandling
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.train_utils.grad_noise_scheduler import GradNoiseScheduler


def default_args(klass) -> Dict:
    sig = inspect.signature(klass.__init__)
    return {k: v.default for k, v in sig.parameters.items() if k != "self"}


def load(fpath, max_duration=None) -> Dict:
    if fpath.endswith(".toml"):
        raise ValueError(".toml config format has been changed to .yaml")

    cfg = yaml.safe_load(open(fpath, "r"))

    # Reload to deep copy shallow copies, which were made with yaml anchors
    yaml.Dumper.ignore_aliases = lambda *args: True
    cfg = yaml.safe_load(yaml.dump(cfg))

    # Modify the config with supported cmdline flags
    if max_duration is not None:
        cfg["input_train"]["audio_dataset"]["max_duration"] = max_duration
        cfg["input_train"]["filterbank_features"]["max_duration"] = max_duration

    return cfg


def validate_and_fill(klass, user_conf, ignore=[], optional=[], deprecated=[]):
    conf = default_args(klass)

    to_ignore = set(ignore).union(deprecated)
    for k, v in user_conf.items():
        assert k in conf or k in to_ignore, f"Unknown parameter {k} for {klass}"
        if k in deprecated:
            continue
        conf[k] = v

    # Keep only mandatory or optional-nonempty
    conf = {
        k: v
        for k, v in conf.items()
        if k not in optional or v is not inspect.Parameter.empty
    }

    # Validate
    for k, v in conf.items():
        assert (
            v is not inspect.Parameter.empty
        ), f"Value for {k} not specified for {klass}"
    return conf


def input(conf_yaml, split="train"):
    conf = copy.deepcopy(conf_yaml[f"input_{split}"])
    conf_dataset = conf.pop("audio_dataset")
    conf_features = conf.pop("filterbank_features")
    conf_splicing = conf.pop("frame_splicing", {})
    conf_specaugm = conf.pop("spec_augment", None)
    _ = conf.pop("cutout_augment", None)

    # Validate known inner classes
    inner_classes = [
        (conf_dataset, "speed_perturbation", SpeedPerturbationParams),
    ]
    for conf_tgt, key, klass in inner_classes:
        if key in conf_tgt:
            conf_tgt[key] = validate_and_fill(klass, conf_tgt[key])

    for k in conf:
        raise ValueError(f"Unknown key {k}")

    # Validate outer classes
    conf_dataset = validate_and_fill(
        PipelineParams,
        conf_dataset,
        optional=["standardize_wer", "replacements", "remove_tags", "user_symbols"],
    )

    conf_splicing = validate_and_fill(features.FrameSplicing, conf_splicing)
    conf_specaugm = conf_specaugm and validate_and_fill(
        features.SpecAugment, conf_specaugm
    )

    # Check params shared between classes
    for shared in ["sample_rate"]:
        assert conf_dataset[shared] == conf_features[shared], (
            f"{shared} should match in Dataset and FeatureProcessor: "
            f"{conf_dataset[shared]}, {conf_features[shared]}"
        )

    return conf_dataset, conf_features, conf_splicing, conf_specaugm


def grad_noise_scheduler(conf) -> Dict:
    """ """
    return validate_and_fill(GradNoiseScheduler, conf["grad_noise_scheduler"])


def rnnt(conf) -> Dict:
    return validate_and_fill(
        RNNT,
        conf["rnnt"],
        optional=["n_classes"],
        deprecated=["hard_activation_functions"],
    )


class TokenizerConfig(BaseModel):
    sentpiece_model: str
    labels: list[str]
    # If using an old config, set sampling to 0.
    sampling: confloat(ge=0.0, le=1.0) = 0.0

    model_config = ConfigDict(extra="forbid")


class SentencePieceConfig(BaseModel):
    unk_handling: UnkHandling = UnkHandling.FAIL

    model_config = ConfigDict(extra="forbid")


@beartype
def get_spm_conf(tokenizer_conf: TokenizerConfig) -> SentencePieceConfig:
    sentpiece_model_path = Path(tokenizer_conf.sentpiece_model)
    spm_config_path = sentpiece_model_path.with_suffix(".yaml")
    raw_spm_config = (
        yaml.safe_load(spm_config_path.read_text()) if spm_config_path.exists() else {}
    )
    return SentencePieceConfig(**raw_spm_config)


@beartype
def get_tokenizer_conf(conf: dict) -> dict:
    """The Tokenizer class needs information from both the
    main YAML config and the SentencePiece YAML config"""
    tokenizer_conf = TokenizerConfig(**conf["tokenizer"])
    spm_conf = get_spm_conf(tokenizer_conf)
    return dict(tokenizer_conf) | dict(spm_conf)
