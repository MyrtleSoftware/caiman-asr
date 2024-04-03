#!/usr/bin/env bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -Eeuo pipefail

: ${DATASET_NAME_LOWER_CASE:="librispeech"}
: ${MAX_DURATION_SECS:=16.7}
: ${SPM_SIZE:=8703}
: ${CONFIG_NAME:=base-8703sp}
: ${DATA_DIR:="/datasets/LibriSpeech"}

. scripts/create_config_set_env.sh

python ./caiman_asr_train/data/datasets/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/train-clean-100 \
    --dest_dir /datasets/LibriSpeech/train-clean-100-wav \
    --output_json /datasets/LibriSpeech/librispeech-train-clean-100-wav.json
python ./caiman_asr_train/data/datasets/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/train-clean-360 \
    --dest_dir /datasets/LibriSpeech/train-clean-360-wav \
    --output_json /datasets/LibriSpeech/librispeech-train-clean-360-wav.json
python ./caiman_asr_train/data/datasets/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/train-other-500 \
    --dest_dir /datasets/LibriSpeech/train-other-500-wav \
    --output_json /datasets/LibriSpeech/librispeech-train-other-500-wav.json


python ./caiman_asr_train/data/datasets/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/dev-clean \
    --dest_dir /datasets/LibriSpeech/dev-clean-wav \
    --output_json /datasets/LibriSpeech/librispeech-dev-clean-wav.json
python ./caiman_asr_train/data/datasets/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/dev-other \
    --dest_dir /datasets/LibriSpeech/dev-other-wav \
    --output_json /datasets/LibriSpeech/librispeech-dev-other-wav.json


python ./caiman_asr_train/data/datasets/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/test-clean \
    --dest_dir /datasets/LibriSpeech/test-clean-wav \
    --output_json /datasets/LibriSpeech/librispeech-test-clean-wav.json
python ./caiman_asr_train/data/datasets/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/test-other \
    --dest_dir /datasets/LibriSpeech/test-other-wav \
    --output_json /datasets/LibriSpeech/librispeech-test-other-wav.json

TRAIN_MANIFESTS="librispeech-train-clean-100-wav.json librispeech-train-clean-360-wav.json librispeech-train-other-500-wav.json"

DATA_DIR=$DATA_DIR TRAIN_MANIFESTS=$TRAIN_MANIFESTS ./scripts/spm_from_json.sh

python caiman_asr_train/utils/generate_mel_stats.py \
    --output_dir $MEL_STATS_DIR \
    --model_config $RUN_CONFIG \
    --dataset_dir $DATA_DIR \
    --train_manifests $TRAIN_MANIFESTS \
    --n_utterances_only 500000
