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
: ${NGRAM_ORDER:=4}

python caiman_asr_train/data/make_datasets/librispeech.py --data_dir $DATA_DIR

TRAIN_MANIFESTS="librispeech-train-clean-100.json librispeech-train-clean-360.json librispeech-train-other-500.json"

./scripts/make_json_artifacts.sh "$DATASET_NAME_LOWER_CASE" "$MAX_DURATION_SECS" \
    "$SPM_SIZE" "$CONFIG_NAME" "$DATA_DIR" "$NGRAM_ORDER" $TRAIN_MANIFESTS
