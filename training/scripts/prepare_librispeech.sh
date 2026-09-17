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
: ${TRAIN_MANIFEST_YAML:="./configs/librispeech.yaml"}
: ${NUM_GPUS:=1}
: ${EXTRA_ARGS:=""}

echo "Preparing LibriSpeech dataset"

python caiman_asr_train/data/make_datasets/librispeech.py --data_dir $(dirname $DATA_DIR) $EXTRA_ARGS

echo "Converting to lhotse-cutsest"

for manifest in "$DATA_DIR"/*.json; do
	python ./caiman_asr_train/lhotse/scripts/convert.py json2cuts \
		"$manifest" \
		"$DATA_DIR" \
		--data-dir "$DATA_DIR"
done

# Transform manifests to include endpointing (EOS)
echo "Segmenting manifests"

python scripts/eos_add.py \
	--manifests "$DATA_DIR" \
	--output_dirs "$DATA_DIR" \
	--overwrite

echo "Creating JSON artifacts"

./scripts/make_json_artifacts.sh \
	"$DATASET_NAME_LOWER_CASE" \
	"$MAX_DURATION_SECS" \
	"$SPM_SIZE" \
	"$CONFIG_NAME" \
	"$NGRAM_ORDER" \
	"$TRAIN_MANIFEST_YAML" \
	"$NUM_GPUS"
