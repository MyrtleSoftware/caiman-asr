#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

echo -e "\e[33mDEPRECATED: this supports the legacy environment variable API.\e[0m"
echo -e "\e[33mplease note that to use CPU you need to parse CPU argument.\e[0m"
echo -e "\e[33mTo get newest features you will need to use ./scripts/val.sh\e[0m"

export OMP_NUM_THREADS=1

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/testing-1023sp_run.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-"/results/RNN-T_best_checkpoint.pt"}}
: ${VAL_MANIFESTS:="$DATA_DIR/librispeech-dev-clean-wav.json"}
: ${VAL_BATCH_SIZE:=1}
: ${SEED=1}
: ${ALPHA:=0.001}
: ${DALI_DEVICE:="cpu"}
: ${CUDNN_BENCHMARK:=true}
: ${DUMP_PREDS:=false}
: ${NUM_GPUS:=1}
: ${READ_FROM_TAR:=false}
: ${PYTHON_COMMAND:="./caiman_asr_train/val.py"}
: ${EXTRA_ARGS:=""}
: ${NO_LOSS:=true}
: ${CPU:=false}

TIMESTAMP=$(date '+%Y_%m_%d_%H_%M_%S')

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --ckpt=$CHECKPOINT"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --val_batch_size=$VAL_BATCH_SIZE"
ARGS+=" --seed=$SEED"
ARGS+=" --dali_device=$DALI_DEVICE"
ARGS+=" --timestamp=$TIMESTAMP"
ARGS+=" --num_gpus=$NUM_GPUS"

[ "$CPU" = true ] &&                 ARGS+=" --cpu"
[ "$CUDNN_BENCHMARK" = false ] &&    ARGS+=" --no_cudnn_benchmark"
[ "$DUMP_PREDS" = true ] &&          ARGS+=" --dump_preds"
[ "$NO_LOSS" = false ] &&            ARGS+=" --calculate_loss"
[ "$READ_FROM_TAR" = true ] &&       ARGS+=" --read_from_tar"
[ "$INSPECT_AUDIO" = true ] &&       ARGS+=" --inspect_audio"
[ -n "$VAL_TAR_FILES" ] &&           ARGS+=" --val_tar_files $VAL_TAR_FILES"
[ -n "$MAX_SYMBOL_PER_SAMPLE" ] &&   ARGS+=" --max_symbol_per_sample=$MAX_SYMBOL_PER_SAMPLE"
[ -n "$PROB_VAL_NARROWBAND" ] &&     ARGS+=" --prob_val_narrowband $PROB_VAL_NARROWBAND"
[ -n "$N_UTTERANCES_ONLY" ] &&       ARGS+=" --n_utterances_only=$N_UTTERANCES_ONLY"


${PYTHON_COMMAND} ${ARGS} ${EXTRA_ARGS}
