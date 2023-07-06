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

# valCPU.sh is derived from val.sh
# rob@myrtle, May 2022

export OMP_NUM_THREADS=1

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/testing-1023sp_run.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-"/results/RNN-T_best_checkpoint.pt"}}
: ${VAL_MANIFESTS:="$DATA_DIR/librispeech-dev-clean-wav.json"}
: ${VAL_BATCH_SIZE:=1}
: ${SEED=1}
: ${MAX_SYMBOL_PER_SAMPLE=300}
: ${ALPHA:=0.001}
: ${AMP:=false}
: ${CUDNN_BENCHMARK:=true}
: ${STREAM_NORM:=false}
: ${RESET_STREAM_STATS:=false}
: ${DUMP_NTH:=None}
: ${DUMP_PREDS:=false}

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --ckpt=$CHECKPOINT"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --val_batch_size=$VAL_BATCH_SIZE"
ARGS+=" --seed=$SEED"
ARGS+=" --max_symbol_per_sample=$MAX_SYMBOL_PER_SAMPLE"
ARGS+=" --alpha=$ALPHA"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ "$STREAM_NORM" = true ] &&         ARGS+=" --streaming_normalization"
[ "$RESET_STREAM_STATS" = true ] &&  ARGS+=" --reset_stream_stats"
[ "$DUMP_NTH" != None ] &&           ARGS+=" --dump_nth=$DUMP_NTH"
[ "$DUMP_PREDS" = true ] &&          ARGS+=" --dump_preds"

python valCPU.py ${ARGS}
