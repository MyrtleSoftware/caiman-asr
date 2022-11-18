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

# val.sh is just train.sh with most of the training related options removed.
# rob@myrtle, May 2022

export OMP_NUM_THREADS=1

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/baseline_v3-1023sp.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-"/results/RNN-T_best_checkpoint.pt"}}
: ${VAL_MANIFESTS:="$DATA_DIR/librispeech-dev-clean-wav.json"}
: ${VAL_BATCH_SIZE:=2}
: ${SEED=1}
: ${DALI_DEVICE:="cpu"}
: ${MAX_SYMBOL_PER_SAMPLE=300}
: ${AMP:=false}
: ${CUDNN_BENCHMARK:=true}
: ${NUM_GPUS:=8}

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --ckpt=$CHECKPOINT"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --val_batch_size=$VAL_BATCH_SIZE"
ARGS+=" --seed=$SEED"
ARGS+=" --dali_device=$DALI_DEVICE"
ARGS+=" --max_symbol_per_sample=$MAX_SYMBOL_PER_SAMPLE"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"

DISTRIBUTED=${DISTRIBUTED:-"-m torch.distributed.launch --nproc_per_node=$NUM_GPUS"}
python ${DISTRIBUTED} val.py ${ARGS}
