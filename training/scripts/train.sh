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

# modified by rob@myrtle

export OMP_NUM_THREADS=1

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/testing-1023sp_run.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-}}
: ${CUDNN_BENCHMARK:=true}
: ${NUM_GPUS:=8}
: ${AMP:=true}
: ${GLOBAL_BATCH_SIZE:=1024}
: ${VAL_BATCH_SIZE:=2}
: ${GRAD_ACCUMULATION_BATCHES:=8}
: ${LEARNING_RATE:=0.004}
: ${MIN_LEARNING_RATE:=4e-4}
: ${NUM_BUCKETS=6} # empty means to use torch.utils.data.distributed.DistributedSampler
: ${EMA:=0.999}
: ${SEED=1}
: ${EPOCHS:=100}
: ${WARMUP_STEPS:=1632}
: ${HOLD_STEPS:=10880}
: ${HALF_LIFE_STEPS:=2805}
: ${SAVE_AT_THE_END:=true}
: ${DUMP_MELMAT:="none"}
: ${DUMP_MEL_STATS:=false}
: ${EPOCHS_THIS_JOB:=0}
: ${RESUME:=false}
: ${FINE_TUNE:=false}
: ${DALI_DEVICE:="cpu"}
: ${VAL_FREQUENCY:=1}
: ${PREDICTION_FREQUENCY:=1000}
: ${BETA1:=0.9}
: ${BETA2:=0.999}
: ${LOG_FREQUENCY:=1}
: ${TRAIN_MANIFESTS:="$DATA_DIR/librispeech-train-clean-100-wav.json \
                      $DATA_DIR/librispeech-train-clean-360-wav.json \
                      $DATA_DIR/librispeech-train-other-500-wav.json"}
: ${VAL_MANIFESTS:="$DATA_DIR/librispeech-dev-clean-wav.json"}
: ${READ_FROM_TAR:=false}
: ${MAX_SYMBOL_PER_SAMPLE=300}
: ${WEIGHTS_INIT_SCALE=0.5}
: ${CLIP_NORM:=1}

TIMESTAMP=$(date '+%Y_%m_%d_%H_%M_%S')

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --train_manifests $TRAIN_MANIFESTS"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --lr=$LEARNING_RATE"
ARGS+=" --min_lr=$MIN_LEARNING_RATE"
ARGS+=" --global_batch_size=$GLOBAL_BATCH_SIZE"
ARGS+=" --num_gpus=$NUM_GPUS"
ARGS+=" --val_batch_size=$VAL_BATCH_SIZE"
ARGS+=" --epochs=$EPOCHS"
ARGS+=" --warmup_steps=$WARMUP_STEPS"
ARGS+=" --hold_steps=$HOLD_STEPS"
ARGS+=" --half_life_steps=$HALF_LIFE_STEPS"
ARGS+=" --epochs_this_job=$EPOCHS_THIS_JOB"
ARGS+=" --ema=$EMA"
ARGS+=" --seed=$SEED"
ARGS+=" --weight_decay=1e-3"
ARGS+=" --log_frequency=$LOG_FREQUENCY"
ARGS+=" --val_frequency=$VAL_FREQUENCY"
ARGS+=" --grad_accumulation_batches=$GRAD_ACCUMULATION_BATCHES"
ARGS+=" --dali_device=$DALI_DEVICE"
ARGS+=" --beta1=$BETA1"
ARGS+=" --beta2=$BETA2"
ARGS+=" --dump_melmat=$DUMP_MELMAT"
ARGS+=" --timestamp=$TIMESTAMP"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$RESUME" = true ] &&              ARGS+=" --resume"
[ "$FINE_TUNE" = true ] &&           ARGS+=" --fine_tune"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ "$SAVE_AT_THE_END" = true ] &&     ARGS+=" --save_at_the_end"
[ "$DUMP_MEL_STATS" = true ] &&      ARGS+=" --dump_mel_stats"
[ "$READ_FROM_TAR" = true ] &&       ARGS+=" --read_from_tar"
[ -n "$TRAIN_TAR_FILES" ] &&         ARGS+=" --train_tar_files $TRAIN_TAR_FILES"
[ -n "$VAL_TAR_FILES" ] &&           ARGS+=" --val_tar_files $VAL_TAR_FILES"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=$CHECKPOINT"
[ -n "$NUM_BUCKETS" ] &&             ARGS+=" --num_buckets=$NUM_BUCKETS"
[ -n "$CLIP_NORM" ] &&               ARGS+=" --clip_norm=$CLIP_NORM"
[ -n "$PREDICTION_FREQUENCY" ] &&    ARGS+=" --prediction_frequency=$PREDICTION_FREQUENCY"
[ -n "$SAVE_BEST" ] &&               ARGS+=" --save_best_from=$SAVE_BEST"
[ -n "$SAVE_FREQUENCY" ] &&          ARGS+=" --save_frequency=$SAVE_FREQUENCY"
[ -n "$START_CLIP" ] &&              ARGS+=" --start_clip=$START_CLIP"
[ -n "$HIDDEN_HIDDEN_BIAS_SCALED" ] && ARGS+=" --hidden_hidden_bias_scale=$HIDDEN_HIDDEN_BIAS_SCALED"
[ -n "$WEIGHTS_INIT_SCALE" ] &&      ARGS+=" --weights_init_scale=$WEIGHTS_INIT_SCALE"
[ -n "$MAX_SYMBOL_PER_SAMPLE" ] &&   ARGS+=" --max_symbol_per_sample=$MAX_SYMBOL_PER_SAMPLE"

DISTRIBUTED=${DISTRIBUTED:-"torchrun --standalone --nnodes 1 --nproc_per_node=$NUM_GPUS"}
${DISTRIBUTED} ./rnnt_train/train.py ${ARGS} 2>&1 | tee $OUTPUT_DIR/training_log_$TIMESTAMP.txt
