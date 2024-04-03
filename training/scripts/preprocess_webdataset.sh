#!/usr/bin/env bash
set -Eeuo pipefail

: ${DATASET_NAME_LOWER_CASE:="librispeech"}
: ${MAX_DURATION_SECS:=16.7}
: ${SPM_SIZE:=8703}
: ${CONFIG_NAME:=base-8703sp}
: ${TRAIN_TAR_FILES:=}
: ${DATA_DIR:=}

. scripts/create_config_set_env.sh

# build spm and move to /datasets/sentencepieces
python caiman_asr_train/data/webdataset/webdataset_spm.py \
    --dataset_dir $DATA_DIR \
    --train_tar_files $TRAIN_TAR_FILES \
    --spm_size $SPM_SIZE \
    --spm_name $SPM_NAME \
    --model_config $RUN_CONFIG

mkdir -p /datasets/sentencepieces
mv $SPM_NAME.* /datasets/sentencepieces/

python caiman_asr_train/utils/generate_mel_stats.py \
    --output_dir $MEL_STATS_DIR \
    --model_config $RUN_CONFIG \
    --dataset_dir $DATA_DIR \
    --train_tar_files $TRAIN_TAR_FILES \
    --val_tar_files "<not used but must be set>" \
    --read_from_tar
