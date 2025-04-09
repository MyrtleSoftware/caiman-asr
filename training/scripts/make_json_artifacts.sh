#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_NAME_LOWER_CASE=$1
MAX_DURATION_SECS=$2
SPM_SIZE=$3
CONFIG_NAME=$4
DATA_DIR=$5
NGRAM_ORDER=$6
shift 6
TRAIN_MANIFESTS=$*

. scripts/create_config_set_env.sh

# ANCHOR: spm_in_mdbook
python caiman_asr_train/data/spm/spm_from_json.py --spm_size "$SPM_SIZE" \
	--spm_name "$SPM_NAME" --data_dir "$DATA_DIR" \
	--train_manifests $TRAIN_MANIFESTS \
	--output_dir /datasets/sentencepieces \
	--model_config "$RUN_CONFIG"
# ANCHOR_END: spm_in_mdbook

python caiman_asr_train/data/generate_mel_stats.py \
	--output_dir "$MEL_STATS_DIR" \
	--model_config "$RUN_CONFIG" \
	--dataset_dir "$DATA_DIR" \
	--train_manifests $TRAIN_MANIFESTS \
	--dali_train_device cpu \
	--n_utterances_only 500000 \
	--canary_exponent -1

python caiman_asr_train/lm/prep_kenlm_data.py \
	--data_dir "$DATA_DIR" \
	--manifests $TRAIN_MANIFESTS \
	--output_path "$NGRAM_DIR/transcripts.txt" \
	--model_config "$RUN_CONFIG"

./scripts/generate_ngram.sh "$NGRAM_ORDER" "$PATH_TO_TXT" "$PATH_TO_ARPA" "$PATH_TO_BINARY"
