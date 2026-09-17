#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_NAME_LOWER_CASE=$1
MAX_DURATION_SECS=$2
SPM_SIZE=$3
CONFIG_NAME=$4
NGRAM_ORDER=$5
TRAIN_MANIFEST_YAML=$6
NUM_GPUS=$7

. scripts/create_config_set_env.sh

echo "Make SentencePiece model"

# If EXTERN_CONFIG defined use it, otherwise use the RUN_CONFIG
if [[ -n ${EXTERN_CONFIG:-} ]]; then
	RUN_CONFIG="$EXTERN_CONFIG"
fi

# ANCHOR: spm_in_mdbook
python -X utf8 caiman_asr_train/data/spm/spm_from_json.py \
	--spm_size "$SPM_SIZE" \
	--spm_name "$SPM_NAME" \
	--train_dataset_yaml $TRAIN_MANIFEST_YAML \
	--output_dir /datasets/sentencepieces \
	--model_config "$RUN_CONFIG" \
	--input_sentence_size 5000000 \
	--num_gpus "$NUM_GPUS"
# ANCHOR_END: spm_in_mdbook

echo "Generate mel stats"

python -X utf8 caiman_asr_train/data/generate_mel_stats.py \
	--output_dir "$MEL_STATS_DIR" \
	--model_config "$RUN_CONFIG" \
	--train_dataset_yaml $TRAIN_MANIFEST_YAML \
	--num_gpus "$NUM_GPUS" \
	--mel_duration_sample 100_000_000 \
	--val_manifests /tmp \
	--val_dataset_dir /tmp \
	--val_standardizer_lang en \
	--mel_batch_duration ${MEL_BATCH_DURATION:-3600.0}

echo "Generate n-gram data"

python -X utf8 caiman_asr_train/lm/prep_kenlm_data.py \
	--train_dataset_yaml $TRAIN_MANIFEST_YAML \
	--output_path "$NGRAM_DIR/transcripts.txt" \
	--model_config "$RUN_CONFIG" \
	--num_gpus "$NUM_GPUS"

./scripts/generate_ngram.sh "$NGRAM_ORDER" "$PATH_TO_TXT" "$PATH_TO_ARPA" "$PATH_TO_BINARY"
