#!/bin/bash
set -Eeuo pipefail

if [ -n "$(ls -A /results)" ]; then
	echo "Stopping script early"
	echo "/results is not empty and this script would delete the contents of /results"
	exit 1
fi

cleanup() {
	# cleanup any artefacts created during testing
	# This is required because, on the host runner, the files (e.g. .hypothesis) are
	# created inside docker (i.e. with root permissions) so the host runner can't
	# delete them
	./scripts/test_cleanup.sh
	# Otherwise each test run will use up 700MB more disk space
	rm -rf /results/*
}

trap cleanup EXIT

# Make a config file that uses the testing model,
# but with the saved mini sentencepiece model
cat configs/testing-1023sp.yaml |
	sed "s|\[TRANSCRIPTS\]||" |
	sed "s|ON_MISSING|raise_error|" |
	sed "s|/datasets/sentencepieces/SENTENCEPIECE.model|tests/test_data/librispeech62.eos.pnc-tag.model|" |
	sed "s|/datasets/stats/STATS_SUBDIR|tests/test_data/|" |
	sed "s|FT_LANGS||" |
	sed "s|LANGS|en|" |
	sed "s|/datasets/words/WORDS|tests/test_data/words|" |
	sed "s/MAX_DURATION/16.7/" |
	sed "s/MIN_TPS/0.0/" |
	sed "s/MAX_TPS/inf/" |
	sed "s/MAX_TOKENS/500/" \
		>/tmp/testing.yaml

# Hence we skip the state dict check because this isn't one of
# the 4 supported architectures
./scripts/train.sh \
	--model_config=/tmp/testing.yaml \
	--skip_state_dict_check \
	--train_dataset_yaml=tests/test_data/peoples-speech-short.yaml \
	--val_manifests=peoples-speech-short.cuts.jsonl \
	--val_data_dir=tests/test_data \
	--num_gpus=1 \
	--batch_duration=30 \
	--grad_accumulation_batches=1 \
	--joint_lattice_cap=256 \
	--training_steps=4 \
	--prediction_frequency 1 \
	--val_batch_size=2 \
	--val_standardizer_lang en \
	--conditional \
	--pnc_prefix pnc \
	--eos_decoding predict \
	--dp_mu 0.0 \
	--skip_val_loss

./scripts/val.sh \
	--model_config=/tmp/testing.yaml \
	--val_manifests=peoples-speech-short.eos.cuts.jsonl \
	--val_data_dir=tests/test_data \
	--num_gpus=1 \
	--val_batch_size=2 \
	--val_standardizer_lang en \
	--eos_decoding predict \
	--calculate_emission_latency \
	--pnc_prefix 'pnc'

# Check that CPU-only validation works

export CUDA_VISIBLE_DEVICES=""

./scripts/val.sh \
	--model_config=/tmp/testing.yaml \
	--val_manifests=peoples-speech-short.eos.cuts.jsonl \
	--val_data_dir=tests/test_data \
	--num_gpus=1 \
	--val_batch_size=2 \
	--val_standardizer_lang en \
	--eos_decoding predict \
	--calculate_emission_latency \
	--cpu \
	--loader_val_device cpu \
	--pnc_prefix 'pnc'
unset CUDA_VISIBLE_DEVICES
