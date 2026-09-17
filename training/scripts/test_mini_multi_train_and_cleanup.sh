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
	sed "s|ON_MISSING|use_default|" |
	sed "s|/datasets/sentencepieces/SENTENCEPIECE.model|tests/test_data/multilingual_artifacts/multilingual.model|" |
	sed "s|/datasets/stats/STATS_SUBDIR|tests/test_data/multilingual_artifacts|" |
	sed "s|FT_LANGS||" |
	sed "s|LANGS|en,fr|" |
	sed "s|/datasets/words/WORDS|tests/test_data/words|" |
	sed "s/MAX_DURATION/16.7/" |
	sed "s/MAX_TOKENS/500/" |
	sed "s/MIN_TPS/0.0/" |
	sed "s/MAX_TPS/inf/" |
	sed 's|lang_detect: null|lang_detect: "<lang_detect>"|' \
		>/tmp/testing.yaml

# Hence we skip the state dict check because this isn't one of
# the 4 supported architectures
./scripts/train.sh \
	--model_config=/tmp/testing.yaml \
	--skip_state_dict_check \
	--train_dataset_yaml=tests/test_data/multilingual.yaml \
	--val_manifests=tests/test_data/peoples-speech-short.cuts.jsonl \
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
	--conditional_language \
	--pnc_prefix pnc \
	--val_lang_prefix lang_en \
	--eos_decoding predict \
	--lang_mask_prob 0.30 \
	--dp_mu 0.0 \
	--skip_val_loss

./scripts/val.sh \
	--model_config=/tmp/testing.yaml \
	--val_manifests=common_voice-short.fr.eos.cuts.jsonl \
	--val_data_dir=tests/test_data \
	--num_gpus=1 \
	--val_batch_size=2 \
	--val_standardizer_lang fr \
	--val_lang_prefix lang_fr \
	--eos_decoding predict \
	--pnc_prefix 'pnc'

# Check that CPU-only validation works

export CUDA_VISIBLE_DEVICES=""

./scripts/val.sh \
	--model_config=/tmp/testing.yaml \
	--val_manifests=common_voice-short.fr.eos.cuts.jsonl \
	--val_data_dir=tests/test_data \
	--num_gpus=1 \
	--val_batch_size=2 \
	--val_standardizer_lang fr \
	--val_lang_prefix lang_fr \
	--eos_decoding predict \
	--cpu \
	--loader_val_device cpu \
	--pnc_prefix 'pnc'
unset CUDA_VISIBLE_DEVICES
