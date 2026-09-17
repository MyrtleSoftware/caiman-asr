#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

cat /workspace/training/configs/testing-1023sp.yaml |
	sed "s|\[TRANSCRIPTS\]||" |
	sed "s|ON_MISSING|raise_error|" |
	sed "s|LANGS|en|" |
	sed "s/MAX_DURATION/20.0/" |
	sed "s/MIN_TPS/0.0/" |
	sed "s/MAX_TPS/inf/" |
	sed "s/MAX_TOKENS/500/" \
		>/tmp/conf.yaml

python caiman_asr_train/data/generate_mel_stats.py \
	--output_dir /tmp \
	--model_config /tmp/conf.yaml \
	--train_dataset_yaml tests/test_data/peoples-speech-short.yaml \
	--loader_train_device cpu \
	--val_manifests /tmp \
	--val_dataset_dir /tmp \
	--val_standardizer_lang en

TEST_EXIT_CODE=$?

# cleanup any artefacts created during testing
# This is required because, on the host runner, the files (e.g. .hypothesis) are
# created inside docker (i.e. with root permissions) so the host runner can't
# delete them
./scripts/test_cleanup.sh

exit $TEST_EXIT_CODE
