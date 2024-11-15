#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

python caiman_asr_train/data/generate_mel_stats.py \
	--output_dir /tmp \
	--model_config configs/testing-1023sp_run.yaml \
	--dataset_dir tests/test_data \
	--train_manifests peoples-speech-short.json \
	--dali_train_device cpu \
	--dump_mel_stats_batch_size 2

TEST_EXIT_CODE=$?

# cleanup any artefacts created during testing
# This is required because, on the host runner, the files (e.g. .hypothesis) are
# created inside docker (i.e. with root permissions) so the host runner can't
# delete them
./scripts/test_cleanup.sh

exit $TEST_EXIT_CODE
