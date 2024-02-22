#!/bin/bash

if [ -n "$(ls -A /results)" ]
then
   echo "Stopping script early"
   echo "/results is not empty and this script would delete the contents of /results"
   exit 1
fi

# Make a config file that uses the testing model,
# but with the saved mini sentencepiece model
cat configs/testing-1023sp.yaml \
    | sed "s|/datasets/sentencepieces/SENTENCEPIECE.model|tests/test_data/librispeech29.model|" \
    | sed "s/MAX_DURATION/16.7/" > /tmp/testing.yaml

# Hence we skip the state dict check because this isn't one of
# the 4 supported architectures
./scripts/train.sh --model_config=/tmp/testing.yaml --skip_state_dict_check \
    --data_dir=tests/test_data --train_manifests=peoples-speech-short.json \
    --val_manifests=peoples-speech-short.json --num_gpus=1 \
    --global_batch_size=2 --grad_accumulation_batches=1 --batch_split_factor=2 --epochs=1 \
    --prediction_frequency 1

TEST_EXIT_CODE=$?

# Otherwise each test run will use up 700MB more disk space
rm -rf /results/*

# cleanup any artefacts created during testing
# This is required because, on the host runner, the files (e.g. .hypothesis) are
# created inside docker (i.e. with root permissions) so the host runner can't
# delete them
./scripts/test_cleanup.sh

exit $TEST_EXIT_CODE
