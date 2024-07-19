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
    | sed "s|/datasets/stats/STATS_SUBDIR|tests/test_data/|" \
    | sed "s/MAX_DURATION/16.7/" \
    > /tmp/testing.yaml

# Hence we skip the state dict check because this isn't one of
# the 4 supported architectures
./scripts/train.sh --model_config=/tmp/testing.yaml --skip_state_dict_check \
    --data_dir=tests/test_data --train_manifests=peoples-speech-short.json \
    --val_manifests=peoples-speech-short.json --num_gpus=1 \
    --global_batch_size=2 --grad_accumulation_batches=1 --batch_split_factor=2 --training_steps=4 \
    --prediction_frequency 1

JSON_TRAIN_EXIT_CODE=$?

./scripts/val.sh --model_config=/tmp/testing.yaml --data_dir=tests/test_data \
    --val_manifests=peoples-speech-short.json --num_gpus=1

JSON_VAL_EXIT_CODE=$?

# Otherwise each test run will use up 700MB more disk space
rm -rf /results/*

# Now run a tarfile train:
./scripts/train.sh --model_config=/tmp/testing.yaml --skip_state_dict_check \
    --read_from_tar --num_gpus=1 \
    --data_dir=/ --train_tar_files /workspace/training/tests/test_data/webdataset-eg.tar \
    --val_tar_files /workspace/training/tests/test_data/webdataset-eg-with-periods.tar \
    --global_batch_size=2 --grad_accumulation_batches=1 --batch_split_factor=2 --training_steps=4 \
    --prediction_frequency 1

TARFILE_TRAIN_EXIT_CODE=$?

./scripts/val.sh --model_config=/tmp/testing.yaml --read_from_tar --num_gpus=1 \
    --data_dir=/ --val_tar_files \
    /workspace/training/tests/test_data/webdataset-eg-with-periods.tar

TARFILE_VAL_EXIT_CODE=$?

rm -rf /results/*


# Now run a json train with Hugging Face validation
./scripts/train.sh --model_config=/tmp/testing.yaml --skip_state_dict_check \
    --data_dir=tests/test_data --train_manifests=peoples-speech-short.json \
    --use_hugging_face \
    --hugging_face_val_dataset distil-whisper/librispeech_asr_dummy \
    --hugging_face_val_split validation[0:2] --num_gpus=1 \
    --global_batch_size=2 --grad_accumulation_batches=1 --batch_split_factor=2 \
    --training_steps=4  --prediction_frequency 1

HF_TRAIN_EXIT_CODE=$?

./scripts/val.sh --model_config=/tmp/testing.yaml --use_hugging_face \
    --hugging_face_val_dataset distil-whisper/librispeech_asr_dummy \
    --hugging_face_val_split validation[0:2] --num_gpus=1

HF_VAL_EXIT_CODE=$?

rm -rf /results/*

# cleanup any artefacts created during testing
# This is required because, on the host runner, the files (e.g. .hypothesis) are
# created inside docker (i.e. with root permissions) so the host runner can't
# delete them
./scripts/test_cleanup.sh

# If any of the above failed, then the script should fail
if [ $JSON_TRAIN_EXIT_CODE -ne 0 ] || [ $TARFILE_TRAIN_EXIT_CODE -ne 0 ] \
    || [ $JSON_VAL_EXIT_CODE -ne 0 ] || [ $TARFILE_VAL_EXIT_CODE -ne 0 ] \
    || [ $HF_TRAIN_EXIT_CODE -ne 0 ] || [ $HF_VAL_EXIT_CODE -ne 0 ]
then
    exit 1
fi
