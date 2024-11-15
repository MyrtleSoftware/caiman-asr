#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
	# cleanup any artefacts created during testing
	# This is required because, on the host runner, the files (e.g. .hypothesis) are
	# created inside docker (i.e. with root permissions) so the host runner can't
	# delete them
	./scripts/test_cleanup.sh
	rm -rf /datasets/LibriSpeechDummy
}

trap cleanup EXIT

python caiman_asr_train/data/make_datasets/hugging_face_to_json.py \
	--hugging_face_dataset distil-whisper/librispeech_asr_dummy \
	--data_dir /datasets/LibriSpeechDummy \
	--hugging_face_split validation
