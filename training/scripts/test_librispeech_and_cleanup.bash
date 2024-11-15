#!/bin/bash
if [ -n "$(ls -A /datasets)" ]; then
	echo "Stopping script early"
	echo "/datasets is not empty and this script would delete the contents of /datasets"
	exit 1
fi

cleanup() {
	# cleanup any artefacts created during testing
	# This is required because, on the host runner, the files (e.g. .hypothesis) are
	# created inside docker (i.e. with root permissions) so the host runner can't
	# delete them
	./scripts/test_cleanup.sh
	# Otherwise each test run will use up 670MB more disk space
	rm -rf /datasets/*
}

trap cleanup EXIT

EXTRA_ARGS="--dataset_parts dev-clean" \
	CONFIG_NAME=testing-1023sp \
	SPM_SIZE=1023 \
	TRAIN_MANIFESTS=librispeech-dev-clean-flac.json \
	./scripts/prepare_librispeech.sh
