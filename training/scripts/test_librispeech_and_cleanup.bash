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

cat configs/testing-1023sp.yaml | sed 's|TRANSCRIPTS||' | sed 's|ON_MISSING|raise_error|' | sed 's|LANGS|en|' | sed 's|MIN_TPS|0.0|' | sed 's|MAX_TPS|inf|' >/tmp/testing-1023sp.yaml

export CONFIG_DIR=/tmp
# Edamame (the CI runner) has only 4GB GPU!
export MEL_BATCH_DURATION=450

EXTRA_ARGS="--dataset_parts dev-clean" \
	CONFIG_NAME=testing-1023sp \
	SPM_SIZE=1023 \
	TRAIN_MANIFEST_YAML=./configs/ls-dev-clean.yaml \
	./scripts/prepare_librispeech.sh
