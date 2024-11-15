#!/usr/bin/env bash
set -Eeuo pipefail

cleanup() {
	# cleanup any artefacts created during testing
	# This is required because, on the host runner, the files (e.g. .hypothesis) are
	# created inside docker (i.e. with root permissions) so the host runner can't
	# delete them
	./scripts/test_cleanup.sh
	rm -f test_spm.model test_spm.vocab
}

trap cleanup EXIT

python caiman_asr_train/data/webdataset/webdataset_spm.py \
	--dataset_dir tests/test_data/ \
	--train_tar_files webdataset-eg-with-periods.tar \
	--spm_name test_spm --spm_size 20 --model_config configs/testing-1023sp_run.yaml
