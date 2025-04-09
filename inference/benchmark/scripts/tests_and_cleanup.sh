#!/bin/bash
set -Eeuo pipefail

cleanup() {
	# cleanup any artefacts created during testing
	# This is required because, on the host runner, the files (e.g. .hypothesis) are
	# created inside docker (i.e. with root permissions) so the host runner can't
	# delete them
	cp /workspace/caiman-asr-repo/caiman-asr/training/scripts/test_cleanup.sh scripts/
	./scripts/test_cleanup.sh
	rm scripts/test_cleanup.sh
}

trap cleanup EXIT

mkdir -p ~/.cache/myrtle/benchmark/sample/
# transcribe_caiman will use these saved results
# instead of contacting the server:
cp tests/test_data/1272-128104-0000.caiman-asr.trans ~/.cache/myrtle/benchmark/sample/
./transcribe_caiman.py --run_name sample --limit 1 --append_results caiman-base --address mock_host --custom_timestamp mock_time
# Raise error if results don't match
diff -q ~/.cache/myrtle/benchmark/sample/results.csv tests/test_data/expected_results.csv
