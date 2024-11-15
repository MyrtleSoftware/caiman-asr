#!/bin/bash

# Don't run benchmarks because flaky
pytest lib/tests --ignore=lib/tests/custom_lstm/test_cuda_bench.py
TEST_EXIT_CODE=$?

# cleanup any artefacts created during testing
# This is required because, on the host runner, the files (e.g. .hypothesis) are
# created inside docker (i.e. with root permissions) so the host runner can't
# delete them
./scripts/test_cleanup.sh

exit $TEST_EXIT_CODE
