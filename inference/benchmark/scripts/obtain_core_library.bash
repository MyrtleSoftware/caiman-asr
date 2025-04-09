#!/usr/bin/env bash
set -Eeuo pipefail
mkdir /workspace/caiman-asr-repo
cd /workspace/caiman-asr-repo
git clone https://github.com/MyrtleSoftware/caiman-asr
cd /workspace/caiman-asr-repo/caiman-asr
CORE_VERSION=$(head -n 1 /workspace/benchmark/core_version)
git checkout $CORE_VERSION
