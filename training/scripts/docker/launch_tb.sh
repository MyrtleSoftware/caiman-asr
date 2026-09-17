#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <results_dir1> [results_dir2 ...] [--port PORT] [--samples NUM] [--reload_interval SECONDS]"
	exit 1
fi

PORT=6010
NUM_SAMPLES=1000
RELOAD_INTERVAL=30
RESULT_DIRS=()

# Parse positional + optional arguments
while [[ $# -gt 0 ]]; do
	case "$1" in
	--port)
		PORT="$2"
		shift 2
		;;
	--samples)
		NUM_SAMPLES="$2"
		shift 2
		;;
	--reload_interval)
		RELOAD_INTERVAL="$2"
		shift 2
		;;
	*)
		RESULT_DIRS+=("$1")
		shift
		;;
	esac
done

DOCKER_NAME=$(cat docker_name)

# Build volume mounts
VOLUME_ARGS=()

for dir in "${RESULT_DIRS[@]}"; do
	VOLUME_ARGS+=(-v "$(realpath "$dir"):/results/$(realpath "$dir")")
done

echo "Launching TensorBoard on port ${PORT}"

docker run -it --rm \
	--gpus=all \
	--shm-size=4g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	"${VOLUME_ARGS[@]}" \
	-v "$PWD:/code" \
	-v "$PWD:/workspace/training" \
	-p "$PORT:$PORT" \
	-e TZ="$(cat /etc/timezone)" \
	"$DOCKER_NAME" sh -c "
    /workspace/training/scripts/docker/settimezone.sh && \
    tensorboard \
      --logdir /results/ \
      --bind_all \
      --port ${PORT} \
      --samples_per_plugin=scalars=${NUM_SAMPLES} \
      --reload_interval=${RELOAD_INTERVAL}
  "
