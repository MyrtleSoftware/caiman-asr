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

RESULTS=$1
: ${PORT:=${2:-6010}}
: ${NUM_SAMPLES:=${3:-1000}}

DOCKER_NAME=$(cat docker_name)

docker run -it --rm \
	--gpus='all' \
	--shm-size=4g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v "$RESULTS":/results/ \
	-v $PWD:/code \
	-v $PWD:/workspace/training \
	-p $PORT:$PORT \
	-e TZ=$(cat /etc/timezone) \
	"$DOCKER_NAME" sh -c "./scripts/docker/settimezone.sh && \
  tensorboard --logdir /results --host 0.0.0.0 --port $PORT \
  --samples_per_plugin=scalars=$NUM_SAMPLES"
