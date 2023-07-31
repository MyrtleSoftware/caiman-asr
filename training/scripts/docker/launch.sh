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

# Modified by Myrtle

#!/bin/bash

DATASETS=$1
CHECKPOINTS=$2
RESULTS=$3

# Any additional arguments are treated as volumes to be mounted
# This allows docker container to follow symlinks in the mounted
# <DATASETS>, <CHECKPOINTS>, and <RESULTS> directories to different
# drives on the host machine
volumes=""
for i in "${@:4}"
do
  volumes=$volumes"-v $i:$i "
done

docker run -it --rm \
  --gpus='all' \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$DATASETS":/datasets \
  -v "$CHECKPOINTS":/checkpoints/ \
  -v "$RESULTS":/results/ \
  -v $PWD:/code \
  -v $PWD:/workspace/rnnt \
  $volumes \
  -e TZ=$(cat /etc/timezone) \
  myrtle/rnnt:v1.3.0 sh -c "./scripts/docker/settimezone.sh && bash"
