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

: ${DOCKER_NAME:=$(cat docker_name)}
: ${EXTRA_VOLUMES:="-v $PWD:/workspace/training"}
: ${COMMAND:="bash"}
: ${TTY=true}
: ${HF_CACHE:="$HOME/.cache/huggingface"}
: ${PORTS:=""}

DOCKER_ARGS=""

if [ "$TTY" = true ] ; then
  DOCKER_ARGS+="-t "
fi

DOCKER_ARGS+="-i --rm --gpus=all --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 "
DOCKER_ARGS+="-v $DATASETS:/datasets "
DOCKER_ARGS+="-v $CHECKPOINTS:/checkpoints/ "
DOCKER_ARGS+="-v $RESULTS:/results/ "
DOCKER_ARGS+="-v $PWD:/code "

# Keep cached Hugging Face datasets even when restarting the container
hf_cache_volume="$HF_CACHE:/root/.cache/huggingface"
DOCKER_ARGS+=" -v $hf_cache_volume "

DOCKER_ARGS+="$EXTRA_VOLUMES $volumes "
DOCKER_ARGS+="-e TZ=$(cat /etc/timezone) "
DOCKER_ARGS+="-e CONTAINER_OWNER=$(logname) "

if [ -n "${SNAKEVIZ_PORT+x}" ]; then
    DOCKER_ARGS+="-e SNAKEVIZ_PORT=$SNAKEVIZ_PORT "
  # Expose this port so snakeviz can use it
    DOCKER_ARGS+="-p $SNAKEVIZ_PORT:$SNAKEVIZ_PORT "
fi

if [ -n "${PORTS}" ]; then
    DOCKER_ARGS+="-p $PORTS "
fi

docker run ${DOCKER_ARGS} ${DOCKER_NAME} sh -c "/workspace/training/scripts/docker/settimezone.sh && $COMMAND"
