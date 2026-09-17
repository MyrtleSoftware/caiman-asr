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
volumes="-v $DATASETS:$DATASETS "
for i in "${@:4}"; do
	volumes=$volumes"-v $i:$i "
done

: ${DOCKER_NAME:=$(cat docker_name)}
: ${EXTRA_VOLUMES:="-v $PWD:/workspace/training"}
: ${COMMAND:="bash"}
: ${TTY=true}
: ${HF_CACHE:="$HOME/.cache/huggingface"}
: ${MYRTLE_CACHE:="$HOME/.cache/myrtle"}
: ${TILELANG_CACHE:="$HOME/.tilelang/cache"}
: ${PORTS:=""}

DOCKER_ARGS=""

if [ "$TTY" = true ]; then
	DOCKER_ARGS+="-t "
fi

DOCKER_ARGS+="-i --rm --gpus=all --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 "
DOCKER_ARGS+="-v $CHECKPOINTS:/checkpoints/ "
DOCKER_ARGS+="-v $RESULTS:/results/ "
DOCKER_ARGS+="-v $PWD:/code "

# Keep cached Hugging Face datasets even when restarting the container
hf_cache_volume="$HF_CACHE:/root/.cache/huggingface"
DOCKER_ARGS+=" -v $hf_cache_volume "

# Keep cached data even when restarting the container
myrtle_cache_volume="$MYRTLE_CACHE:/root/.cache/myrtle"
DOCKER_ARGS+=" -v $myrtle_cache_volume "

# Keep compiled tilelang kernels (and the autotune bench cache) across containers.
# Pre-create the host dir so it stays user-owned (docker would make it root-owned).
mkdir -p "$TILELANG_CACHE"
tilelang_cache_volume="$TILELANG_CACHE:/root/.tilelang/cache"
DOCKER_ARGS+=" -v $tilelang_cache_volume "

DOCKER_ARGS+="$EXTRA_VOLUMES $volumes "
DOCKER_ARGS+="-e TZ=$(cat /etc/timezone) "
# The container runs as root and mounts the repo, so any .pyc / cache it writes into
# the mounted tree is root-owned and the unprivileged CI runner can't clean it -- a
# later actions/checkout then fails to remove it (EACCES). Don't write bytecode at all,
# and keep pytest's cache and hypothesis's example database in the container's
# ephemeral /tmp rather than the mount (hypothesis writes to $CWD/.hypothesis by
# default, independently of pytest's cache_dir).
DOCKER_ARGS+="-e PYTHONDONTWRITEBYTECODE=1 "
DOCKER_ARGS+="-e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis "
# PYTEST_ADDOPTS may grow spaces, and DOCKER_ARGS is expanded unquoted -- pass it
# as a separately quoted -e at the `docker run` call below.
PYTEST_ENV="PYTEST_ADDOPTS=${PYTEST_ADDOPTS:+$PYTEST_ADDOPTS }--override-ini=cache_dir=/tmp/pytest_cache"

if [ "$USER" == 'root' ]; then
	container_owner="$SUDO_USER"
else
	container_owner="$USER"
fi
DOCKER_ARGS+="-e CONTAINER_OWNER=$container_owner "

if [ -n "${SNAKEVIZ_PORT+x}" ]; then
	DOCKER_ARGS+="-e SNAKEVIZ_PORT=$SNAKEVIZ_PORT "
	# Expose this port so snakeviz can use it
	DOCKER_ARGS+="-p $SNAKEVIZ_PORT:$SNAKEVIZ_PORT "
fi

if [ -n "${PORTS}" ]; then
	DOCKER_ARGS+="-p $PORTS "
fi

# Name the container (Jenkins sets JENKINS_JOB, GitHub CI sets CI_CONTAINER) so
# that cleanup can `docker rm -f` it if the job is cancelled mid-run.
: "${CI_CONTAINER:=${JENKINS_JOB:-}}"
if [ -n "$CI_CONTAINER" ]; then
	DOCKER_ARGS+="--name $CI_CONTAINER "
fi

# Symlink to /datasets to preserve relative symlinks between drives
LINK="ln -s $DATASETS /datasets"

docker run ${DOCKER_ARGS} -e "$PYTEST_ENV" ${DOCKER_NAME} sh -c "/workspace/training/scripts/docker/settimezone.sh && $LINK && $COMMAND"
