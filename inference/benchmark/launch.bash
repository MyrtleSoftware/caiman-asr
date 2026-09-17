#!/usr/bin/env bash

if [ "$USER" == 'root' ]; then
	container_owner="$SUDO_USER"
else
	container_owner="$USER"
fi

: ${DEVELOPER_MODE=false}
: ${TTY=true}
: ${HOST_CACHE="/home/${container_owner}/.cache/myrtle"}

DOCKER_ARGS=""
if [ "$TTY" = true ]; then
	DOCKER_ARGS+="-t "
fi

if [ "$#" -eq 0 ]; then
	command="/bin/bash"
else
	command="$*"
fi

if $DEVELOPER_MODE; then
	DOCKER_NAME=myrtle_benchmark_main
else
	DOCKER_NAME=myrtle_benchmark_$(head -n 1 core_version)
fi

DOCKER_ARGS+=" --network=host "
DOCKER_ARGS+=" -v $PWD:/workspace/benchmark "
DOCKER_ARGS+=" -v $HOST_CACHE:/root/.cache/myrtle "
DOCKER_ARGS+=" -e USER_HOME=/home/$container_owner "
DOCKER_ARGS+=" -e TZ=$(cat /etc/timezone) "

# As in training/scripts/docker/launch.sh: the container runs as root and mounts
# the checkout, so keep bytecode, pytest's cache, and hypothesis's example
# database out of the mounted tree -- the unprivileged CI runner can't delete
# root-owned artefacts. PYTEST_ADDOPTS may grow spaces, so it is passed as a
# separately quoted -e at the `docker run` call below (DOCKER_ARGS is expanded
# unquoted).
DOCKER_ARGS+=" -e PYTHONDONTWRITEBYTECODE=1 "
DOCKER_ARGS+=" -e HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis "
PYTEST_ENV="PYTEST_ADDOPTS=${PYTEST_ADDOPTS:+$PYTEST_ADDOPTS }--override-ini=cache_dir=/tmp/pytest_cache"

# Name the container so CI cleanup can `docker rm -f` it if the job is cancelled.
if [ -n "${CI_CONTAINER:-}" ]; then
	DOCKER_ARGS+=" --name $CI_CONTAINER "
fi

docker run -i --rm $DOCKER_ARGS -e "$PYTEST_ENV" "$DOCKER_NAME" sh -c "/workspace/caiman-asr-repo/caiman-asr/training/scripts/docker/settimezone.sh && $command; ret=\$?; chmod -R 777 /root/.cache/myrtle; exit \$ret"
