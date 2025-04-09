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

docker run -i $DOCKER_ARGS "$DOCKER_NAME" sh -c "/workspace/caiman-asr-repo/caiman-asr/training/scripts/docker/settimezone.sh && $command; ret=\$?; chmod -R 777 /root/.cache/myrtle; exit \$ret"
