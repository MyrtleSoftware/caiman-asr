#!/usr/bin/env bash
set -Eeuo pipefail

if [ -z "${SNAKEVIZ_PORT+x}" ]; then
	echo "This docker container doesn't have a port to the host,"
	echo "so you won't be able to view snakeviz."
	echo "Please exit the container and relaunch it via:"
	echo "SNAKEVIZ_PORT=[an unused port] ./scripts/docker/launch.sh ..."
	exit 1
fi

# Takes a file path as an argument and runs snakeviz on it
# The extra arguments are there to make it work within Docker
snakeviz -H 0.0.0.0 -s -p $SNAKEVIZ_PORT $1
