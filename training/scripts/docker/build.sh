#!/bin/bash

DOCKER_NAME=$(cat docker_name)

docker build . --rm -t "$DOCKER_NAME"
