#!/usr/bin/env bash
set -Eeuo pipefail
nvidia-smi > /dev/null # exit if nvidia-smi isn't present
while true
do
    nvidia-smi >> $1
    sleep 5
done
