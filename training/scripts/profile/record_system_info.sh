#!/usr/bin/env bash

wget https://raw.github.com/tdulcet/Linux-System-Information/master/info.sh -qO - | bash -s >$1

lscpu >>$1

echo "GPU(s) model: " >>$1
nvidia-smi --query-gpu=name --format=csv,noheader >>$1
nvidia-smi >>$1
