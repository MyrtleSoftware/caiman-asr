#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

export OMP_NUM_THREADS=1

: ${PYTHON_COMMAND:="./rnnt_train/val_multiple.py"}

TIMESTAMP=$(date '+%Y_%m_%d_%H_%M_%S')

ARGS+=" --timestamp=$TIMESTAMP"
# the following args are ignored by val_multiple.py but are 'required' as we reuse the val args so
ARGS+=" --dataset_dir=/tmp"
ARGS+=" --val_manifests=/tmp"

${PYTHON_COMMAND} "$@" $ARGS
