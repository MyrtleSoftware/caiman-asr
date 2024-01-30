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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.03-py3
FROM ${FROM_IMAGE_NAME}

# pytorch version taken from here:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-03.html#rel-23-03
ENV PYTORCH_VERSION=2.0.0a0+1767026

# Added by rob@myrtle May 2022 to fix NVIDIA key rotation problem.
# See https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for details.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# need to set the tzdata time noninteractively
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y libsndfile1 sox git cmake jq ffmpeg && \
    apt-get install -y --no-install-recommends numactl && \
    apt-get install -y iproute2 && \
    # Used for logging CPU usage:
    apt-get install -y aha && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/training

# Install the latest (beta) version of package torchdata before development paused:
# https://github.com/pytorch/data/issues/1196
# This may change when new torchdata releases are available
RUN pip install --no-dependencies torchdata==0.6.1

COPY requirements.txt .
RUN pip install --no-cache --disable-pip-version-check -U -r requirements.txt

COPY lib lib
# Setuptool requires we are in the working directory of the package
WORKDIR /workspace/training/lib
# # Separating the build/install steps gives better stdout/stderr diagnostics
RUN python setup.py build
# This is a non-editable install (we need it to put the cuda extensions in the module path which -e does not do)
RUN python -m pip install --use-feature=in-tree-build .
# Reset the workspace, needed by following scripts
WORKDIR /workspace/training

COPY rnnt_train rnnt_train
COPY setup.py .
RUN python -m pip install -e .
