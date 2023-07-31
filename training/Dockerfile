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

# modified by rob@myrtle

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.11-py3
FROM ${FROM_IMAGE_NAME}

# pytorch version taken from here:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-11.html#rel-22-11
ENV PYTORCH_VERSION=1.13.0a0+936e930

# Added by rob@myrtle May 2022 to fix NVIDIA key rotation problem.
# See https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for details.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y libsndfile1 sox git cmake jq ffmpeg && \
    apt-get install -y --no-install-recommends numactl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rnnt

# Install torchdata from source.
# This is necessary because, when running pip install torchdata, the PyTorch
# version in this nvcr image is not recognized as PYTORCH_VERSION=1.13.0a0+936e930
# is different to the `1.13.0` expected by pip.
#
# We also checkout to torchdata==v0.5.0 as this matches PyTorch 1.13.0
RUN mkdir deps -p && git clone https://github.com/pytorch/data.git deps/data && \
    cd deps/data && \
    git checkout v0.5.0 && \
    git submodule update --init --recursive --jobs 8 && \
    python setup.py install

COPY requirements.txt .
RUN pip install --no-cache --disable-pip-version-check -U -r requirements.txt

COPY . .
RUN python -m pip install -e .
