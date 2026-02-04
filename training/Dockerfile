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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.10-py3
FROM ${FROM_IMAGE_NAME}

# pytorch version taken from here:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10
ENV PYTORCH_VERSION=2.1.0a0+32f93b1

# fix NVIDIA key rotation problem.
# See https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for details.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# need to set the tzdata time noninteractively
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y libsndfile1 sox libsox-fmt-mp3 git cmake jq ffmpeg fish && \
    apt-get install -y --no-install-recommends numactl && \
    apt-get install -y iproute2 && \
    # Used for logging CPU usage:
    apt-get install -y aha && \
    # Install Boost and Eigen3 dependencies for kenlm:
    apt-get install -y libboost-all-dev libeigen3-dev && \
    # For Jenkins
    apt-get install -y openjdk-17-jre uuid-runtime && \
    rm -rf /var/lib/apt/lists/*

# Install KenLM
WORKDIR /workspace
RUN wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz && \
    mkdir kenlm/build && \
    cd kenlm/build && \
    cmake .. && \
    make -j2

WORKDIR /workspace/training

# Install the latest (beta) version of package torchdata before development paused:
# https://github.com/pytorch/data/issues/1196
# This may change when new torchdata releases are available
RUN pip install --no-dependencies torchdata==0.6.1

WORKDIR /workspace/audio
RUN git clone https://github.com/pytorch/audio && \
    cd audio && \
    git checkout v2.1.0 && \
    python setup.py install
WORKDIR /workspace/training

COPY lib lib
# Setuptool requires we are in the working directory of the package
WORKDIR /workspace/training/lib
# Separating the build/install steps gives better stdout/stderr diagnostics.
RUN python setup.py build -j $(nproc)
# This is a non-editable install (we need it to put the cuda extensions in the module path which -e does not do)
RUN python -m pip install .
# Reset the workspace, needed by following scripts
WORKDIR /workspace/training

COPY requirements.txt .
RUN pip install --disable-pip-version-check -U -r requirements.txt

COPY caiman_asr_train caiman_asr_train
COPY setup.py .
RUN python -m pip install -e .

# Set the library path to prevent UCX error after installing KenLM
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
