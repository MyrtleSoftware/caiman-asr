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

ARG NVIDIA_IMAGE

# See https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-10.html for details
FROM nvcr.io/nvidia/pytorch:${NVIDIA_IMAGE}-py3

# Must be after the from statement
ARG TORCH_AUDIO_VERSION

# The 24.10 image pins the now-retired pypi.ngc.nvidia.com as an extra index
# A non-empty env value replaces it with NVIDIA's live successor.
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com

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
  # For pyicu
  apt-get install -y libicu-dev && \
  apt-get install -y pkg-config && \
  rm -rf /var/lib/apt/lists/*

# Install KenLM
WORKDIR /workspace
RUN wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz && \
  mkdir kenlm/build && \
  cd kenlm/build && \
  cmake .. && \
  make -j2

WORKDIR /workspace/training
COPY setup_nemo.sh .
RUN bash setup_nemo.sh

# We generate a constraints file to freeze all versions
# inside the container base image.
WORKDIR /workspace/training
RUN pip list --format=freeze > /workspace/constraints.txt

WORKDIR /workspace/training
COPY lib lib
# Setuptool requires we are in the working directory of the package
WORKDIR /workspace/training/lib
# Separating the build/install steps gives better stdout/stderr diagnostics.
RUN python setup.py build -j "$(nproc)"
# This is a non-editable install (we need it to put the cuda extensions in the module path which -e does not do)
RUN python -m pip install . --no-cache-dir -c /workspace/constraints.txt

WORKDIR /workspace/training
COPY requirements_final.txt ./requirements.txt
RUN pip install -r requirements.txt --no-cache-dir -c /workspace/constraints.txt

# Because we have a custom version of pytorch we must build torchaudio from
# source to ensure ABI compatibility, this version of torchaudio depends on
# the exact base commit of pytorch that this version of the container ships.
WORKDIR /workspace/
RUN git clone -b v${TORCH_AUDIO_VERSION} --single-branch https://github.com/pytorch/audio --depth 1
WORKDIR /workspace/audio
RUN pip install -v . -c /workspace/constraints.txt \
  --no-use-pep517 \
  --no-cache-dir \
  --no-deps \
  --no-build-isolation

WORKDIR /workspace/training
COPY caiman_asr_train caiman_asr_train
COPY setup.py .
RUN python -m pip install -e . --no-cache-dir -c /workspace/constraints.txt

# Set the library path to prevent UCX error after installing KenLM
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
