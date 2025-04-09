FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y sox python3 python3-pip git curl tzdata

WORKDIR /workspace/benchmark
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY core_version core_version

COPY tmp-repo-copy tmp-repo-copy
COPY scripts/obtain_core_library.bash scripts/obtain_core_library.bash
COPY scripts/obtain_core_library_dev.bash scripts/obtain_core_library_dev.bash
ARG obtain_core_script
RUN bash ${obtain_core_script}

WORKDIR /workspace/caiman-asr-repo/caiman-asr/training
RUN pip install -e .

WORKDIR /workspace/benchmark

ENV AM_I_IN_A_DOCKER_CONTAINER=Yes
