#!/bin/bash

cd /workspace
git clone --depth 1 --branch v2.6.2 https://github.com/NVIDIA/NeMo.git
cd /workspace/NeMo
pip install -e .[asr]
