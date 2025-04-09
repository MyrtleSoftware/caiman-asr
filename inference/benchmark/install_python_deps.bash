#!/usr/bin/env bash
set -Eeuo pipefail
: ${USE_UV:=false}

if [ "$USE_UV" = true ]; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv --python 3.10
	source .venv/bin/activate
	uv pip install -r requirements.txt
else
	/usr/bin/python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
fi
version=$(head -n 1 core_version)

starting_dir=$(pwd)
caiman_dir=$(mktemp -d)
git clone https://github.com/MyrtleSoftware/caiman-asr $caiman_dir
cd $caiman_dir
git checkout $version
cd training
if [ "$USE_UV" = true ]; then
	uv pip install -e .
else
	pip install -e .
fi
cd $starting_dir

deactivate
