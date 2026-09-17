#!/bin/bash
set -Eeuo pipefail

cleanup() {
	rm -rf setup_nemo.sh requirements_final.txt
}

trap cleanup EXIT

DOCKER_NAME=$(cat docker_name)

cp requirements.txt requirements_final.txt

# Host-specific defaults, but only if user has NOT already set them
if [[ ${HOSTNAME:-} == "leavitt" || ${HOSTNAME:-} == "edamame" ]]; then
	: "${NVIDIA_IMAGE:=24.10}"
	: "${TORCH_AUDIO_VERSION:=2.6.0}"
else
	: "${NVIDIA_IMAGE:=25.10}"
	: "${TORCH_AUDIO_VERSION:=2.9.0}"
fi

# Old container requires datasets>=2.2.1,<=4.4.1
# Nemo has datasets as an unpinned dependency and when it installs
# it resolves to datasets==4.8.5 which causes a conflict
# Since Nemo isn't required for anything that's a part of the training
# pipeline, we can omit it from the old container and install it
# only with the new container
if [[ $NVIDIA_IMAGE > 24.10 ]]; then
	sed -i 's/datasets>=2.2.1,<=4.4.1/datasets>=2.2.1/g' requirements_final.txt
	sed -i 's/jiwer==3.0.5/jiwer==3.1.0/g' requirements_final.txt
	cp install_nemo.sh setup_nemo.sh
else
	touch setup_nemo.sh
fi

echo "Building Docker image $DOCKER_NAME with NVIDIA_IMAGE=$NVIDIA_IMAGE and TORCH_AUDIO_VERSION=$TORCH_AUDIO_VERSION"

docker build . --rm -t "$DOCKER_NAME" \
	--build-arg NVIDIA_IMAGE="${NVIDIA_IMAGE}" \
	--build-arg TORCH_AUDIO_VERSION="${TORCH_AUDIO_VERSION}"
