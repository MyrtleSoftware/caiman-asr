#!/usr/bin/env bash
set -Eeuo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive \
	apt-get install -y sox python3 python3-pip python3-venv git curl tzdata
