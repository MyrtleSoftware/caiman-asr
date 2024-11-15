#!/usr/bin/env sh
docker run -i \
	-v $(realpath -e $(pwd)/..):/code \
	myrtle_asr_docs \
	./scripts/build_mdbook_inside_docker.bash
