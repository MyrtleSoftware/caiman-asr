#!/usr/bin/env sh

: ${DEVELOPER_MODE=false}

if $DEVELOPER_MODE; then
	DOCKER_NAME=myrtle_benchmark_main
	# Can't copy a folder into itself, so first
	# copy the repo to a temporary folder
	tmp_dir=$(mktemp -d)
	rsync -az --exclude=.git --exclude=.jj "$PWD"/../.. "$tmp_dir"
	# Put the copy of the repo in this folder
	# so the Dockerfile can see it
	cp -r "$tmp_dir" tmp-repo-copy
	DOCKER_ARGS=" --build-arg obtain_core_script=scripts/obtain_core_library_dev.bash "
else
	DOCKER_NAME=myrtle_benchmark_$(head -n 1 core_version)
	# Docker needs to able to copy this folder,
	# even if it is not used in the build
	touch tmp-repo-copy
	DOCKER_ARGS=" --build-arg obtain_core_script=scripts/obtain_core_library.bash "
fi

docker build . -t "$DOCKER_NAME" $DOCKER_ARGS

rm -rf tmp-repo-copy
