#!/usr/bin/env sh
docker run -i \
    -v $(pwd):/code/docs \
    -v $(realpath -e $(pwd)/../training):/code/training \
    myrtle_caiman_asr_docs \
    ./scripts/build_mdbook_inside_docker.bash
