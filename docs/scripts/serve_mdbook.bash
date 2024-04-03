#!/usr/bin/env sh
PORT="${1:-3000}"
docker run -i -p $PORT:$PORT \
    -v $(pwd):/code/docs \
    -v $(realpath -e $(pwd)/../training):/code/training \
    myrtle_caiman_asr_docs \
    ./scripts/serve_mdbook_inside_docker.bash $PORT
