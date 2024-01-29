#!/usr/bin/env bash
set -Eeuo pipefail
while true
do
    # Start htop, immediately quit it with `q`,
    # and reformat the output to HTML.
    # From https://askubuntu.com/a/726661
    echo q | htop --sort-key PERCENT_CPU | aha --black --line-fix >> $1
    sleep 5
done
