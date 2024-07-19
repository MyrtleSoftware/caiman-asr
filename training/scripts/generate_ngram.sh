#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 NGRAM_ORDER /path/to/transcripts.txt /path/to/ngram.arpa /path/to/ngram.binary"
    exit 1
}

# Validate the number of arguments
if [ "$#" -ne 4 ]; then
    usage
fi

NGRAM_ORDER=$1
PATH_TO_TXT=$2
PATH_TO_ARPA=$3
PATH_TO_BINARY=$4

cd ../kenlm/build
# Try to run the first command, if it fails, run the second command.
# If the text corpus is too small, the `--discount_fallback` argument must be used,
# which applies predefined discount values for smoothing, rather than estimating them from the data.
# This ensures model robustness despite limited data.
bin/lmplz -o $NGRAM_ORDER < $PATH_TO_TXT > $PATH_TO_ARPA || \
bin/lmplz --discount_fallback -o $NGRAM_ORDER < $PATH_TO_TXT > $PATH_TO_ARPA

bin/build_binary $PATH_TO_ARPA $PATH_TO_BINARY
