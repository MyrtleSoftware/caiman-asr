#!/usr/bin/env bash

: ${DATASET_NAME_LOWER_CASE:="librispeech"}
: ${MAX_DURATION_SECS:=16.7}
: ${SPM_SIZE:=8703}
: ${CONFIG_NAME:=base-8703sp}
: ${CONFIG_DIR:=configs}

SPM_NAME=$DATASET_NAME_LOWER_CASE$SPM_SIZE
CONFIG=${CONFIG_DIR}/${CONFIG_NAME}.yaml
RUN_CONFIG=${CONFIG_DIR}/${CONFIG_NAME}_run.yaml

WIN_SIZE=$(cat $CONFIG | grep "window_size" | awk '{print $2}')
STATS_SUBDIR=$DATASET_NAME_LOWER_CASE-winsz$WIN_SIZE
MEL_STATS_DIR=/datasets/stats/$STATS_SUBDIR
NGRAM_DIR=/datasets/ngrams/$SPM_NAME
PATH_TO_TXT=$NGRAM_DIR/transcripts.txt
PATH_TO_ARPA=$NGRAM_DIR/ngram.arpa
PATH_TO_BINARY=$NGRAM_DIR/ngram.binary

# populate run config
cat $CONFIG | \
    sed s/SENTENCEPIECE/$SPM_NAME/g | \
    sed s/STATS_SUBDIR/$STATS_SUBDIR/g | \
    sed s/MAX_DURATION/$MAX_DURATION_SECS/g | \
    sed s/NGRAM_SUBDIR/$SPM_NAME/g > $RUN_CONFIG

echo "Setting env variables:"
echo ""
echo "DATASET_NAME_LOWER_CASE=$DATASET_NAME_LOWER_CASE"
echo "SPM_SIZE=$SPM_SIZE"
echo "SPM_NAME=$SPM_NAME"
echo "CONFIG=$CONFIG"
echo "STATS_SUBDIR=$STATS_SUBDIR"
echo "MEL_STATS_DIR=$MEL_STATS_DIR"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "RUN_CONFIG=$RUN_CONFIG"
echo "NGRAM_DIR=$NGRAM_DIR"
echo ""
