set -Eeuo pipefail

: ${SPM_SIZE:=}
: ${SPM_NAME:=}
: ${CONFIG_NAME:=}
: ${DATA_DIR:=}
: ${TRAIN_MANIFESTS:=}

if [[ -z "$SPM_SIZE" ]]; then
    echo "Must provide SPM_SIZE in env variable" 1>&2
    exit 1
fi
if [[ -z "$SPM_NAME" ]]; then
    echo "Must provide SPM_NAME in env variable" 1>&2
    exit 1
fi
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Must provide CONFIG_NAME in env variable" 1>&2
    exit 1
fi
if [[ -z "$DATA_DIR" ]]; then
    echo "Must provide DATA_DIR in env variable" 1>&2
    exit 1
fi
if [[ -z "$TRAIN_MANIFESTS" ]]; then
    echo "Must provide TRAIN_MANIFESTS in env variable" 1>&2
    exit 1
fi

# write transcripts to a tmpfile
rm -f /tmp/txt.txt
cd $DATA_DIR
for manifest in $TRAIN_MANIFESTS; do
    echo "Write transcripts to file for $manifest"
    jq -r '.[]["transcript"]' "$manifest" >> /tmp/txt.txt
done
cd -

python -c "import sentencepiece as spm; spm.SentencePieceTrainer.train(input='/tmp/txt.txt', model_prefix='$SPM_NAME', vocab_size=$SPM_SIZE, character_coverage=1.0, bos_id=-1, eos_id=-1, model_type='unigram', train_extremely_large_corpus=True)"
mkdir -p /datasets/sentencepieces
mv ${SPM_NAME}.* /datasets/sentencepieces/
