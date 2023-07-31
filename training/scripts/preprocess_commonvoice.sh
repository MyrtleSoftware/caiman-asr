#!/usr/bin/env bash

python ./rnnt_train/utils/convert_commonvoice.py \
    --input_dir /datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en \
    --input_tsv train.tsv \
    --output_json train.json

python ./rnnt_train/utils/convert_commonvoice.py \
    --input_dir /datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en \
    --input_tsv dev.tsv \
    --output_json dev.json

python ./rnnt_train/utils/convert_commonvoice.py \
    --input_dir /datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en \
    --input_tsv test.tsv \
    --output_json test.json

bash scripts/create_commonvoice_sentencepieces.sh
