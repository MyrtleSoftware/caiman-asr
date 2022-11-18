#!/usr/bin/env bash

# Copyright (c) 2022 Myrtle.ai
# iria [& rob]

path="/datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en"
mp3path=$path/"clips"
wavpath=$path/"wav_clips"

mkdir -p $wavpath

# There is a single mp3 file in CommonVoice Version 10 which contains invalid mp3 data (says ffmpeg).
# The file appears in invalidated.tsv so would not have been used in any case.
# We remove it here.
rm -f $mp3path/common_voice_en_680567.mp3

for FILE in $mp3path/*.mp3; do
    f=$(basename $FILE)
    # The original CommonVoice mp3 files are at a mixture of 32kHz and 48kHz sampling rates
    # Convert them into 16kHz 16-bit mono wav files
    ffmpeg -i "$FILE" -acodec pcm_s16le -ac 1 -ar 16000 -y $wavpath/${f%.mp3}.wav >& /dev/null
done
