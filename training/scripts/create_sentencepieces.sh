# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################## EDIT AS REQUIRED ##########################

# If you are using a different dataset and/or model configuration,
# you will need to edit the following variables to match your setup.
# If you are changing the dataset, you will also need to edit the glob
# below that currently reads:
# '/datasets/LibriSpeech/librispeech-train-*-wav.json'
# to point at your dataset's json files.
SPM_SIZE=1023
CONFIG_NAME=testing-1023sp
MAX_DURATION_SECS=16.7
DATASET_NAME_LOWER_CASE="librispeech"

########################## EDIT AS REQUIRED ##########################

SPM_NAME=$DATASET_NAME_LOWER_CASE$SPM_SIZE
CONFIG=configs/${CONFIG_NAME}.yaml
RUN_CONFIG=configs/${CONFIG_NAME}_run.yaml

mkdir -p /datasets/sentencepieces
# EDIT line below to point to the correct json file(s)
jq -r '.[]["transcript"]' /datasets/LibriSpeech/librispeech-train-*-wav.json > /tmp/txt.txt
python -c "import sentencepiece as spm; spm.SentencePieceTrainer.train(input='/tmp/txt.txt', model_prefix='$SPM_NAME', vocab_size=$SPM_SIZE, character_coverage=1.0, bos_id=-1, eos_id=-1, model_type='unigram')"
cp $SPM_NAME.* /datasets/sentencepieces/
cat $CONFIG | sed s/SENTENCEPIECE/$SPM_NAME/g | sed s/MAX_DURATION/$MAX_DURATION_SECS/g > $RUN_CONFIG
