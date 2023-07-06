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

# modified from create_sentencepieces.sh by rob@myrtle

mkdir -p /datasets/sentencepieces
jq -r '.[]["transcript"]' /datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en/train.json > /tmp/txt.txt
python -c "import sentencepiece as spm; spm.SentencePieceTrainer.train(input='/tmp/txt.txt', model_prefix='commonvoice1023', vocab_size=1023, character_coverage=1.0, bos_id=-1, eos_id=-1, model_type='unigram')"
cp commonvoice1023.* /datasets/sentencepieces/
cat configs/testing-1023sp.yaml | sed s/SENTENCEPIECE/commonvoice1023/g | sed s/MAX_DURATION/7.75/g > configs/testing-1023sp_run.yaml
