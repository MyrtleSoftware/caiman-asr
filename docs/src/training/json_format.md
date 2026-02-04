# `JSON` format

The `JSON` format is the default in this repository and if you are training on your own data it is recommended to manipulate it into this format. Note that the data preparation steps are slightly different given the model you have decided to train so please refer to the [model configuration](model_yaml_configurations.md) page first.

### Page contents

- [Prepare LibriSpeech in `JSON` format](#librispeech_json)
  - [Quick Start](#quick_start)
  - [Details: LibriSpeech data preparation](#prepare_librispeech_sh)
- [Prepare your own dataset in `JSON` format](#other_datasets_json)

## Prepare LibriSpeech in `JSON` format <a name="librispeech_json"></a>

This page takes LibriSpeech as it is distributed from the <https://www.openslr.org> website and prepares it into a JSON manifest format.

### Quick Start <a name="quick_start"></a>

To run the data preparation steps for LibriSpeech and the `base` model run the following from the `training/` directory:

```bash
# Download data to /datasets/LibriSpeech: requires 120GB of disk
./scripts/prepare_librispeech.sh
```

To run preprocessing for the `testing` or `large` configurations, instead run:

```bash
SPM_SIZE=1023 CONFIG_NAME=testing-1023sp ./scripts/prepare_librispeech.sh
SPM_SIZE=17407 CONFIG_NAME=large-17407sp ./scripts/prepare_librispeech.sh
```

```admonish
If `~/datasets` on the host is mounted to `/datasets`, the downloaded data will be accessible outside the container at `~/datasets/LibriSpeech`.
```

### Further detail: `prepare_librispeech.sh` <a name="prepare_librispeech_sh"></a>

The script will:

1. Download data
2. Create `JSON` manifests for each subset of LibriSpeech
3. Convert the manifests into end-pointed manifests
4. Create a sentencepiece tokenizer from the train-960h subset
5. Record log-mel stats for the train-960h subset
6. Populate the [missing fields](model_yaml_configurations.md#missing_yaml_fields) of a YAML configuration template
7. Generate an n-gram language model with KenLM from the train-960h subset

#### 1. Data download

Having run the script, the following folders should exist inside the container:

- `/datasets/LibriSpeech`
  - `train-clean-100/`
  - `train-clean-360/`
  - `train-other-500/`
  - `dev-clean/`
  - `dev-other/`
  - `test-clean/`
  - `test-other/`

#### 2. JSON manifests

- `/datasets/LibriSpeech/`
  - `librispeech-train-clean-100-flac.json`
  - `librispeech-train-clean-360-flac.json`
  - `librispeech-train-other-500-flac.json`
  - `librispeech-train-clean-100-flac.eos.json`
  - `librispeech-train-clean-360-flac.eos.json`
  - `librispeech-train-other-500-flac.eos.json`
  - `librispeech-dev-clean-flac.json`
  - `librispeech-dev-other-flac.json`
  - `librispeech-test-clean-flac.json`
  - `librispeech-test-other-flac.json`

#### 3. Sentencepiece tokenizer

- `/datasets/sentencepieces/`
  - `librispeech8703.model`
  - `librispeech8703.vocab`

#### 4. Log-mel stats

- `/datasets/stats/STATS_SUBDIR`:
  - `melmeans.pt`
  - `meln.pt`
  - `melvars.pt`

The `STATS_SUBDIR` will differ depending on the model since these stats are affected by the feature extraction window size. They are:

- `testing`: `/datasets/stats/librispeech-winsz0.02`
- {`base`, `large`}: `/datasets/stats/librispeech-winsz0.025`

#### 5. `_run.yaml` config

In the `configs/` directory. Depending on the model you are training you will have one of:

- `testing`: `configs/testing-1023sp_run.yaml`
- `base`: `configs/base-8703sp_run.yaml`
- `large`: `configs/large-17407sp_run.yaml`

`_run` indicates that this is a complete config, not just a template.

#### 6. N-gram language model

- `/datasets/ngrams/librispeech8703/`
  - `transcripts.txt`
  - `ngram.arpa`
  - `ngram.binary`

To train an n-gram on a different dataset, see [n-gram docs](ngram_lm.md).

## Prepare Other Datasets <a name="other_datasets_json">

### Convert your dataset to the `JSON` format

Options:

- Adapt the code in `caiman_asr_train/data/make_datasets/librispeech.py`.
- If your dataset is in Hugging Face format,
  you can use the script described
  [here](hugging_face_dataset_format.md#converting-a-hugging-face-dataset-to-json-format)

### Generate artifacts needed for training

Suppose you have preprocessed CommonVoice, organized like this:

```
CommonVoice17.0
|-- common_voice_17.0_dev
|-- common_voice_17.0_dev.json
|-- common_voice_17.0_test
|-- common_voice_17.0_test.json
|-- common_voice_17.0_train
|-- common_voice_17.0_train.json
```

To generate the training artifacts, run the following:

```bash
DATASET_NAME_LOWER_CASE=commonvoice
MAX_DURATION_SECS=20.0
SPM_SIZE=8703
CONFIG_NAME=base-8703sp
DATA_DIR=/datasets/CommonVoice17.0
NGRAM_ORDER=4
TRAIN_MANIFESTS=/datasets/CommonVoice17.0/common_voice_17.0_train.json
./scripts/make_json_artifacts.sh $DATASET_NAME_LOWER_CASE $MAX_DURATION_SECS \
    $SPM_SIZE $CONFIG_NAME $DATA_DIR $NGRAM_ORDER $TRAIN_MANIFESTS
```

where:

- `DATASET_NAME_LOWER_CASE` will determine the name of the generated `SENTENCEPIECE` and `STATS_SUBDIR`
- `MAX_DURATION_SECS` is number of seconds above which audio clips will be discarded during training
- `SPM_SIZE` is the size of the sentencepiece model---in this case, the base model
- `CONFIG_NAME` is the name of the template configuration file to read
- `DATA_DIR` is the path to your dataset
- `NGRAM_ORDER` is the order of the n-gram language model that can be used during beam search
- `TRAIN_MANIFESTS` can be a space-separated list

It is advised that you use all of your training data transcripts to build the sentencepiece tokenizer but it is ok to use a subset of the data to calculate the mel stats via the `--n_utterances_only` flag to `caiman_asr_train/data/generate_mel_stats.py`.

Before running make_json_artifacts.sh on your custom dataset, you may want to create an EOS version as explained [here](./endpointing.md#generating-segmented-transcripts)

## Selecting transcripts

Each JSON manifest should include a "transcript" field, which contains the
transcript of the corresponding audio file. However, the key for the transcript
field can be customized, the default is "transcript".

If multiple transcripts exist for the same audio file, the model can be trained
using a specific transcript by specifying it in the use_transcripts field of
the YAML configuration:

```yaml
use_transcripts:
  to_use: ["<transcript_key_1>", "<transcript_key_2>", ...]
  on_missing: "raise_error" # or "skip" or "use_default"
```

The first transcript key that is found in the JSON manifest will be used.

If no key is found in the JSON manifest, the behaviour will depend on the
value of the on_missing field:

- "raise_error": an error will be raised.
- "skip": the audio file will be skipped.
- "use_default": the default transcript will be used, if this does not exist,
  an error will be raised.

Selecting transcripts is only supported when the JSON format is used.

If your dataset format is different, follow the respective documentation,
and use the default "transcript" field.

## Standardizing transcripts

Training on multiple datasets can negatively impact performance, as identical words may
be transcribed differently across datasets - for example, "colour" (British English)
vs "color" (American English). This inconsistency splits the model's probability distribution
across multiple spellings, reducing prediction confidence.

To address this, we standardize the training transcripts using Whisper EnglishTextNormalizer.
This process follows the same approach as WER standardization
(see [WER calculation docs](./wer_calculation.md) for more details).

There are two methods to do this:

### Recommended approach: Pre-standardizing manifests

The recommended approach is to standardize your manifests before training using the `standardize_manifest.py` script:

```bash
python caiman_asr_train/data/text/standardize_manifest.py \
  --manifests input.json \
  --preserve-case \
  --preserve-punctuation
```

This creates standardized versions of each transcript field as `<field-name>-standardized` (e.g., `transcript-standardized`).
You can then use these standardized fields during training by configuring your model's YAML:

```yaml
use_transcripts:
  to_use: ["transcript-standardized"]
  on_missing: "raise_error" # or "skip" or "use_default"
```

This approach offers several advantages:

- Faster training startup (standardization is done only once)
- Ability to standardize multiple transcript fields in the same manifest

Multiple manifests and fields can be standardized at once:

```bash
python caiman_asr_train/data/text/standardize_manifest.py \
  --manifests input1.json input2.json \
  --transcript-fields transcript transcript2 transcript3 \
```

This will standardize the `transcript`, `transcript2`, and `transcript3` fields in each of `input1.json` and `input2.json`,
producing `input1_standardized.json` and `input2_standardized.json` with the following fields:

- `transcript`
- `transcript2`
- `transcript3`
- `transcript-standardized`
- `transcript2-standardized`
- `transcript3-standardized`

All arguments:

- `--manifests [str] ...`: Absolute path to dataset manifest (.json) files [required].
- `--transcript-fields [str] ...`: Names of transcript fields to standardize [default: "transcript"].
- `--preserve-case`: Flag to preserve the original casing of the text [optional].
- `--preserve-punctuation`: Flag to preserve the original punctuation of the text [optional].
- `--no-expand-contractions`: Flag to disable expansion of contractions (e.g. "don't" -> "do not") [optional].
- `--output-dir [str]`: Directory to save standardized manifests [default: same directories as input manifests].
- `--output-filenames [str]`: Filenames to save standardized manifest
  [default: same as input manifest plus "\_standardized"].
- `--model-config [str]`: Path to model config to extract user symbols to preserve e.g. `<EOS>`
  \[default: `configs/testing-1023sp_run.yaml`\].
- `--num-workers [int]`: Number of worker processes to use [default: CPU count].
- `--overwrite`: Flag to enable overwrite of existing output files.

```admonish
If using offline standardization of manifests, please ensure to disable on-the-fly standardization by setting
`standardize_text` to false in the model config (.yaml) file.
```

```admonish
By default, standardization expands all contractions (e.g "don't" -> "do not").
As a result, if you train a model on standardized transcripts,
it will always predict the expanded form e.g. it will output "do not"
whether the speaker says "don't" or "do not".

If this behaviour is not desired,
you can disable contraction expansion during train-time standardization
by passing the `--no-expand-contractions` flag to the `standardize_manifest.py` script.
This will however worsen WERs by ~1-2% relative.
```

### Alternative: On-the-fly standardization

Alternatively, the YAML config includes the entry `standardize_text: true|false`,
enabling on-the-fly standardization of training transcripts during training.
This process standardizes all transcripts at the start of training,
which leads to longer start-up times, but can be more flexible if you are constantly changing datasets.

### Preserving casing and punctuation

By default, standardization lowercases and strips punctuation,
which is the desired behaviour for WER standardization.
However, this is undesirable when training models that predict
punctuation or casing (i.e. when `normalize_transcripts` is not set to `lowercase`).

Hence, both approaches support these standardization parameters:

- `preserve_case: true|false` - preserve the original casing of the text during standardization.
- `preserve_punctuation: true|false` - retain the original punctuation in the text during standardization.

```admonish
If `normalize_transcripts` is set to `lowercase`, these arguments have no effect,
as the text is already being lowercased before standardization.
```

Whether you choose pre-standardization or on-the-fly standardization, the process generally improves WER by ensuring consistent text formatting across datasets.

Default settings:

```yaml
normalize_transcripts: digit_to_word
...
standardize_text: true
preserve_case: true
preserve_punctuation: true
```

## Next steps

Having run the data preparation steps, go to the [training docs](./training.md) to start training.

### See also

- [WebDataset format for training](WebDataset_format.md)
- [Supported dataset formats](supported_dataset_formats.md)
- [Input activation normalization](log_mel_feature_normalization.md)
