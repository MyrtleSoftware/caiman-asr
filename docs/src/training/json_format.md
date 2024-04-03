# `JSON` format

The `JSON` format is the default in this repository and if you are training on your own data it is recommended to manipulate it into this format. Note that the data preparation steps are slightly different given the model you have decided to train so please refer to the [model configuration](model_yaml_configurations.md) page first.

### Page contents

* [Prepare LibriSpeech in `JSON` format](#librispeech_json)
  * [Quick Start](#quick_start)
  * [Details: LibriSpeech download](#download_librispeech_sh)
  * [Details: LibriSpeech data preparation](#preprocess_librispeech_sh)
* [Prepare your own dataset in `JSON` format](#other_datasets_json)

## Prepare LibriSpeech in `JSON` format <a name="librispeech_json"></a>

This page takes LibriSpeech as it is distributed from the www.openslr.org website and prepares it into a JSON manifest format.

### Quick Start <a name="quick_start"></a>

To run the data preparation steps for LibriSpeech and the `base` model run the following from the `training/` directory:

```bash
# Download data to /datasets/LibriSpeech: requires 60GB of disk
./scripts/download_librispeech.sh
./scripts/preprocess_librispeech.sh
```

To run preprocessing for the `testing` or `large` configurations, instead run:

```bash
SPM_SIZE=1023 CONFIG_NAME=testing-1023sp ./scripts/preprocess_librispeech.sh
SPM_SIZE=17407 CONFIG_NAME=large-17407sp ./scripts/preprocess_librispeech.sh
```

In the next two sections, these steps are described in more detail.

### Further detail: `download_librispeech.sh` <a name="download_librispeech_sh"></a>

Having run the script, the following folders should exist inside the container:

* `/datasets/LibriSpeech`
  * `train-clean-100/`
  * `train-clean-360/`
  * `train-other-500/`
  * `dev-clean/`
  * `dev-other/`
  * `test-clean/`
  * `test-other/`

```admonish
If `~/datasets` on the host is mounted to `/datasets`, the downloaded data will be accessible outside the container at `~/datasets/LibriSpeech`.
```

### Further detail: `preprocess_librispeech.sh` <a name="preprocess_librispeech_sh"></a>

The script will:

1. Create `JSON` manifests for each subset of LibriSpeech
2. Create a sentencepiece tokenizer from the train-960h subset
3. Record log-mel stats for the train-960h subset
4. Populate the [missing fields](model_yaml_configurations.md#missing_yaml_fields) of a YAML configuration template

Having run the script, the respective files should exist at the following locations:

#### 1. JSON manifests

* `/datasets/LibriSpeech/`
  * `librispeech-train-clean-100.json`
  * `librispeech-train-clean-360.json`
  * `librispeech-train-other-500.json`
  * `librispeech-dev-clean.json`
  * `librispeech-dev-other.json`
  * `librispeech-test-clean.json`
  * `librispeech-test-other.json`

#### 2. Sentencepiece tokenizer

* `/datasets/sentencepieces/`
  * `librispeech-1023sp.model`
  * `librispeech-1023sp.vocab`

#### 3. Log-mel stats

* `/datasets/stats/STATS_SUBDIR`:
  * `melmeans.pt`
  * `meln.pt`
  * `melvars.pt`

The `STATS_SUBDIR` will differ depending on the model since these stats are affected by the feature extraction window size. They are:

* `testing`: `/datasets/stats/librispeech-winsz0.02`
* {`base`, `large`}: `/datasets/stats/librispeech-winsz0.025`

#### 4. `_run.yaml` config

In the `configs/` directory. Depending on the model you are training you will have one of:

* `testing`: `configs/testing-1023sp_run.yaml`
* `base`: `configs/base-8703sp_run.yaml`
* `large`: `configs/large-17407sp_run.yaml`

`_run` indicates that this is a complete config, not just a template.

## Preprocessing Other Datasets <a name="other_datasets_json">

To convert your own data into the `JSON` format, adapt the steps in `scripts/preprocess_librispeech.sh`. The `JSON` manifest creation step is specific to LibriSpeech, but the remaining steps should be configurable via env variables to the script. For example, if you have created a copy of the script called `scripts/preprocess_commonvoice.sh` you can run it like:

```bash
DATASET_NAME_LOWER_CASE=commonvoice DATA_DIR=/datasets/CommonVoice MAX_DURATION_SECS=10.0 scripts/preprocess_commonvoice.sh
```

where:

* `DATASET_NAME_LOWER_CASE` will determine the name of generated `SENTENCEPIECE` and `STATS_SUBDIR`
* `DATA_DIR` is the path to which `JSON` manifests will be written
* `MAX_DURATION_SECS` is number of seconds above which audio clips will be discarded during training

It is advised that you use all of your training data transcripts to build the sentencepiece tokenizer but it is ok to use a subset of the data to calculate the mel stats via the `--n_utterances_only` flag to `caiman_asr_train/utils/generate_mel_stats.py`.

### Next steps

Having run the data preparation steps, go to the [training docs](./training.md) to start training.

### See also

* [WebDataset format for training](WebDataset_format.md)
* [Supported dataset formats](supported_dataset_formats.md)
* [Input activation normalization](log_mel_feature_normalization.md)
