# WebDataset README

This README gives instructions to read training and validation data from the [WebDataset format](https://github.com/webdataset/webdataset#the-webdataset-format) as opposed to the default `json` format described in the [training README](../README.md) and, serves as a drop in replacement for the [Data section](../README.md#data) in the `webdataset` case.
It is assumed you have already completed the [install instructions in the main README](../README.md#installation).

In the `webdataset` format, `<key>.{flac,wav}` audio files are stored with associated `<key>.txt` transcripts in tar file shards. The tar file samples are read sequentially which increases I/O rates compared with random access.

## Data Preprocessing

All commands in this README should be run from the [`rnnt/training` directory](..).

### WebDataset building

If you would like to build your own WebDataset you should refer to the following resources:

1. Script that converts from WeNet legacy format to WebDataset: [make_shard_list.py](https://github.com/wenet-e2e/wenet/blob/main/tools/make_shard_list.py)
2. Tutorial on creating WebDataset shards: [Creating a WebDataset](https://webdataset.github.io/webdataset/creating/)

At tarfile creation time you must ensure that each audio file is stored sequentially with its associated .txt transcript file.

#### Text normalisation

As discussed in more detail in the [training README](../README.md#data_preprocess) it is necessary to normalize your transcripts so that they contain just spaces, apostrophes and lower-case letters. It is recommended to normalize samples on the fly by setting `normalize_transcripts: true` in your config file. Another option is to perform this step offline when you create the WebDataset shards.

### Tokenizer and config building

Before running training it is necessary to build a sentencepiece tokenizer model. In our experience we see appreciably lower WERs when we train the sentencepiece model on the same training data that we use for the RNN-T training. As such, it is recommended that you repeat this step if you change or expand your training data.

To create a sentencepiece model and config file first [start a container](../README.md#run-container) and then run the following script inside it:

```bash
DATA_DIR=/datasets/TarredDataset TRAIN_TAR_FILES="train_*tar.tar" DATASET_NAME_LOWER_CASE=librispeech960 ./scripts/preprocess_webdataset.sh
```

This will create a sentencepiece model at `/datasets/sentencepieces/<dataset-name>.*` and a `_run` config file in the `../configs/` directory that, by default, matches the [`base-8703sp.yaml`](../configs/base-8703sp.yaml) config. This script accepts the following arguments:

- `DATA_DIR`: Directory containing tar files.
- `TRAIN_TAR_FILES`: One or more shard file paths or globs.
- `DATASET_NAME_LOWER_CASE`: Name of dataset to use for naming sentencepiece model. Defaults to `librispeech`.
- `MAX_DURATION_SECS`: The maximum duration in seconds that you want to train on. Defaults to `16.7` as per LibriSpeech.
- `CONFIG_NAME`: Model name to use for the config [from this table](../README.md#models). Defaults to `base-8703sp`.
- `SPM_SIZE`: Sentencepiece model size. Must match `CONFIG_NAME`. Defaults to `8703`.

Note that, in this script, we normalize transcripts before building a sentencepiece model with the [_clean_text function](./../rnnt_train/common/text/ito/__init__.py) in order that the character set matches the one used for normalising transcripts during training.

## Training and validation

To trigger training or validation for data stored in WebDataset format you should pass `--read_from_tar` to `train.sh`, `val.sh`.

You will also need to pass `--val_tar_files` (and for training, `--train_tar_files`) as one or more tar shard files/globs in `data_dir`. For example if all of your training and tar files are in a flat `data_dir` directory you might run:

```bash
./scripts/train.sh --read_from_tar --data_dir=/datasets/TarredDataset --train_tar_files train_*.tar --val_tar_files dev_*.tar
```

where `{train,val}_tar_files` can be one or more filenames or fileglobs. In this mode, your training and validation tar files must have non-overlapping names. Alternatively, if you have a nested file structure you can set `--data_dir=/` and then pass absolute paths/globs to `--train_tar_files` and `--val_tar_files` for example like:

```bash
./scripts/train.sh --read_from_tar --data_dir=/ --train_tar_files /datasets/TarredDataset/train/** --val_tar_files /datasets/TarredDataset/dev/**
```

Note that in the second case (when paths are absolute), glob expansions will be performed by your shell rather than the `WebDatasetReader` class.

For validation you might run:

```bash
./scripts/val.sh --read_from_tar --data_dir=/datasets/TarredDataset --val_tar_files dev_*.tar
# or, absolute paths
./scripts/val.sh --read_from_tar --data_dir=/ --val_tar_files /datasets/TarredDataset/dev/**
```

You should refer to the [training README](../README.md#training) for more details on training and validation arguments unrelated to this data format.

## WebDataset Limitations

Our WebDataset support currently has the following limitations:

- It isn't currently possible to mix and match `json` and `webdataset` formats for the training and validation data passed to `./scripts/train.sh`.
- It is necessary to have more shards per dataset (including validation data) than `num_gpus` so that each GPU can read from a different shard.
