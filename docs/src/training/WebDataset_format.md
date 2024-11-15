# WebDataset format

This page gives instructions to read training and validation data from the [WebDataset format](https://github.com/webdataset/webdataset#the-webdataset-format) as opposed to the default `JSON` format described in the [Data Formats](supported_dataset_formats.md) documentation.

In the `WebDataset` format, `<key>.{flac,wav}` audio files are stored with associated `<key>.txt` transcripts in tar file shards. The tar file samples are read sequentially which increases I/O rates compared with random access.

## Data Preparation

All commands in this README should be run from the `training` directory of the repo.

### WebDataset building

If you would like to build your own WebDataset you should refer to the following resources:

1. Script that converts from WeNet legacy format to WebDataset: [`make_shard_list.py`](https://github.com/wenet-e2e/wenet/blob/main/tools/make_shard_list.py)
2. [Tutorial on creating WebDataset shards](https://webdataset.github.io/webdataset/creating/)

At tarfile creation time, you must ensure that each audio file is stored sequentially with its associated .txt transcript file.

#### Text normalization

```admonish
As discussed in more detail [here](./data_preparation.md#text_norm) it is necessary to normalize your transcripts so that they contain just spaces, apostrophes and lower-case letters. It is recommended to do this on the fly by setting `normalize_transcripts: true` in your config file. Another option is to perform this step offline when you create the WebDataset shards.
```

### Data preparation: `preprocess_webdataset.sh`

In order to create the artefacts described in the [data preparation intro](data_preparation.md), run the following inside a running container:

```bash
DATA_DIR=/datasets/TarredDataset TRAIN_TAR_FILES="train_*tar.tar" DATASET_NAME_LOWER_CASE=librispeech ./scripts/preprocess_webdataset.sh
```

This script accepts the following arguments:

- `DATA_DIR`: Directory containing tar files.
- `TRAIN_TAR_FILES`: One or more shard file paths or globs.
- `DATASET_NAME_LOWER_CASE`: Name of dataset to use for naming sentencepiece model. Defaults to `librispeech`.
- `MAX_DURATION_SECS`: The maximum duration in seconds that you want to train on. Defaults to `16.7` as per LibriSpeech.
- `CONFIG_NAME`: Model name to use for the config [from this table](model_yaml_configurations.md). Defaults to `base-8703sp`.
- `SPM_SIZE`: Sentencepiece model size. Must match `CONFIG_NAME`. Defaults to `8703`.
- `NGRAM_ORDER`: Order of n-gram language model. Defaults to 4.

## Training and validation

To trigger training or validation for data stored in WebDataset format you should pass `--read_from_tar` to `train.sh`, `val.sh`.

You will also need to pass `--val_tar_files` (and for training, `--train_tar_files`) as one or more tar shard files/globs in `--data_dir`. For example if all of your training and tar files are in a flat `--data_dir` directory you might run:

```bash
./scripts/train.sh --read_from_tar --data_dir=/datasets/TarredDataset --train_tar_files train_*.tar --val_tar_files dev_*.tar
```

where `{train,val}_tar_files` can be one or more filenames or fileglobs. In this mode, your training and validation tar files must have non-overlapping names. Alternatively, if you have a nested file structure you can set `--data_dir=/` and then pass absolute paths/globs to `--train_tar_files` and `--val_tar_files` for example like:

```bash
./scripts/train.sh --read_from_tar --data_dir=/ --train_tar_files /datasets/TarredDataset/train/** --val_tar_files /datasets/TarredDataset/dev/**
```

Note that in the second case (when paths are absolute), glob expansions will be performed by your shell rather than the `WebDatasetReader` class.

You should refer to the [Training command](training.md) documentation for more details on training arguments unrelated to this data format.

For validation you might run:

```bash
./scripts/val.sh --read_from_tar --data_dir=/datasets/TarredDataset --val_tar_files dev_*.tar
# or, absolute paths
./scripts/val.sh --read_from_tar --data_dir=/ --val_tar_files /datasets/TarredDataset/dev/**
```

```admonish
Training and validation support the use of zip files in addition to tar files.
Ensure that the zip files adhere to the  [WebDataset format](https://github.com/webdataset/webdataset#the-webdataset-format).
Additionally, be sure that the arguments passed to `--val_tar_files` and `--train_tar_files` are either all tar files or all zip files,
and not a combination of both formats.
```

## WebDataset Limitations

Our WebDataset support currently has the following limitations:

- It isn't currently possible to mix and match `JSON` and `WebDataset` formats for the training and validation data passed to `./scripts/train.sh`.
- It is necessary to have more shards per dataset (including validation data) than `num_gpus` so that each GPU can read from a different shard.
