# ML training flow

This document describes the flow of training the `base` model on LibriSpeech.
This configuration is used as an example as it is quicker to train than `large`.

## Environment Setup

Clone the repo, build the image and set up the container with the appropriate volumes
(as described [here](installation.md#install)) with the following commands:

```bash
git clone https://github.com/MyrtleSoftware/caiman-asr.git && cd caiman-asr/training
```

```bash
./scripts/docker/build.sh
```

```bash
./scripts/docker/launch.sh <DATASETS> <CHECKPOINTS> <RESULTS>
```

## Data Preparation

From inside the container, run the following command to download LibriSpeech, prepare JSON manifests, create a tokenizer, and a populated yaml configuration file `configs/base-8703sp_run.yaml`.

```bash
./scripts/prepare_librispeech.sh
```

More details on preparing LibriSpeech into a JSON format can be found [here](json_format.md#prepare-librispeech-in-json-format).

## Training

Modify `<NUM_GPU>` based on your machine and then run the following command to train a `base` model.
A more detailed description of the training process can be found [here](training.md#training).

```bash
./scripts/train.sh \
  --data_dir /datasets/LibriSpeech \
  --train_manifests librispeech-train-clean-100-flac.json librispeech-train-clean-360-flac.json librispeech-train-other-500-flac.json \
  --val_manifests librispeech-dev-clean-flac.json \
  --model_config configs/base-8703sp_run.yaml \
  --num_gpus 2 \
  --global_batch_size 1024 \
  --grad_accumulation_batches 8 \
  --batch_split_factor 8 \
  --val_batch_size 1 \
  --training_steps 42000
```

In particular, this command assumes you're using a
2 x RTX4090 (24GB) system. See [here](batch_size_hyperparameters.md)
for how to adjust these numbers for your system.

## Validation

The following command will run the validation script and calculate the WER [%].
See [here](validation.md#validation) for more details.

```bash
./scripts/val.sh --model_config configs/base-8703sp_run.yaml
```
