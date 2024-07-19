# ML training flow

This document describes the flow of training the `testing` model on LibriSpeech.
This configuration is used as an example as it is quicker to train than either `base` or `large`.

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

From inside the container, run the following command to download LibriSpeech, prepare JSON manifests, create a tokenizer, and a populated yaml configuration file.

```bash
SPM_SIZE=1023 CONFIG_NAME=testing-1023sp ./scripts/prepare_librispeech.sh
```

More details on preparing LibriSpeech into a JSON format can be found [here](json_format.md#prepare-librispeech-in-json-format).

## Training

Modify `<NUM_GPU>` based on your machine and then run the following command to train a `testing` model.
A more detailed description of the training process can be found [here](training.md#training).

```bash
./scripts/train.sh \
  --data_dir /datasets/LibriSpeech \
  --train_manifests librispeech-train-clean-100.json librispeech-train-clean-360.json librispeech-train-other-500.json \
  --val_manifests librispeech-dev-clean.json \
  --model_config configs/testing-1023sp_run.yaml \
  --num_gpus <NUM_GPU> \
  --global_batch_size 1008 \
  --grad_accumulation_batches 42 \
  --training_steps 42000
```

## Validation

The following command will run the validation script and calculate the WER \[%\].
See [here](validation.md#validation) for more details.

```bash
./scripts/val.sh
```
