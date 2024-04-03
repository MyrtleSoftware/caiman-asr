# ML training flow

This document describes the flow of training the `testing` model.
This configuration is used as an example as it is quicker to train than either `base` or `large`.

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

From inside the container, run the following command to download and extract LibriSpeech
(more details [here](json_format.md#prepare-librispeech-in-json-format)),
and preprocess it into the JSON format:

```bash
./scripts/download_librispeech.sh
```

```bash
SPM_SIZE=1023 CONFIG_NAME=testing-1023sp ./scripts/preprocess_librispeech.sh
```

After the datasets are ready, in order to train a `testing` model, run the following command.
A more detailed description of the training process can be found [here](training.md#training).


```bash
./scripts/train.sh \
  --data_dir /datasets/LibriSpeech \
  --train_manifests librispeech-train-clean-100-wav.json librispeech-train-clean-360-wav.json librispeech-train-other-500-wav.json \
  --val_manifests librispeech-dev-clean-wav.json \
  --model_config configs/testing-1023sp_run.yaml \
  --num_gpus 1 \
  --global_batch_size 1008 \
  --grad_accumulation_batches 42 \
  --training_steps 42000
```

After the training is finished, the model can be evaluated with the following command.
See [here](validation.md#validation) for more details.
```bash
./scripts/val.sh
```
