# Training <a name="training"></a>

## Training Command

### Quick Start <a name="training_quick_start"></a>

This example demonstrates how to train a model on the LibriSpeech dataset using the `testing` model configuration.
This guide assumes that the user has followed the [installation guide](installation.md)
and has prepared LibriSpeech according to the [data preparation guide](json_format.md#librispeech_json).

Selecting the batch size arguments is based on the machine specifications.
More information on choosing them can be found [here](batch_size_hyperparameters.md).

Recommendations for LibriSpeech training are:

- a global batch size of 1008 for a 24GB GPU
- use all `train-*` subsets and validate on `dev-clean`
- 42000 steps is sufficient for 960hrs of train data
- adjust number of GPUs using the `--num_gpus=<NUM_GPU>` argument

To launch training inside the container, using a single GPU, run the following command:

```bash
./scripts/train.sh \
  --data_dir=/datasets/LibriSpeech \
  --train_manifests librispeech-train-clean-100.json librispeech-train-clean-360.json librispeech-train-other-500.json \
  --val_manifests librispeech-dev-clean.json \
  --model_config configs/testing-1023sp_run.yaml \
  --num_gpus 1 \
  --global_batch_size 1008 \
  --grad_accumulation_batches 42 \
  --training_steps 42000
```

The output of the training command is logged to `/results/training_log_[timestamp].txt`.
The arguments are logged to `/results/training_args_[timestamp].json`,
and the config file is saved to `/results/[config file name]_[timestamp].yaml`.

### Defaults to update for your own data

When training on your own data you will need to change the following args from their defaults to reflect your setup:

- `--data_dir`
- `--train_manifests`/`--train_tar_files`
  - To specify multiple training manifests, use `--train_manifests` followed by space-delimited file names, like this: `--train_manifests first.json second.json third.json`.
- `--val_manifests`/`--val_tar_files`/(`--val_audio_dir` + `--val_txt_dir`)
- `--model_config=configs/base-8703sp_run.yaml` (or the `_run.yaml` config file created by your `scripts/preprocess_<your dataset>.sh` script)

```admonish
The audio paths stored in manifests are **relative** with respect to `--data_dir`. For example,
if your audio file path is `train/1.flac` and the data_dir is `/datasets/LibriSpeech`, then the dataloader
will try to load audio from `/datasets/LibriSpeech/train/1.flac`.
```

The learning-rate scheduler argument defaults are tested on 1k-50k hrs of data but when training on larger
datasets than this you may need to tune the values. These arguments are:

1. `--warmup_steps`: number of steps over which learning rate is linearly increased from `--min_learning_rate`
2. `--hold_steps`: number of steps over which the learning rate is kept constant after warmup
3. `--half_life_steps`: the half life (in steps) for exponential learning rate decay

If you are using more than 50k hrs, it is recommended to start with `half_life_steps=10880` and increase if necessary. Note that increasing
`--half_life_steps` increases the probability of diverging later in training.

### Arguments

To resume training or fine tune a checkpoint see the documentation [here](./resuming_and_fine_tuning.md).

The default setup saves an overwriting checkpoint every time the Word Error Rate (WER) improves on the dev set.
Also, a non-overwriting checkpoint is saved at the end of training.
By default, checkpoints are saved every 5000 steps, and the frequency can be changed by setting `--save_frequency=N`.

For a complete set of arguments and their respective docstrings see
[`args/train.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/train.py)
and
[`args/shared.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/shared.py).

### Data Augmentation for Difficult Target Data

If you are targeting a production setting where background noise is common or audio arrives at 8kHZ,
see [here](challenging_target_data.md) for guidelines.

### Monitor training <a name="monitor_training"></a>

To view the progress of your training you can use TensorBoard.
See the [TensorBoard documentation](tensorboard.md) for more information of how to set up and use TensorBoard.

### Profiling <a name="profiling"></a>

To profile training, see these [instructions](profiling.md).

### Controlling emission latency <a name="emission_latency"></a>

See these [instructions](delay_penalty.md) on how to control emission latency of a model.

## Next Steps

Having trained a model:

- If you'd like to evaluate it on more test/validation data go to the [validation](./validation.md) docs.
- If you'd like to export a model checkpoint for inference go to the [hardware export](./export_inference_checkpoint.md) docs.

### See also

- [Supported dataset formats](supported_dataset_formats.md)
