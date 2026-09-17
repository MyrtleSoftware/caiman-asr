# Training <a name="training"></a>

## Training Command

### Quick Start <a name="training_quick_start"></a>

This example demonstrates how to train a model on the LibriSpeech dataset using the `base` model configuration.
This guide assumes that the user has followed the [installation guide](installation.md)
and has prepared LibriSpeech according to the [data preparation guide](jsonl_format.md#librispeech_json).

Selecting the batch size arguments is based on the machine specifications.
More information on choosing them can be found [here](batch_size_hyperparameters.md).

Recommendations for LibriSpeech training are:

- a global batch duration of ~20k seconds for a 24GB GPU
- use all `train-*` subsets and validate on `dev-clean`
- 42000 steps is sufficient for 960hrs of train data
- adjust number of GPUs using the `--num_gpus=<NUM_GPU>` argument

To launch training inside the container, using a single GPU, run the following command:

```bash
./scripts/train.sh \
  --train_dataset_yaml ./configs/librispeech.yaml \
  --val_manifests librispeech-dev-clean.cuts.jsonl.gz \
  --val_dataset_dir /datasets/LibriSpeech \
  --model_config ./configs/base-8703sp_run.yaml \
  --num_gpus 1 \
  --batch_duration 1800 \
  --grad_accumulation_batches 10 \
  --val_batch_size 1 \
  --training_steps 42000
```

The output of the training command is logged to `/results/training_log_[timestamp].txt`.
The arguments are logged to `/results/training_args_[timestamp].json`,
and the config file is saved to `/results/[config file name]_[timestamp].yaml`.

### Defaults to update for your own data

When training on your own data you will need to change the following args from their defaults to reflect your setup:

- `--train_dataset_yaml`
- `--val_data_dir`
- `--val_manifests`

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

### Controlling the proportion of data from each manifest <a name="controlling-the-proportion-of-data-from-each-manifest"></a>

If you would like to adapt the proportion of data that the model sees from each
manifest per batch you can adjust the weights in the train dataset yaml, for
example:

```yaml
format: multilingual-jsonl
languages:
  lang_code:
    datasets:
      - /datasets/demo/hi_quality.cuts.jsonl.gz:
          weight: 1.0
      - /datasets/demo/lo_quality.cuts.jsonl.gz:
          weight: 1.0
```

This would have 50% of utterances from `hi_quality` and 50% from `lo_quality` in
each batch. This is useful if, for example, `lo_quality` is much larger than
`hi_quality` and more efficient than truncating `lo_quality` as the model would
then not see all the data.

When manifest balancing is on we use the word epoch to mean: _the minimum time
until any sample is seen again_. This parallels the definition used when
manifest balancing is off but relaxes the condition that all the data must be
seen.

#### Canary-manifest balancing

Setting the train manifest ratios can be a laborious task that requires much
experimentation, a sensible default can be obtained from the [canary
paper](https://arxiv.org/pdf/2406.19674v1):

$$
p_s \sim \left( \frac{n_s}{N} \right)^\alpha
$$

With, \\(p_s\\) the prob of sampling from the \\(x\\) manifest, \\(n_s\\) the
number of hours in the corresponding manifest and \\(x\\) the total number of
hours:

$$
N = \sum_s n_s
$$

We do manifest balancing in utterance space instead of time space, these are
related by a manifest dependent constant:

$$
n_s \approx k_s u_s
$$

With \\(u_s\\) the number of utterances in the \\(x\\) manifest. Hence if we
want the number of hours of each epoch to match the canary proportions we need:

$$
\begin{align}
r_s = \frac{p_s}{k_s}
 = \frac{u_s}{n_s}  \left( \frac{n_s}{\sum_i n_i} \right)^\alpha
\end{align}
$$

Where \\(r_s\\) is the manifest ratio needed in the `weight` field of the
dataset yaml. This is all computed automatically when the following is added to
the yaml file for datasets of a language:

```yaml
format: multilingual-shar
languages:
  lang_code:
    canary:
      alpha: 0.5
```

##### Multilingual manifest balancing

When training multilingual models it is often desirable to balance both within
and across languages. This can be achieved through hierarchical manifest
balancing. For example, consider the following dataset yaml:

```yaml
format: multilingual-shar
canary:
  alpha: 0.5
languages:
  bg:
    canary:
      alpha: 0.75
    datasets:
      - /shar/dataset/a:
      - /shar/dataset/b:
  da:
    canary:
      alpha: 0.5
    datasets:
      - /shar/dataset/c:
      - /shar/dataset/d:
```

This would balance within each language `bg` with `alpha=0.75` and `da` with
`alpha=0.5` to produce two sets of two weights. Then it would manifest balance
across the two languages `bg` and `da` (treating each as a single dataset) with
`alpha=0.5` to produce another set of two weights. The final weight for each
dataset is the product of the two weights.

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
