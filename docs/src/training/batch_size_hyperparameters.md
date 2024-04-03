# Batch size hyperparameters

If you are training on an `8 x A100 (80GB)` or `8 x A5000 (24GB)` machine, the recommended batch size hyper-parameters are given [here](training_times.md). Otherwise, this page gives guidance on how to select them. For a training command on `num_gpus` there are three command line args:

* `global_batch_size`
* `grad_accumulation_batches`
* `batch_split_factor`

The [Summary](#summary) section at the bottom of this page describes how to select them.
Before that, hyper-parameters and the motivation behind their selection are provided.

### `global_batch_size`

This is the batch size seen by the model before taking an optimizer step.

RNN-T models require large `global_batch_size`s in order to reach good WERs, but the larger the value, the longer training takes. The recommended value is `--global_batch_size=1024` and many of the defaults in the repository (e.g. learning rate schedule) assume this value.

### `grad_accumulation_batches`

This is the number of gradient accumulation steps performed on each GPU before taking an optimizer step. The actual `PER_GPU_BATCH_SIZE` is not controlled directly but can be calculated using the formula:

```
PER_GPU_BATCH_SIZE * grad_accumulation_batches * num_gpus = global_batch_size
```

The highest training throughput is achieved by using the highest `PER_GPU_BATCH_SIZE` (and lowest `grad_accumulation_batches`) possible without incurring an out-of-memory error (OOM) error.

Reducing `grad_accumulation_batches` will increase the training throughput but shouldn't have any affect on the WER.

### `batch_split_factor`

The joint network output is a 4-dimensional tensor that requires a large amount of GPU VRAM. For the models in this repo, the maximum `PER_GPU_JOINT_BATCH_SIZE` is much lower than the maximum `PER_GPU_BATCH_SIZE` that can be run through the encoder and prediction networks without incurring an OOM. When `PER_GPU_JOINT_BATCH_SIZE`=`PER_GPU_BATCH_SIZE`, the GPU will be underutilised during the encoder and prediction forward and backwards passes which is important because these networks constitute the majority of the training-time compute.

The `batch_split_factor` arg makes it possible to increase the `PER_GPU_BATCH_SIZE` whilst keeping the `PER_GPU_JOINT_BATCH_SIZE` constant where:

```
PER_GPU_BATCH_SIZE / batch_split_factor = PER_GPU_JOINT_BATCH_SIZE
```

Starting from the default `--batch_split_factor=1` it is usually possible to achieve higher throughputs by reducing`grad_accumulation_batches` and increasing `batch_split_factor` **while keeping their product constant**.

Like with `grad_accumulation_batches`, changing `batch_split_factor` should not impact the WER.

## Summary <a name="summary"></a>

In your training command it is recommended to:

1. Set `--global_batch_size=1024`
2. Find the smallest possible `grad_accumulation_batches` that will run without an OOM in the joint network or loss calculation
3. Then, progressively decrease `grad_accumulation_batches` and increase `batch_split_factor` keeping their product constant until you see an OOM in the encoder. Use the highest `batch_split_factor` that runs.

In order to test these, it is recommended to **use your full training dataset** as the utterance length distribution is important.
To check this quickly set `--n_utterances_only=10000` in order to sample 10k utterances randomly from your data,
and `--training_steps=20` in order to run 2 epochs (at the default `--global_batch_size=1024`).
When comparing throughputs it is better to compare the `avg train utts/s` from the second epoch as the first few iterations of the first epoch can be slow.

### Special case: OOM in step 3

There is some constant VRAM overhead attached to batch splitting so for some machines, when you try step 3. above you will see OOMs. In this case you should:

* Take the `grad_accumulation_batches` from step 2. and increase by *=2
* Then perform step 3.

In this case it's not a given that your highest throughput setup with `batch_split_factor` > 1 will be higher than the throughput from step 2. with `--batch_size-factor=1` so you should use whichever settings give a higher throughput.
