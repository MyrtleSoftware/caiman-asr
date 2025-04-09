# Training times <a name="train-timings"></a>

Training times for v1.12.0 on an `8 x A100 (80GB)` system are as follows:

| Model   | Train-time (days) | Throughput (utt/s) | Throughput (s/s) | No. of updates | `grad_accumulation_batches` | `batch_split_factor` |
| ------- | ----------------- | ------------------ | ---------------- | -------------- | --------------------------- | -------------------- |
| `base`  | 0.9               | 1400               | 23,200           | 100k           | 1                           | 8                    |
| `large` | 1.8               | 700                | 11,700           | 100k           | 1                           | 16                   |

Training times for v1.12.0 on a `2 x RTX4090 (24GB)` system are as follows:

| Model   | Train-time (days) | Throughput (utt/s) | Throughput (s/s) | No. of updates | `grad_accumulation_batches` | `batch_split_factor` |
| ------- | ----------------- | ------------------ | ---------------- | -------------- | --------------------------- | -------------------- |
| `base`  | 8.4\*             | 150                | 2,500            | 100k           | 8                           | 8                    |
| `large` | 28\*              | 45                 | 750              | 100k           | 16                          | 8                    |

Training

where:

- **Throughput (s/s)** is the number of seconds of audio trained on per second (higher is better).
- **Throughput (utt/s)** is the number of samples/utterances seen per second during training (higher is better). **NOTE:** This metric is deprecated and will be removed in a future update, it is provided here for comparison.
- **No. of updates** is the number of optimiser steps taken at `--global_batch_size=1024`. You may need fewer/more steps depending on your dataset size.
- **`grad_accumulation_batches`** is the number of gradient accumulation steps performed on each GPU before taking an optimizer step
- **`batch_split_factor`** is the number of sub-batches that the `PER_GPU_BATCH_SIZE` is split into before these sub-batches are passed through the joint network and loss.
- Times appended with a '\*' are estimates from throughput scaling and extrapolation.

For more details on these hyper-parameters, including how to set them, please refer to the [batch size arguments](batch_size_hyperparameters.md) documentation. For some information about tuning DALI parameters see the [heterogeneous CPU](heterogeneous_cpu.md) page.
