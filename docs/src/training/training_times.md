# Training times <a name="train-timings"></a>

Training throughputs on an `8 x A100 (80GB)` system are as follows:

| Model   | Training time | Throughput  | No. of updates | `grad_accumulation_batches` | `batch_split_factor` |
|---------|---------------|-------------|----------------|-----------------------------|----------------------|
| `base`  | 1.6 days      | 729 utt/sec | 100k           |                           1 |                    8 |
| `large` | 2.2 days      | 550 utt/sec | 100k           |                           1 |                   16 |

Training times on an `8 x A5000 (24GB)` system are as follows:

| Model   | Training time | Throughput  | No. of updates | `grad_accumulation_batches` | `batch_split_factor` |
|---------|---------------|-------------|----------------|-----------------------------|----------------------|
| `base`  | 3.1 days      | 379 utt/sec | 100k           |                           1 |                   16 |
| `large` | 8.5 days      | 140 utt/sec | 100k           |                           8 |                    4 |

where:

- **Throughput** is the number of utterances seen per second during training (higher is better)
- **No. of updates** is the number of optimiser steps at `--global_batch_size=1024` that are required to train the models on the 50k hrs training dataset. You may need fewer steps when training with less data
- **`grad_accumulation_batches`** is the number of gradient accumulation steps performed on each GPU before taking an optimizer step
- **`batch_split_factor`** is the number of sub-batches that the `PER_GPU_BATCH_SIZE` is split into before these sub-batches are passed through the joint network and loss.

For more details on these hyper-parameters, including how to set them, please refer to the [batch size arguments](batch_size_hyperparameters.md) documentation.
