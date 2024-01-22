# RNN-Transducer Speech Recognition

This repository contains code for training a Recurrent Neural Network Transducer (RNN-T)
Automatic Speech Recognition (ASR) model designed to run on Myrtle.ai’s acceleration solutions for
FPGA hardware.  The solution is designed for low-latency realtime streaming ASR workloads.

Code and training instructions are in the [./training](training/README.md) directory.

The following section summarises the inference performance, word error rates and training times of the available model configurations.

## Model Configurations <a name="model-configs"></a>

The solution supports two model configurations:

| Model   | Parameters | Realtime streams (RTS) | p99 latency at max RTS    | p99 latency at RTS=32 |
|---------|------------|------------------------|---------------------------|-----------------------|
| `base`  | 85M        |          2000          |            25 ms          |          15 ms        |
| `large` | 196M       |     800<sup>§</sup>    |     50 ms<sup>§</sup>     |  20 ms<sup>§</sup>    |

where:

* **Realtime streams (RTS)** is the number of concurrent streams that can be serviced by a single accelerator
* **p99 latency** is the 99th-percentile latency to process a single 60 ms audio frame and return any predictions. Note that latency increases with more concurrent streams.

<sup>§</sup>The `large` model inference performance figures are provisional.

The **solution scales linearly with number of accelerators in the server** (tested up to 8000 RTS per server).

The `base` and `large` configurations are optimised for inference on FPGA with Myrtle's IP to achieve high-utilisation of the available resources. They were chosen after hyperparameter searches on 10k-50k hrs of training data.

### Word Error Rates (WERs)

When training on the 50k hrs of open-source data described below, the solution has the following WERs:

| Model   | MLS   | LibriSpeech-dev-clean | LibriSpeech-dev-other | Earnings21<sup>*</sup> |
|---------|-------|-----------------------|-----------------------|------------------------|
| `base`  | 9.37% |                 3.01% |                 8.14% |                 26.98% |
| `large` | 7.93% |                 2.69% |                 7.14% |                 23.33% |

These WERs are for streaming scenarios without additional forward context. Both configurations have a frame size of 60ms, so, for a given segment of audio, the model sees between 0 and 60ms of future context before making predictions.

The 50k hrs of training data is a mixture of the following open-source datasets:

* LibriSpeech-960h
* Common Voice Corpus 10.0 (version `cv-corpus-10.0-2022-07-04`)
* Multilingual LibriSpeech (MLS)
* Peoples' Speech: filtered internally to take highest quality ~10k hrs out of 30k hrs total

This data has a `maximum_duration` of 20s and a mean length of 12.75s.

**<sup>*</sup>** None of these training data subsets include near-field unscripted utterances nor financial terminology. As such the Earnings21 benchmark is out-of-domain for these systems.

### Training times <a name="train-timings"></a>

Training throughputs on an `8 x A100 (80GB)` system are as follows:

| Model   | Training time | Throughput  | No. of updates | per-gpu `batch_size` | `GRAD_ACCUMULATION_BATCHES` |
|---------|---------------|-------------|----------------|----------------------|-----------------------------|
| `base`  | 1.8 days      | 671 utt/sec | 100k           |                   32 |                           4 |
| `large` | 3.1 days      | 380 utt/sec | 100k           |                   16 |                           8 |

Training times on an `8 x A5000 (24GB)` system are as follows:

| Model   | Training time | Throughput  | No. of updates | per-gpu `batch_size` | `GRAD_ACCUMULATION_BATCHES` |
|---------|---------------|-------------|----------------|----------------------|-----------------------------|
| `base`  | 4.4 days      | 268 utt/sec | 100k           |                    8 |                          16 |
| `large` | 12.9 days     | 92 utt/sec  | 100k           |                    4 |                          32 |

where:

* **Throughput** is the number of utterances seen per second during training (higher is better)
* **No. of updates** is the number of optimiser steps at `GLOBAL_BATCH_SIZE=1024` that are required to train the models on the 50k hrs training dataset. You may need fewer steps when training with less data
* **`GRAD_ACCUMULATION_BATCHES`** is the number of gradient accumulation steps per gpu required to achieve the `GLOBAL_BATCH_SIZE` of 1024. For all configurations the **per-gpu `batch_size`** is as large as possible meaning that `GRAD_ACCUMULATION_BATCHES` is set as small as possible.

For more details on the batch size hyperparameters refer to the [Training Commands subsection of training/README.md](training/README.md#training). To get started with training see the [training/README.md](training/README.md).
