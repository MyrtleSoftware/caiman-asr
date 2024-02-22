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
* **p99 latency** is the 99th-percentile latency to process a single 60 ms audio frame and return any predictions. Note that latency increases with the number of concurrent streams.

The **solution scales linearly up to 8 accelerators and we have measured a single server supporting 16000 RTS** with the `base` model.

The `base` and `large` configurations are optimised for inference on FPGA with Myrtle's IP to achieve high-utilisation of the available resources. They were chosen after hyperparameter searches on 10k-50k hrs of training data.

<sup>§</sup>The `large` model inference performance figures are provisional.

### Word Error Rates (WERs)

When training on the 50k hrs of open-source data described below, the solution has the following WERs:

| Model             | MLS   | LibriSpeech-dev-clean | LibriSpeech-dev-other | Earnings21<sup>*</sup> |
|-------------------|-------|-----------------------|-----------------------|------------------------|
| `base`<sup>†</sup> | 9.37% |                 3.01% |                 8.14% |                 26.98% |
| `large`           | 7.70% |                 2.53% |                 6.90% |                 21.85% |

These WERs are for streaming scenarios without additional forward context. Both configurations have a frame size of 60ms, so, for a given segment of audio, the model sees between 0 and 60ms of future context before making predictions.

The 50k hrs of training data is a mixture of the following open-source datasets:

* LibriSpeech-960h
* Common Voice Corpus 10.0 (version `cv-corpus-10.0-2022-07-04`)
* Multilingual LibriSpeech (MLS)
* Peoples' Speech: filtered internally to take highest quality ~10k hrs out of 30k hrs total

This data has a `maximum_duration` of 20s and a mean length of 12.75s.

<sup>*</sup>None of these training data subsets include near-field unscripted utterances nor financial terminology. As such the Earnings21 benchmark is out-of-domain for these systems.
<sup>†</sup>`base` model WERs were not updated for the latest release. The provided values are from version [v1.6.1](https://github.com/MyrtleSoftware/myrtle-rnnt/releases/tag/v1.6.0).

### Training times <a name="train-timings"></a>

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

* **Throughput** is the number of utterances seen per second during training (higher is better)
* **No. of updates** is the number of optimiser steps at `--global_batch_size=1024` that are required to train the models on the 50k hrs training dataset. You may need fewer steps when training with less data
* **`grad_accumulation_batches`** is the number of gradient accumulation steps performed on each GPU before taking an optimizer step
* **`batch_split_factor`** is the number of sub-batches that the `PER_GPU_BATCH_SIZE` is split into before these sub-batches are passed through the joint network and loss.

For more details on these hyper-parameters, including how to set them, please refer to the [batch size arguments](training/docs/batch_size_hyperparameters.md) documentation.

To get started with training see the [training/README.md](training/README.md).
