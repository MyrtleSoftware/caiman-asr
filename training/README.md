# Myrtle.ai RNN-T PyTorch Training Script

# 0. Contents

1.  [Installation](#installation)
2.  [Models](#models)
3.  [Data](#data)
4.  [Training Commands](#training)
5.  [Validation Sets](#validation)
6.  [Validation on CPU](#valcpu)
7.  [Hardware Inference Server Support](#server)
8.  [Python Inference](#inference)
9.  [Acknowledgement](#ack)


# 1. Installation <a name="installation"></a>

These steps have been tested on Ubuntu 18.04 and 20.04.
Other Linux versions may work, since most processing takes place in a Docker container.
However, the install_docker.sh script is currently specific to Ubuntu.
Your machine does need NVIDIA GPU drivers installed.
Your machine does NOT need CUDA installed.

1. Clone the repository
```
git clone https://github.com/MyrtleSoftware/myrtle-rnnt.git
```
2. Install Docker
```
source training/install_docker.sh
```
3. Add your username to the docker group:
```
sudo usermod -a -G docker [user]
```
   Run the following in the same terminal window, and you might not have to log out and in again:
```
newgrp docker
```
4. Build the docker image
```
# Build from Dockerfile
cd training
./scripts/docker/build.sh
```

### Requirements

Currently, the reference uses CUDA-12.2 (see [Dockerfile](Dockerfile#L15)).
Here you can find a table listing compatible drivers: https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver

### Contributing

If you are planning to contribute to this repository, please follow the install steps in [CONTRIBUTING.md](./docs/CONTRIBUTING.md#dev_install).

### Start an interactive session in the Docker container to run data download, training and inference <a name="run-container"></a>

```
./scripts/docker/launch.sh <DATASETS> <CHECKPOINTS> <RESULTS>
```

Within the container, the contents of the `training` directory will be copied to the `/workspace/training` directory.
The container directories `/datasets`, `/checkpoints`, and `/results` are mounted as volumes
and mapped to the corresponding directories `<DATASETS>`, `<CHECKPOINTS>`, `<RESULTS>` on the host.

Note that the host directories passed to `./scripts/docker/launch.sh` must have absolute paths.  Checkpoints are saved to the  `/results` folder during training so it is sometimes convenient to load checkpoints from `/results` rather than from `/checkpoints`.

If your `<DATASETS>` folder contains symlinks to other drives (i.e. if your data is too large to fit on a single drive), they will not be accessible from within the running container. In this case, you can pass the absolute paths to your drives as the 4th, 5th, 6th, ... arguments to `./scripts/docker/launch.sh`. This will enable the container to follow symlinks to these drives.


# 2. Models <a name="models"></a>

Before training, you must select the model configuration you wish to train. Please refer to the [top-level README.md](../README.md#model-configs) for a description of the options available. Having selected a configuration it is necessary to note the config path and sentencepiece vocab size ("spm size") of your chosen config from the following table as these will be needed in the data preprocessing steps below:


|    Name   | Parameters | spm size |                        config                        | Acceleration supported?  |
|:---------:|:----------:|:--------:|:----------------------------------------------------:|--------------------------|
| `testing` | 49M        |     1023 | [testing-1023sp.yaml](./configs/testing-1023sp.yaml) | :heavy_multiplication_x: |
| `base`    | 85M        |     8703 | [base-8703sp.yaml](./configs/base-8703sp.yaml)       | :white_check_mark:       |
| `large`   | 196M       |    17407 | [large-17407sp.yaml](./configs/large-17407sp.yaml)   | :white_check_mark:       |

The `testing` config is not described in the [top-level README.md](../README.md#model-configs) as it is not supported on the accelerator. This configuration is included because it is quicker to train than either `base` or `large`. It is recommended to train the `testing` model on either LibriSpeech or CommonVoice as described below before training `base` or `large` on your own data.

The configs referenced above are not intended to be edited directly.  Instead, they are used as templates to create `<config-name>_run.yaml` files in the preprocessing section below.

# 3. Data <a name="data"></a>

This repository supports reading data from two formats. These are:

1. `json`: All audio as wav (or flac) files in a single directory hierarchy with transcripts in [json file(s)](examples/example.json) referencing these audio files.
2. `webdataset`: Audio `<key>.{flac,wav}` files stored with associated `<key>.txt` transcripts in tar file shards. Format described [here](https://github.com/webdataset/webdataset#the-webdataset-format).

In this README there are instructions for how to download and preprocess data in the `json` format. To use the `webdataset` format see the [WebDataset README](./docs/WebDataset.md). `json` examples are provided for the following datasets:

* LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12)
* CommonVoice [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)

To train on your own proprietary dataset you will need to arrange for it to be in the same format produced
by the data download and preprocessing scripts below.

## 3.1 Data Download

No GPU is required for data download and preprocessing. Therefore, if GPU usage is a limited resource, launch
the container for this section on a CPU-only machine via `./scripts/docker/launch.sh` as described above.

Note: Downloading and preprocessing LibriSpeech and CommonVoice requires up to 1TB of free disk space and can take several
hours to complete.

#### LibriSpeech

LibriSpeech contains 1000 hours of 16kHz read English speech derived from public domain audiobooks from
LibriVox project and has been carefully segmented and aligned. For more information, see the [LIBRISPEECH: AN
ASR CORPUS BASED ON PUBLIC DOMAIN AUDIO BOOKS](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) paper.

Inside the container, download and extract the datasets into the required format for later training and
inference:

```bash
./scripts/download_librispeech.sh
```
Once the data download is complete, the following folders should exist inside the container:

* `/datasets/LibriSpeech/`
   * `train-clean-100/`
   * `train-clean-360/`
   * `train-other-500/`
   * `dev-clean/`
   * `dev-other/`
   * `test-clean/`
   * `test-other/`

Since `/datasets/` is mounted to `<DATASETS>` on the host, once the dataset is downloaded it will be accessible
from outside of the container at `<DATASETS>/LibriSpeech`.

#### CommonVoice

CommonVoice is a Mozilla project to collect an open source, multi-language dataset of voices that anyone
can use to train speech-enabled applications.  For more information see the
[CommonVoice website](https://commonvoice.mozilla.org/en).

From the [website](https://commonvoice.mozilla.org/en/datasets) download the English Language Common Voice
Corpus Version 10.0 dataset.  If a different Version is used, the scripts below will need modification.

After downloading and untarring the following files and folders should exist on the host:

* `<DATASETS>/CommonVoice/cv-corpus-10.0-2022-07-04/en`
   * `clips/`
   * `dev.tsv`
   * `invalidated.tsv`
   * `other.tsv`
   * `reported.tsv`
   * `test.tsv`
   * `train.tsv`
   * `validated.tsv`

Since `/datasets/` is mounted to `<DATASETS>` on the host, once the dataset is downloaded it will be accessible
from inside the container at `/datasets/CommonVoice`.

## 3.2 Data Preprocessing <a name="data_preprocess"></a>

Note that while the scripts below prepare wav files, the NVIDIA DALI code that loads the speech can also
load flac and ogg files.  In our experiments we have found training on flac files to be just as fast as
training on wav files (both read from SSD) and we will be using flac files in the future to save space.

#### Note on text normalization

The examples below assume a character set of size 28: a space, an apostrophe and 26 lower case letters. The preprocessing scripts
and model configs assume this too. As such, these example scripts **will lowercase and remove punctuation from validation data** as
well as training data meaning that dev-set WERs may be lower that they might be for the full character set.
If transcripts aren't normalized during this preprocessing stage, it is possible to normalize them on the fly during training **and validation** with the [_clean_text](./rnnt_train/common/text/ito/__init__.py) function by setting `normalize_transcripts: true` in your config. By default we set `normalize_transcripts: false`.

#### LibriSpeech

Next, convert the data into WAV files:
```bash
./scripts/preprocess_librispeech.sh
```
Once the data is converted, the following additional files and folders should exist:
* `/datasets/LibriSpeech/`
   * `librispeech-train-clean-100-wav.json`
   * `librispeech-train-clean-360-wav.json`
   * `librispeech-train-other-500-wav.json`
   * `librispeech-dev-clean-wav.json`
   * `librispeech-dev-other-wav.json`
   * `librispeech-test-clean-wav.json`
   * `librispeech-test-other-wav.json`
   * `train-clean-100-wav/`
   * `train-clean-360-wav/`
   * `train-other-500-wav/`
   * `dev-clean-wav/`
   * `dev-other-wav/`
   * `test-clean-wav/`
   * `test-other-wav/`

For training, the following manifest files are used:
   * `librispeech-train-clean-100-wav.json`
   * `librispeech-train-clean-360-wav.json`
   * `librispeech-train-other-500-wav.json`

For evaluation, the `librispeech-dev-clean-wav.json` is used.

The above script also creates the sentencepiece model in /datasets/sentencepieces/ and a
config file `configs/testing-1023sp_run.yaml` that uses this model and data. This `_run.yaml` config is
derived from the generic `configs/testing-1023sp.yaml` file.

#### CommonVoice

The CommonVoice Version 10.0 clips/ directory contains mp3 files at a mixture of sample rates, including
48000Hz and 32000Hz.  Since we are currently interested in building RNN-T systems at 16kHz we resample to
16kHz during decompression to reduce subsequent disk usage, access times and processing times.

To decompress the mp3 files into 16kHz 16-bit mono wav files, inside the container run
```
./scripts/decompress_commonvoice.sh
```
This took several days on our system; you may wish to parallelize.
The script removes one defective mp3 file containing invalid data and listed in invalidated.tsv.


To convert CommonVoice .tsv files into RNN-T .json files, inside the container run
```
./scripts/preprocess_commonvoice.sh
```

Once the data is converted, the following additional files and folders should exist:

* `/datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en`
   * `wav_clips/`
   * `dev.json`
   * `test.json`
   * `train.json`

For training, the `train.json` manifest file is used.
For evaluation, the `dev.json` manifest file is used.

The above script also creates the sentencepiece model in /datasets/sentencepieces/ and adjusts the
`configs/testing-1023sp_run.yaml` file to use this model and data.

#### Other Datasets

To train on your own data, you should adapt the steps in the LibriSpeech preprocessing pipeline
[`scripts/preprocess_librispeech.sh`](scripts/preprocess_librispeech.sh). As mentioned above, we
recommend you train the `base` or `large` model rather than the `testing` model.
This means that in your version of [`create_sentencepieces.sh`](scripts/create_sentencepieces.sh) you will need to set `SPM_SIZE=8703`
and `CONFIG_NAME=base-8703sp` for the `base` configuration and `SPM_SIZE=17407` and `CONFIG_NAME=large-17407sp` for the `large` configuration.

It is advised that you use all of your training data transcripts to build
the sentencepiece tokenizer.

Depending on the maximum length of your utterances you may also need to edit the `MAX_DURATION_SECS` variable.

After running your version of `scripts/preprocess_<your dataset>.sh` you should have:

* json manifest files for each data subset
* sentencepiece `.model` and `.vocab` files in `/datasets/sentencepieces/`
* a newly created config file in `configs` with the suffix `_run.yaml` that is populated with your sentencepiece model path

# 4. Training Commands <a name="training"></a>


RNN-T trains use large synthetic batches specified with the GLOBAL_BATCH_SIZE variable.  These large synthetic batches are split between the NUM_GPUS where each gpu accumulates gradients over GRAD_ACCUMULATION_BATCHES until a single update is applied to the model (one "step").  The actual `batch_size`
fed to each GPU for each computation is not controlled directly but can be calculated using the formula

```
batch_size * GRAD_ACCUMULATION_BATCHES * NUM_GPUS = GLOBAL_BATCH_SIZE
```

In order to achieve the highest throughput during training, we recommend that you use the highest `batch_size` possible without incurring an OOM error. Please see [../README.md#train-times](../README.md#train-times) for the recommended batch size hyperparameters on `8 x A100 (80GB)` and `8 x A5000 (24GB)` machines.

The recommended target is a GLOBAL_BATCH_SIZE of 1024 when your data has a length distribution similar to LibriSpeech. For discussions of how to select a target GLOBAL_BATCH_SIZE, please refer to the discussions in CommonVoice's 'Training Commands' section below.
It is not always possible to achieve your target GLOBAL_BATCH_SIZE due to the memory constraints of the GPU and the GRAD_ACCUMULATION_BATCHES needed at a given batch_size. In this case, you may settle on a value slightly higher or lower than your target. For example, when training the `testing` model on LibriSpeech using a 2x 24GB TITAN RTX as described below, we use GLOBAL_BATCH_SIZE=1008.

Our step control recommendations for the learning rate scheduler assume a global batch size of ~1024.

## 4.1 Example Training Commands

#### LibriSpeech

Before running any training commands be sure that your `_run.yaml` config file reflects your sentencepiece model and data.
If you did not run the scripts above this can be achieved by running:

```
cat configs/testing-1023sp.yaml | sed s/SENTENCEPIECE/librispeech1023/g | sed s/MAX_DURATION/16.7/g > configs/testing-1023sp_run.yaml
```

The `testing` model can be run on LibriSpeech at batch_size of 24 without incurring an out-of-memory
error.  On a x2 TITAN RTX system we select this by setting GLOBAL_BATCH_SIZE=1008, GRAD_ACCUMULATION_BATCHES=21 and
NUM_GPUS=2.  16GB V100s can support a batch_size of 16, and 40GB A100s can
support a batch_size of 40.

The default setup saves an overwriting checkpoint every time dev Word Error Rate (WER) improves and a non-overwriting
checkpoint at the end of training. You can set `SAVE_AT_THE_END=false` to disable the final checkpoint save.
Additionally, if you would like to save a checkpoint every Nth epoch, set `SAVE_FREQUENCY=N`.

The default number of epochs to train for is 100.
Set, for example, EPOCHS=150 to make a different choice.

So, on two 24GB TITAN RTX GPUs, training LibriSpeech, run the following inside the container:

```
NUM_GPUS=2 GLOBAL_BATCH_SIZE=1008 GRAD_ACCUMULATION_BATCHES=21 EPOCHS=150 ./scripts/train.sh
```

The output of the training command is logged to `/results/training_log_[a timestamp].txt`.
Arguments are logged to `/results/training_args_[a timestamp].json`.
The config file is saved to `/results/[config file name]_[a timestamp].txt`.

To resume training see the [`RESUME=true` docs.](docs/resume_finetune.md).

#### CommonVoice

Before running any training commands be sure that your `_run.yaml` config file reflects your sentencepiece model and data.
If you did not run the scripts above this can be achieved by running:

```
cat configs/testing-1023sp.yaml | sed s/SENTENCEPIECE/commonvoice1023/g | sed s/MAX_DURATION/7.75/g > configs/testing-1023sp_run.yaml
```

CommonVoice utterances are shorter than LibriSpeech utterances, averaging 5.7s in duration compared to 12.3s.
Experience suggests that reducing the value of max_duration in the config file from the LibriSpeech value
of 16.7s to 7.75s (maintaining the max/mean duration ratio at 1.36) helps maintain gradient quality during
training.  High quality gradients are required for the model to learn well.  Experience also suggests that
the amount of speech per model update implicit in the above LibriSpeech command, roughly 12,500 seconds, or
3.5 hours, of speech per update, also helps maintain gradient quality; see below.

In principle the shorter max_duration means we ought to be able to increase batch_size with CommonVoice to
take advantage of all available GPU memory and maximize throughput.  In our experiments we have found
this triggers the freezing pathology described in the Training Examples section below, and so thus far
on a 24GB RTX we have used batch_size 24 as in the LibriSpeech case.

We therefore currently scale GRAD_ACCUMULATION_BATCHES by a factor of 12.3/5.7 to 45, and set
GLOBAL_BATCH_SIZE to GRAD_ACCUMULATION_BATCHES * NUM_GPUS * batch_size, which is 2160, to get
approximately 3.5 hours of speech per update.

For CommonVoice training, also on two 24GB TITAN RTX GPUs, we therefore run:

```
NUM_GPUS=2 GLOBAL_BATCH_SIZE=2160 GRAD_ACCUMULATION_BATCHES=45 EPOCHS=150 DATA_DIR=/datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en TRAIN_MANIFESTS=train.json VAL_MANIFESTS=dev.json ./scripts/train.sh
```

The max duration of 7.75s means this model is trained on approximately 1208 hrs of data.


#### Other Datasets

Depending on the amount of training data you are using, you may need to update the learning rate schedule. This is controlled via the following args:

1. `HALF_LIFE_STEPS`: the half life (in steps) for exponential learning rate decay
2. `WARMUP_STEPS`: number of steps over which learning rate is linearly increased from `MIN_LEARNING_RATE`
3. `HOLD_STEPS`: number of steps over which we hold the learning rate constant after warmup

For training experiments with 1k-3k hrs of data we find that it is best to keep `WARMUP_STEPS` and `HOLD_STEPS` at their default values.
The best value of `HALF_LIFE_STEPS` will change depending on the amount of data you are using. Specifically, we
see that for 1-3k hrs of training data `HALF_LIFE_STEPS=2805` works well whereas for 10k hrs we use a value of `10880`.
If you are using more than 10k hrs we recommend starting with `10880` and increasing from there. Note that increasing
`HALF_LIFE_STEPS` increases the probability of diverging late in training.

Alongside the normal training args like `EPOCHS`, etc you will also need to pass the following to `./scripts/train.sh`:

* `DATA_DIR`
* `TRAIN_MANIFESTS`
* `VAL_MANIFESTS`
* `MODEL_CONFIG=configs/base-8703sp_run.yaml` (or the `_run.yaml` config file created by your `scripts/preprocess_<your dataset>.sh` script)
* `HALF_LIFE_STEPS`

## 4.2 TensorBoard

The training scripts write TensorBoard logs to /results during training.

To monitor training using TensorBoard, launch the port-forwarding TensorBoard container in another terminal:

```
./scripts/docker/launch_tb.sh <RESULTS> <OPTIONAL PORT NUMBER>
```

If `<OPTIONAL PORT NUMBER>` isn't passed then we default to port 6010.

Then navigate to `http://traininghostname:<OPTIONAL PORT NUMBER>` in a web browser.

If a connection dies and you can't reconnect to your port because it's already allocated, run:

```
docker ps
docker stop <name of docker container with port forwarding>
```

## 4.3 Training Examples

TensorBoard training curve examples are given in directory [examples](examples/).

In [examples/success/LibriSpeech](examples/success/LibriSpeech) we present learning rate,
loss and word-error-rate curves for a successful train on the LibriSpeech dataset.

In [examples/success/CommonVoice](examples/success/CommonVoice) we present learning rate, grad-norm,
loss and word-error-rate curves for a successful mixed-precision train on the CommonVoice dataset.

In [examples/failure](examples/failure/) we include grad-norm, loss and word-error-rate
curves for a pathological CommonVoice training case in which grad-norm blew up to infinity and then went
to NaN.  This pathology is caused by exploding gradients in the RNN-T model encoder.  This pathology
affected roughly 1 in 10 LibriSpeech trains.  Training with larger GLOBAL_BATCH_SIZE values and smaller
max/mean duration ratios both appear to make this pathology less likely to occur.  If this pathology does
occur, resuming training from the last-saved good checkpoint is sometimes successful.

Another pathology we see is training runs freezing, with one GPU going to 0% utilization and the other to
100% utilization, with one Python process dying, but no error messages.  In this case reducing the batch
size usually avoids the problem.  We hope to investigate this pathology further in the future.

## 4.4 Mixed Precision Training <a name="tr_amp"></a>

By default we enable PyTorch mixed-precision-training with training arg `AMP=true`. To disable it set `AMP=false`.
We previously benchmarked NVIDIAs Apex mixed-precision-training as approximately 1.85x faster on the
TITAN RTXs, 1.61x faster on V100s, but only about 1.1x faster on A100s (which operate at 19 bits by
default); we expect the PyTorch version results to be similar.  We switched to the PyTorch
version after noting that models trained with the Apex version took longer to perform inference.
This problem does not occur with PyTorch mixed-precision trained models.

## 4.5 Large Tokenizer Training <a name="large_tokenizers"></a>

Training the RNN-T system with a large tokenizer (i.e. bigger than 1023 tokens) often results in exploding gradients
at 20-30k training steps.  This appears to be caused by the output matrix weights learning to be very small over time,
to the point that updates applied to this matrix using the same learning rate used for the rest of the model become too
large, leading to poor weights and gradient explosions during backprop.  This problem can be avoided by setting a custom
learning rate factor for this output matrix in the `rnnt` section of the model config file.  Experience has shown that a
good value is `1/sqrt(tokenizer_size/1024)`.  For example, to train a model with a tokenizer with 8703 tokens, set

```
joint_net_lr_factor: 0.343
```

Note that this is already set for the 85M parameter model with tokenizer size 8703 in `configs/base-8703sp.yaml`, as well as for the 196M parameter model with tokenizer size 17407 in `configs/large-17407sp.yaml`.

## 4.6 Random State Passing

RNN-Ts can find it difficult to generalise to sequences longer than those seen during training, as described in [Chiu et al, 2020](https://arxiv.org/abs/2005.03271).

To fix this, we implemented Random State Passing (RSP) as in [Narayanan et al., 2019](https://arxiv.org/abs/1910.11455).

On our in-house validation data, RSP reduces WERs on long (~1 hour) utterances by roughly 40% relative.

Experiments indicated:
- It was better to apply RSP 1% of the time, instead of 50% as in the paper.
- Applying RSP from the beginning of training raised WERs, so RSP is only applied after `RSP_DELAY` steps
  - `RSP_DELAY` can be set on the command line but, by default, is set to the step at which the learning rate has decayed to 1/8 of its initial value (i.e. after x3 `HALF_LIFE_STEPS` have elapsed). To see the benefits from RSP we find that we need >=5k updates after this point so this heuristic will not be appropriate if you intend to cancel training much sooner than this. See [docstring of `set_rsp_delay_default` function](rnnt_train/common/rsp.py) for more details.

RSP is on by default, and can be modified via the `RSP_SEQ_LEN_FREQ` argument, e.g. `RSP_SEQ_LEN_FREQ="99 0 1"`.
This parameter controls RSP's frequency and amount; see the `--rsp_seq_len_freq` docstring in [`train.py`](./rnnt_train/train.py).


RSP requires Myrtle.ai's [CustomLSTM](./docs/custom_lstm.md) which is why `custom_lstm: true` is set by default in the yaml configs.

## 4.7 Gradient Noise <a name="grad_noise"></a>

Adding Gaussian noise to the gradients of the network is a way of assisting the model generalize on out-of-domain datasets
by not over-fitting on the datasets it is trained on. Inspired by the research paper by
[Neelakantan et. al.](https://openreview.net/pdf?id=rkjZ2Pcxe),
we are sampling the noise level from a Gaussian distribution with $mean=0.0$ and
standard deviation that is time-dependent. The standard deviation is decaying with time following the
formula:

$\sigma(t)=\frac{noise\textunderscore level}{{(1+t-start\textunderscore step)}^{decay\textunderscore const}}$,

where $noise\textunderscore level$ is the level of noise when the gradient noise is switched on,
$decay\textunderscore const=0.55$ is the time decaying constant, $t$ is the step, and
$start\textunderscore step$ is the step when the gradient noise is switched on.

Training with noise is switched off by default.
It can be switched on by setting the noise level $noise\textunderscore level$ to be a positive value in the config file.

According to our experiments, the best time to switch on the gradient noise is after the warm-up period
(i.e. after the `WARMUP_STEPS` in the `scripts/train.sh` file). Moreover, the noise is only added in the gradients of
the encoder components, hence if during training the user chooses to freeze the encoder, adding grad noise will be off
by default.

# 5. Validation Sets <a name="validation"></a>

Validation is performed throughout training.  However, we can also validate models after training.
During training the maximum number of tokens that can be decoded per utterance is capped by the `MAX_SYMBOL_PER_SAMPLE` arg
which defaults to 300 (about a minute of speech). This is necessary since untrained models may ramble on indefinitely during decoding.
However, during validation `MAX_SYMBOL_PER_SAMPLE` is not set so no maximum decoding length is applied. It is
therefore possible that the same dev set may yield different word error rates in training vs after training.

The default evaluation command validates the best `testing-1023sp_run` model checkpoint saved during training (to be found in
/results/RNN-T_best_checkpoint.pt) against the LibriSpeech dev-clean partition.  Inside the container run

```
NUM_GPUS=2 ./scripts/val.sh
```

To validate against the noisy-speech dev-other partition, run

```
NUM_GPUS=2 VAL_MANIFESTS=/datasets/LibriSpeech/librispeech-dev-other-wav.json ./scripts/val.sh
```

To validate against the CommonVoice dev set, run

```
NUM_GPUS=2 DATA_DIR=/datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en VAL_MANIFESTS=dev.json ./scripts/val.sh
```

Validation can be performed using other checkpoints by specifying the CHECKPOINT variable.
Validation can be performed using other model config files by specifying the MODEL_CONFIG variable.

## 5.1 Long utterances <a name="long_utterances"></a>

Besides the WER, `val.sh` also calculates the transducer loss on the dev set.
This calculation may cause the GPU to run out of memory on
very long utterances. To skip the loss calculation and only
calculate WER, run
```
NO_LOSS=true NUM_GPUS=2 ./scripts/val.sh
```

# 6. Validation on CPU <a name="valcpu"></a>

It is also possible to validate models using CPU only.  The CPU version has additional functionality not found
in the GPU version, some of which is described in the Sections below.

To validate the `testing-1023sp_run` model against the LibriSpeech dev-clean partition, run:

```
./scripts/valCPU.sh
```

As above, the MODEL_CONFIG, VAL_MANIFESTS, DATA_DIR and CHECKPOINT variables allow you to specify other options.

# 7. Hardware Inference Server Support <a name="server"></a>

To run your model on Myrtle's hardware-accelerated inference server you will need to dump mel statistics from your dataset to
support streaming normalization and create a hardware checkpoint to enable transfer of this and other data.

### Streaming Normalization

Audio processing is carried out during training using NVIDIA's DALI in a Python environment.  For the
hardware-accelerated inference server Myrtle has implemented most of the same algorithms in Rust.

Feature normalization is an exception since the DALI algorithm computes the mean and standard deviation used
to normalize each mel-bin over the full length of each utterance, which is incompatible with streaming.  An
adaptive streaming normalization algorithm is implemented in [./rnnt_train/common/stream_norm.py](./rnnt_train/common/stream_norm.py)
which can be run using valCPU.\* as shown below.

The adaptive streaming normalization algorithm uses exponentially weighted moving averages and variances to
perform normalization of each new frame of mel-bin values.  This requires an initial set of mel-bin
mean and variance values which can be obtained by running the training script with
DUMP_MEL_STATS set to true with NUM_GPUS=1:

```
DUMP_MEL_STATS=true NUM_GPUS=1 GLOBAL_BATCH_SIZE=1008 GRAD_ACCUMULATION_BATCHES=21 EPOCHS=1 ./scripts/train.sh
```

The training data statistics are written to `<RESULTS>` as melmeans.\* and melvars.\* as both PyTorch
tensors and Numpy arrays.  These arrays can be transferred to the Rust inference server code and read by
valCPU.py for use by the adaptive streaming normalization algorithm written in Python:

```
STREAM_NORM=true ./scripts/valCPU.sh
```

The exponential weighting uses alpha to weight new values and 1-alpha to weight old values.
The default value of alpha is 0.001 but this can be changed using the ALPHA command line variable.

### Streaming Normalization Resets

The default behaviour of valCPU.py when applying streaming normalization is to update its internal statistics
with every frame of every utterance in the manifest being evaluated.  This is significant because, for example,
LibriSpeech dev-clean contains roughly 8 minutes of speech from each of 40 speakers, and the 8 minutes from
each speaker is in consecutive utterances.  The adaptive streaming normalization algorithm will therefore
adapt, in a few seconds depending on the value of alpha, to each speaker in turn, and then have a period of
stability before encountering the next speaker.

The Rust inference server code initializes each new Channel it creates using the training data statistics.
Therefore, if a new Channel is created for each utterance when evaluating on dev-clean, each Channel will
have to adapt from the training data initial statistics to the current speaker over just one utterance,
every time.  This difference from the Python system's default behaviour results in different reported
Word Error Rates.

To make the Python system behave in the same way as the Rust system, that is, to reset the streaming
normalization statistics to the training data statistics for each utterance in the manifest, set the
RESET_STREAM_STATS command line variable

```
STREAM_NORM=true RESET_STREAM_STATS=true ./scripts/valCPU.sh
```

In practical hardware systems it would be advantageous to arrange for multiple utterances from the same speaker
to be sent to the same Channel so that any speaker adaptation is retained and reused to minimize Word Error
Rate.

### Hardware Checkpoints

To transfer a trained model to Myrtle's hardware-accelerated inference server create a hardware checkpoint.

Inside the container run:

```
python ./rnnt_train/utils/hardware_ckpt.py \
    --ckpt /results/RNN-T_best_checkpoint.pt \
    --config <path/to/config.yaml> \
    --melmeans /results/melmeans.pt \
    --melvars /results/melvars.pt \
    --melalpha 0.001 \
    --output_ckpt /results/hardware_ckpt.pt
```

where `/results/RNN-T_best_checkpoint.pt` is your best checkpoint.
The hardware checkpoint also contains the sentencepiece model specified in the config file, the dataset mel
means and variances as dumped above, and the specified alpha value for mel statistics decay in streaming
normalization.

This checkpoint will load into val.py and valCPU.py with "EMA" warnings that can be ignored.


# 8. Python Inference <a name="inference"></a>

To dump the predicted text for a list of input wav files set the DUMP_PREDS variable and call valCPU.sh:

```
DUMP_PREDS=true VAL_MANIFESTS=/results/your-inference-list.json ./scripts/valCPU.sh
```

Predicted text will be written to /results/preds.txt

DUMP_PREDS can be used whether or not there are ground-truth transcripts in the json file.  If there are
then the word error rate reported by valCPU will be accurate, if not then it will be nonsense and should
be ignored.  The minimal json file for inference (with 2 wav files) looks like this:

```
[
  {
    "transcript": "dummy",
    "files": [
      {
        "fname": "relative-path/to/stem1.wav"
      }
    ],
    "original_duration": 0.0
  },
  {
    "transcript": "dummy",
    "files": [
      {
        "fname": "relative-path/to/stem2.wav"
      }
    ],
    "original_duration": 0.0
  }
]
```

where "dummy" can be replaced by the ground-truth transcript for accurate word error rate calculation,
where the filenames are relative to the $DATA_DIR variable fed to (or defaulted to by) valCPU.sh, and where
the original_duration values are effectively ignored (compared to infinity) but must be present.
Predictions can be generated using other checkpoints by specifying the CHECKPOINT variable.



# 9. Acknowledgement <a name="ack"></a>

This repository is based on the [MLCommons MLPerf RNN-T training benchmark.](https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch)
