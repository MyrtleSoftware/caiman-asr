# Myrtle.ai RNN-T PyTorch Training Script

**NOTE:** As of v1.8, the API of `scripts/train.sh` & `scripts/val.sh` have changed. These scripts now take command line arguments instead of environment variables (`--num_gpus=8` instead of `NUM_GPUS=8`).
For backwards compatibility, the scripts `scripts/legacy/train.sh` and `scripts/legacy/val.sh` still use the former API but these legacy scripts do not support features introduced after v1.7.1, and they will be removed in a future release.


# 0. Contents

1.  [Installation](#installation)
2.  [Models](#models)
3.  [Data](#data)
4.  [Training Commands](#training)
5.  [Validation Commands](#validation)
6.  [Hardware Inference Server Support](#server)
7.  [Python Inference](#inference)
8.  [Acknowledgement](#ack)


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

This repository supports reading data from three formats. These are:

1. `json`: All audio as wav (or flac) files in a single directory hierarchy with transcripts in [json file(s)](examples/example.json) referencing these audio files.
2. `webdataset`: Audio `<key>.{flac,wav}` files stored with associated `<key>.txt` transcripts in tar file shards. Format described [here](https://github.com/webdataset/webdataset#the-webdataset-format).
3. `directories`: Audio (wav or flac) files and the respective text transcripts are in two separate directories.

In this README there are instructions for how to download and preprocess data in the `json` format. To use the `webdataset` format see the [WebDataset README](./docs/WebDataset.md).
The `directories` format is supported only for validation, and more information can be found in the [validation on directories README](./docs/validation_on_directories.md).
`json` examples are provided for the following datasets:

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

#### Note on text normalization and Standardization

The examples below assume a character set of size 28: a space, an apostrophe and 26 lower case letters. The preprocessing scripts
and model configs assume this too. As such, these example scripts **will lowercase and remove punctuation from validation data** as
well as training data meaning that dev-set WERs may be lower that they might be for the full character set.
If transcripts aren't normalized during this preprocessing stage, they will be normalized
on the fly during training (and validation) with the [_clean_text](./rnnt_train/common/text/ito/__init__.py)
function. Normalization of transcripts can be turned off by setting `normalize_transcripts: false` in your config.

Note that we standardize all references and hypotheses with the Whisper normalizer before calculating WERs,
as described in the [WER calculation docs](./docs/wer_calculation.md).
To switch off standardization, modify the respective config file entry to read `standardize_wer: false`.


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

#### Preprocessing Other Datasets

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

Before starting training you must select your batch size hyper-parameters. Please see the [Batch size arguments](docs/batch_size_hyperparameters.md) documentation for this.

Note that the example training commands below for LibriSpeech and CommonVoice don't use the recommended `--global_batch_size=1024`. These examples will be changed to use the recommended values in a future release.

## 4.1 Example Training Commands

#### LibriSpeech

Before running any training commands be sure that your `_run.yaml` config file reflects your sentencepiece model and data.
If you did not run the scripts above this can be achieved by running:

```
cat configs/testing-1023sp.yaml | sed s/SENTENCEPIECE/librispeech1023/g | sed s/MAX_DURATION/16.7/g > configs/testing-1023sp_run.yaml
```

The default setup saves an overwriting checkpoint every time dev Word Error Rate (WER) improves and a non-overwriting
checkpoint at the end of training. You can pass `--dont_save_at_the_end` to disable the final checkpoint save.
Additionally, if you would like to save a checkpoint every Nth epoch, set `--save_frequency=N`.

The default number of epochs to train for is 100.
Set, for example, `--epochs=150` to make a different choice.

So, on two 24GB TITAN RTX GPUs, training LibriSpeech, run the following inside the container:

```
./scripts/train.sh --num_gpus=2 --global_batch_size=1008 --grad_accumulation_batches=21 --epochs=150
```

The output of the training command is logged to `/results/training_log_[a timestamp].txt`.
Arguments are logged to `/results/training_args_[a timestamp].json`.
The config file is saved to `/results/[config file name]_[a timestamp].txt`.

To resume training see the [`--resume` docs.](docs/resume_finetune.md).

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

In principle the shorter max_duration means we ought to be able to increase `PER_GPU_BATCH_SIZE` with CommonVoice to
take advantage of all available GPU memory and maximize throughput.  In our experiments we have found
this triggers the freezing pathology described in the Training Examples section below, and so thus far
on a 24GB RTX we have used `PER_GPU_BATCH_SIZE` 24 as in the LibriSpeech case.


We therefore currently scale `grad_accumulation_batches` by a factor of 12.3/5.7 to 45, and set
`global_batch_size` to `grad_accumulation_batches * num_gpus * PER_GPU_BATCH_SIZE`, which is 2160, to get
approximately 3.5 hours of speech per update.

For CommonVoice training, also on two 24GB TITAN RTX GPUs, we therefore run:

```
./scripts/train.sh --num_gpus=2 --global_batch_size=2160 --grad_accumulation_batches=45 --epochs=150 --data_dir=/datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en --train_manifests=train.json --val_manifests=dev.json
```

The max duration of 7.75s means this model is trained on approximately 1208 hrs of data.


#### Defaults to update for your own data

When training on your own data you will need to change the following args from their defaults to reflect your setup:

* `--data_dir`
* `--train_manifests`/`--train_tar_files`
    * The way to pass multiple `train_manifests` is `--train_manifests first.json second.json third.json`.
* `--val_manifests`/`--val_tar_files`/(`--val_audio_dir` + `--val_txt_dir`)
* `--model_config=configs/base-8703sp_run.yaml` (or the `_run.yaml` config file created by your `scripts/preprocess_<your dataset>.sh` script)

The learning-rate schedule arg defaults are tested on 1k-50k hrs of data but when training on larger datasets than this you may need to tune the values. These arguments are:

1. `--half_life_steps`: the half life (in steps) for exponential learning rate decay
2. `--warmup_steps`: number of steps over which learning rate is linearly increased from `--min_learning_rate`
3. `--hold_steps`: number of steps over which we hold the learning rate constant after warmup

If you are using more than 50k hrs we recommend starting with `half_life_steps=10880` and increasing from there. Note that increasing
`--half_life_steps` increases the probability of diverging late in training.


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
affected roughly 1 in 10 LibriSpeech trains.  Training with larger `global_batch_size` values and smaller
max/mean duration ratios both appear to make this pathology less likely to occur.  If this pathology does
occur, resuming training from the last-saved good checkpoint is sometimes successful.

Another pathology we see is training runs freezing, with one GPU going to 0% utilization and the other to
100% utilization, with one Python process dying, but no error messages.  In this case reducing the batch
size usually avoids the problem.  We hope to investigate this pathology further in the future.

## 4.4 Mixed Precision Training <a name="tr_amp"></a>

By default we enable PyTorch mixed-precision-training . To disable it, pass `--no_amp`.
We previously benchmarked NVIDIAs Apex mixed-precision-training as approximately 1.85x faster on the
TITAN RTXs, 1.61x faster on V100s, but only about 1.1x faster on A100s (which operate at 19 bits by
default); we expect the PyTorch version results to be similar.  We switched to the PyTorch
version after noting that models trained with the Apex version took longer to perform inference.
This problem does not occur with PyTorch mixed-precision trained models.

## 4.5 Noise Augmented Training <a name="tr_noise"></a>

We can apply two types of noise augmentation during training: background noise and babble noise. These are both applied sample-wise and are set via the `--prob_background_noise` and `--prob_babble_noise` arguments
respectively which must be in `[0.0, 1.0]`. These types of noise augmentation are applied independently so if
both have probabilities greater than 0.0 then some samples will have both augmentation types applied.

By default, `prob_background_noise` is `0.25` and `prob_babble_noise` is `0.0`.

On an `8 x A100 (80GB)` system, turning off background noise augmentation increases the base model's training throughput by ~17% and the large model's throughput by ~11%.

Babble is applied by taking other utterances from the same batch and mixing them with the speech whereas
background noise takes a non-speech noise file and mixes it with the speech.

The noise data is combined with speech data on-the-fly during training.  For each utterance in each batch a
signal to noise ratio (SNR) is randomly chosen between an internally held 'low' and 'high' value.  The
noise is adjusted to have this SNR relative to the speech, the noise and speech are combined, and the result is
scaled to have the same volume as the original speech signal.

Before combination, the noise audio will be duplicated to become at least as long as the speech utterance.

The initial values for 'low' and 'high' can be specified (in dB) using the `--noise_initial_low` and
`--noise_initial_high` arguments when calling `train.sh`.  This range is then maintained for the number of
steps specified by the `--noise_delay_steps` argument after which the noise level is ramped up over
`--noise_ramp_steps` to its final range. These arguments are shared between both types of noise.
The final range for background noise is 0-30dB (taken from the Google paper "Streaming
end-to-end speech recognition for mobile devices", [He et al., 2018](https://arxiv.org/abs/1811.06621)) while the final range of
babble noise is 15-30dB.

By default, background noise will use [Myrtle/CAIMAN-ASR-BackgroundNoise](https://huggingface.co/datasets/Myrtle/CAIMAN-ASR-BackgroundNoise) from the [Hugging Face Hub](https://huggingface.co/docs/hub/en/datasets-overview).

Note that this dataset will be cached in `~/.cache/huggingface/` in order to persist between containers.
You can change this location like so: `HF_CACHE=[path] ./scripts/docker/launch.sh ...`.

To change the default noise dataset, set `--noise_dataset` to an audio dataset on the Hugging Face Hub.
The training script will use all the audios in the noise dataset's `train` split.

If you instead wish to train with local noise files, make sure your noise is organized in the Hugging Face [AudioFolder](https://huggingface.co/docs/datasets/en/audio_dataset#audiofolder) format.
Then set `--noise_dataset` to be the path to the directory containing your noise data (i.e. the parent of the `data` directory), and pass `--use_noise_audio_folder`.

The following command will train the base model on the LibriSpeech dataset on an `8 x A100 (80GB)` system with these settings:
- applying background noise to 25% of samples
- applying babble noise to 10% of samples
- using the default noise schedule
  - initial values 30--60dB
  - noise delay of 4896 steps
  - noise ramp of 4896 steps

```bash
./scripts/train.sh --model_config=configs/base-8703sp_run.yaml --num_gpus=8 \
    --grad_accumulation_batches=4 --epochs=150 --prob_background_noise=0.25 \
    --prob_babble_noise=0.1 \
    --val_manifests=/datasets/LibriSpeech/librispeech-dev-other-wav.json
```

### Inspecting audio

To listen to the effects of noise augmentation, pass `--inspect_audio`. All audios will then be saved to `/results/augmented_audios` after augmentations have been applied. This is intended for debugging only---DALI is slower with this option, and a full epoch of saved audios will use as much disk space as the training dataset.

## 4.6 Large Tokenizer Training <a name="large_tokenizers"></a>

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

## 4.7 Random State Passing

RNN-Ts can find it difficult to generalise to sequences longer than those seen during training, as described in [Chiu et al, 2020](https://arxiv.org/abs/2005.03271).

To fix this, we implemented Random State Passing (RSP) as in [Narayanan et al., 2019](https://arxiv.org/abs/1910.11455).

On our in-house validation data, RSP reduces WERs on long (~1 hour) utterances by roughly 40% relative.

Experiments indicated:
- It was better to apply RSP 1% of the time, instead of 50% as in the paper.
- Applying RSP from the beginning of training raised WERs, so RSP is only applied after `--rsp_delay` steps
  - `--rsp_delay` can be set on the command line but, by default, is set to the step at which the learning rate has decayed to 1/8 of its initial value (i.e. after x3 `half_life_steps` have elapsed). To see the benefits from RSP we find that we need >=5k updates after this point so this heuristic will not be appropriate if you intend to cancel training much sooner than this. See [docstring of `set_rsp_delay_default` function](rnnt_train/common/rsp.py) for more details.

RSP is on by default, and can be modified via the `--rsp_seq_len_freq` argument, e.g. `--rsp_seq_len_freq 99 0 1`.
This parameter controls RSP's frequency and amount; see the `--rsp_seq_len_freq` docstring in [`train.py`](./rnnt_train/train.py).


RSP requires Myrtle.ai's [CustomLSTM](./docs/custom_lstm.md) which is why `custom_lstm: true` is set by default in the yaml configs.

## 4.8 Gradient Noise <a name="grad_noise"></a>

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
(i.e. after `warmup_steps`). Moreover, the noise is only added in the gradients of
the encoder components, hence if during training the user chooses to freeze the encoder, adding grad noise will be off
by default.

## 4.9 Narrowband training <a name="narrowband"></a>

For some target domains, data is recorded at (or compressed to) 8 kHz (narrowband). For models trained with audio >8 kHz (16 kHz is the default) the audio will be upsampled to the higher sample rate before inference. This creates a mismatch between training and inference, since the model will partly rely on information from the higher frequency bands.

This can be partly mitigated by resampling a part of the training data to narrowband and back to higher frequencies, so the model is trained on audio that more closely resembles the validation data.

To apply this downsampling on-the-fly to a random half of batches, set `--prob_train_narrowband=0.5` in your training command.

## 4.10 Profiling <a name="profiling"></a>

To profile training, see these [instructions](docs/profiling.md).

# 5. Validation Commands <a name="validation"></a>

Validation is performed throughout training.  However, we can also validate models after training.
During training the maximum number of tokens that can be decoded per utterance is capped by the `--max_symbol_per_sample` arg
which defaults to 300 (about a minute of speech). This is necessary since untrained models may ramble on indefinitely during decoding.
However, during validation `--max_symbol_per_sample` is not set so no maximum decoding length is applied. It is
therefore possible that the same dev set may yield different word error rates in training vs after training.

The default evaluation command validates the best `testing-1023sp_run` model checkpoint saved during training (to be found in
`/results/RNN-T_best_checkpoint.pt`) against the LibriSpeech dev-clean partition.  Inside the container run

```
./scripts/val.sh --num_gpus=2
```

To validate against the noisy-speech dev-other partition, run

```
./scripts/val.sh --num_gpus=2 --val_manifests=/datasets/LibriSpeech/librispeech-dev-other-wav.json
```

To validate against the CommonVoice dev set, run

```
./scripts/val.sh --num_gpus=2 --data_dir=/datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en --val_manifests=dev.json
```

Validation can be performed using other checkpoints by specifying the `--checkpoint` argument.
Validation can be performed using other model config files by specifying the `--model_config` argument.
It is possible to run validation on the CPU by passing the `--cpu` flag.

The output of the validation command is logged to `/results/validation_log_[a timestamp].txt`.

## 5.1 Transducer Loss <a name="transducer_loss"></a>

Besides the WER, `val.sh` can also calculate the transducer loss on the dev set.
This calculation may cause the GPU to run out of memory on
very long utterances. To perform the loss calculation, run
```
./scripts/val.sh --calculate_loss --num_gpus=2
```
Please note that the loss calculation is not available when using the `--cpu` flag.

# 6. Hardware Inference Server Support <a name="server"></a>

To run your model on Myrtle's hardware-accelerated inference server you will need to dump mel statistics from your dataset to
support streaming normalization and create a hardware checkpoint to enable transfer of this and other data.

### Streaming Normalization

Audio processing is carried out during training using NVIDIA's DALI in a Python environment.  For the
hardware-accelerated inference server Myrtle has implemented most of the same algorithms in Rust.

Feature normalization is an exception since the DALI algorithm computes the mean and standard deviation used
to normalize each mel-bin over the full length of each utterance, which is incompatible with streaming.  An
adaptive streaming normalization algorithm is implemented in [./rnnt_train/common/stream_norm.py](./rnnt_train/common/stream_norm.py)
which can be run using val.\* as shown below.

The adaptive streaming normalization algorithm uses exponentially weighted moving averages and variances to
perform normalization of each new frame of mel-bin values.  This requires an initial set of mel-bin
mean and variance values which can be obtained by running the training script with
`--dump_mel_stats` and `--num_gpus=1`:

```
./scripts/train.sh --dump_mel_stats --num_gpus=1 --global_batch_size=1008 --grad_accumulation_batches=21 --epochs=1
```

The training data statistics are written to `<RESULTS>` as melmeans.\* and melvars.\* as both PyTorch
tensors and Numpy arrays.  These arrays can be transferred to the Rust inference server code and read by
val.py for use by the adaptive streaming normalization algorithm written in Python:

```
./scripts/val.sh --stream_norm
```

The exponential weighting uses alpha to weight new values and 1-alpha to weight old values.
The default value of alpha is 0.001 but this can be changed using the `--alpha` command line argument.

### Streaming Normalization Resets

The default behaviour of val.py when applying streaming normalization is to reset its internal statistics to the training data statistics after every utterance.

This is significant because, for example,
LibriSpeech dev-clean contains roughly 8 minutes of speech from each of 40 speakers, and the 8 minutes from
each speaker is in consecutive utterances.
If this resetting didn't happen, then the adaptive streaming normalization algorithm would therefore
adapt, in a few seconds depending on the value of alpha, to each speaker in turn, and then have a period of
stability before encountering the next speaker.

Experiments have shown that not resetting the statistics leads to different Word Error Rates.

The Rust inference server code initializes each new Channel it creates using the training data statistics.
Therefore, if a new Channel is created for each utterance when evaluating on dev-clean, each Channel will
have to adapt from the training data initial statistics to the current speaker over just one utterance,
every time.

To make the Python system behave differently from the Rust system, that is, to keep updating the streaming
normalization statistics for each utterance in the manifest, pass the
`--dont_reset_stream_stats` command line argument:

```
./scripts/val.sh --stream_norm --dont_reset_stream_stats
```

In most cases you should not pass this argument, since you will usually want to match the behavior of the Rust system.

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

This checkpoint will load into val.py with "EMA" warnings that can be ignored.

### Initial padding

See [here](docs/initial_padding.md).

# 7. Python Inference <a name="inference"></a>

To dump the predicted text for a list of input wav files, pass the `--dump_preds` argument and call `val.sh`:

```
./scripts/val.sh --dump_preds --val_manifests=/results/your-inference-list.json
```

Predicted text will be written to `/results/preds.txt`

The argument `--dump_preds` can be used whether or not there are ground-truth transcripts in the json file.  If there are,
then the word error rate reported by val will be accurate; if not, then it will be nonsense and should
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
where the filenames are relative to the `--data_dir` argument fed to (or defaulted to by) `val.sh`, and where
the original_duration values are effectively ignored (compared to infinity) but must be present.
Predictions can be generated using other checkpoints by specifying the `--checkpoint` argument.


# 8. Acknowledgement <a name="ack"></a>

This repository is based on the [MLCommons MLPerf RNN-T training benchmark.](https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch)
