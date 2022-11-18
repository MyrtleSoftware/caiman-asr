# Myrtle.ai RNN-T PyTorch Training Script

# 0. Contents

1.  [Installation](#installation)
2.  [Data](#data)
3.  [Training](#training)
4.  [Validation Sets](#validation)
5.  [Validation on CPU](#valcpu)
6.  [Hardware Inference Server Support](#server)
7.  [Python Inference](#inference)
8.  [Test Sets](#test)
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

#### Requirements
Currently, the reference uses CUDA-11.0 (see [Dockerfile](Dockerfile#L15)).
Here you can find a table listing compatible drivers: https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver


# 2. Data <a name="data"></a> 

This repository provides scripts for the following datasets:

* LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12)
* CommonVoice [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)

To train on your own proprietary dataset you will need to arrange for it to be in the same format produced
by the data download and preprocessing scripts below.  In short, this means all audio as wav (or flac) files
in a single directory hierarchy, all (text-normalized) transcripts in [json file(s)](examples/example.json)
referencing these audio files and their durations, a sentencepiece model built from these transcripts and a
suitably modified config file.  We recommend reading through both examples below and suggest working through
at least the LibriSpeech example before trying to train on your own data.

## 2.1 Data Download

### Start an interactive session in the Docker container to run data download, training and inference

```
./scripts/docker/launch.sh <DATASETS> <CHECKPOINTS> <RESULTS>
```

Within the container, the contents of this repository will be copied to the `/workspace/rnnt` directory.
The container directories `/datasets`, `/checkpoints`, and `/results` are mounted as volumes
and mapped to the corresponding directories `<DATASETS>`, `<CHECKPOINTS>`, `<RESULTS>` on the host.

Note that the host directories passed to docker/launch.sh must have absolute paths.  Note also that, at
present, /checkpoints is not actually used; checkpoints are written to and read from /results;
this may change in the future.

### Download the dataset

No GPU is required for data download and preprocessing. Therefore, if GPU usage is a limited resource, launch 
the container for this section on a CPU machine by following previous steps.

Note: Downloading and preprocessing the dataset requires up to 1TB of free disk space and can take several
hours to complete.

#### LibriSpeech

LibriSpeech contains 1000 hours of 16kHz read English speech derived from public domain audiobooks from
LibriVox project and has been carefully segmented and aligned. For more information, see the [LIBRISPEECH: AN
ASR CORPUS BASED ON PUBLIC DOMAIN AUDIO BOOKS](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf)
paper.

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

## 2.2 Data Preprocessing

Note that while the scripts below prepare wav files the NVIDIA DALI code that loads the speech can also
load flac and ogg files.  In our experiments we have found training on flac files to be just as fast as
training on wav files (both read from SSD) and we will be using flac files in the future to save space.

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

The above script also creates the sentencepiece model in /datasets/sentencepieces/ and adjusts the config
file to use this model and data.

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

The above script also creates the sentencepiece model in /datasets/sentencepieces/ and adjusts the config
file to use this model and data.


# 3. Training <a name="training"></a> 

We recommend the use of PyTorch automatic-mixed-precision on all training runs, enabled by setting 
`AMP=true` as shown below.  

RNN-T trains using very large synthetic batches, typically including 1000 to 4000 utterances, specified here
with the GLOBAL_BATCH_SIZE variable.  These large synthetic batches are split between the NUM_GPUS working
on the problem, and nibbled away at over GRAD_ACCUMULATION_STEPS computation steps over which gradients are
accumulated until a single update is applied to the model.  The actual batch_size fed to each GPU is not
directly under our control but can be calculated using the formula

```
batch_size * NUM_GPUS * GRAD_ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE
```

## 3.1 Training Commands

#### LibriSpeech

Before running any training commands be sure that your config file reflects your sentencepiece model and data.
If you did not run the scripts above this can be achieved by running:

```
cat configs/base.yaml | sed s/SENTENCEPIECE/librispeech1023/g | sed s/MAX_DURATION/16.7/g > configs/baseline_v3-1023sp.yaml
```

On LibriSpeech our 24GB TITAN RTX GPUs can be run with a batch_size of 24 without incurring an out-of-memory 
error.  This is selected by setting GLOBAL_BATCH_SIZE=1008 and GRAD_ACCUMULATION_STEPS=21 together with 
NUM_GPUS=2.  On the cloud we have found that 16GB V100s can support a batch_size of 16, and 40GB A100s can 
support a batch_size of 40.

The default setup saves no checkpoints at all until the 200th epoch.  
Set SAVE_BEST=1 to get an overwriting checkpoint every time dev Word Error Rate (WER) improves.  
Set SAVE_AT_THE_END=true to get a non-overwriting checkpoint at the end of training.  

The default number of epochs to train for is 100.  
Set, for example, EPOCHS=150 to make a different choice.  

So, on two 24GB TITAN RTX GPUs, training LibriSpeech, run the following inside the container:

```
AMP=true NUM_GPUS=2 GLOBAL_BATCH_SIZE=1008 GRAD_ACCUMULATION_STEPS=21 SAVE_BEST=1 SAVE_AT_THE_END=true EPOCHS=150 ./scripts/train.sh
```

#### CommonVoice

Before running any training commands be sure that your config file reflects your sentencepiece model and data.
If you did not run the scripts above this can be achieved by running:

```
cat configs/base.yaml | sed s/SENTENCEPIECE/commonvoice1023/g | sed s/MAX_DURATION/7.75/g > configs/baseline_v3-1023sp.yaml
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
we have used batch_size 24 as in the LibriSpeech case.

We therefore currently scale GRAD_ACCUMULATION_STEPS by a factor of 12.3/5.7 to 45, and set
GLOBAL_BATCH_SIZE to GRAD_ACCUMULATION_STEPS * NUM_GPUS * batch_size, which is 2160, to get
approximately 3.5 hours of speech per update.

For CommonVoice training, also on two 24GB TITAN RTX GPUs, we therefore run:

```
AMP=true NUM_GPUS=2 GLOBAL_BATCH_SIZE=2160 GRAD_ACCUMULATION_STEPS=45 SAVE_BEST=1 SAVE_AT_THE_END=true EPOCHS=150 DATA_DIR=/datasets/CommonVoice/cv-corpus-10.0-2022-07-04/en TRAIN_MANIFESTS=train.json VAL_MANIFESTS=dev.json ./scripts/train.sh
```

Fine-tuning CommonVoice models to use hard activation functions (see Section 6) has proven more difficult 
than fine-tuning LibriSpeech models, with highly trained checkpoints being the most problematic.  For this 
reason it can be useful to save intermediate checkpoints during training by specifying `SAVE_FREQUENCY=10`,
together with `SAVE_MILESTONES="10 20 30 40"` etc to prevent their deletion, and to fine-tune from one of
these rather than the final checkpoint.

The rest of this README defaults to LibriSpeech but all commands can be adapted to work with CommonVoice by
setting the appropriate variables from the training command just shown.

## 3.2 TensorBoard

The training scripts write TensorBoard logs to /results during training.

To monitor training using TensorBoard, launch the port-forwarding TensorBoard container in another terminal:

```
./scripts/docker/launch_tb.sh <DATASETS> <CHECKPOINTS> <RESULTS>
```

Inside this container run

```
tensorboard --logdir /results --host 0.0.0.0 --port 6010
```

Then navigate to http://traininghostname:6010 in a web browser.  

If a connection dies and you can't reconnect to port 6010 because it's already allocated, run:

```
docker ps
docker stop <name of docker container with port forwarding>
```

Note that TensorBoard datapoint timestamps are given in your local timezone while training script terminal 
logs and nvlog.json logs use UTC.

## 3.3 Training Examples

TensorBoard training curve examples are given in directory [examples](examples/).

In [examples/success/LibriSpeech](examples/success/LibriSpeech) we present learning rate,
loss and word-error-rate curves for a successful train on the LibriSpeech dataset.  

In [examples/success/CommonVoice](examples/success/CommonVoice) we present learning rate, grad-norm, 
loss and word-error-rate curves for a successful mixed-precision train on the CommonVoice dataset.  

In [examples/failure](examples/failure/) we include grad-norm, loss and word-error-rate 
curves for a pathalogical CommonVoice training case in which grad-norm blew up to infinity and then went
to NaN.  This pathology is caused by exploding gradients in the RNN-T model encoder.  This pathology
affected roughly 1 in 10 LibriSpeech trains; we do not yet have enough data to estimate its prevalence
with CommonVoice.  Training with larger GLOBAL_BATCH_SIZE values and smaller max/mean duration ratios
both appear to make this pathology less likely to occur.  If this pathology does occur, resuming training
from the last-saved good checkpoint is sometimes successful.

Another pathology we see is training runs freezing, with one GPU going to 0% utilization and the other to
100% utilization, with one Python process dying, but no error messages.  In this case reducing the batch
size usually avoids the problem.  We hope to investigate this pathology further in the future.


# 4. Validation Sets <a name="validation"></a> 

Validation is performed throughout training.  However, we can also validate models after training.  The 
default validates the best checkpoint saved in training (to be found in /results/RNN-T_best_checkpoint.pt) 
against the LibriSpeech dev-clean partition.  Inside the container run  

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


# 5. Validation on CPU <a name="valcpu"></a> 

It is also possible to validate models using CPU only.  The CPU version has additional functionality not found 
in the GPU version, some of which is described in the Sections below.  You can also increase 
MAX_SYMBOL_PER_SAMPLE to validate or infer over much longer utterances than may be possible on GPU.  

To validate against the LibriSpeech dev-clean partition, run  

```
./scripts/valCPU.sh
```

As above, the VAL_MANIFESTS, DATA_DIR and CHECKPOINT variables allow you to specify other options.  

# 6. Hardware Inference Server Support <a name="server"></a> 

To run your model on Myrtle's hardware-accelerated inference server you will need to fine-tune your model
weights using the custom LSTM with hard activation functions, dump mel statistics from your dataset to
support streaming normalization, and create a hardware checkpoint to enable transfer of this and other data.

### Custom LSTM

We can swap out the standard PyTorch LSTM for our own [CustomLSTM](./common/custom_lstm.py) by changing the
config file in directory [configs](configs/) to read  

```
custom_lstm: true
```

The value of the Custom LSTM is that it exposes the internal LSTM machinery enabling us to experiment with 
hard activation functions, quantization, and to apply dropout to more weights than the standard PyTorch 
version allows.  Hard activation functions can be switched on in the config file using

```
custom_lstm: true
hard_activation_functions: true
```

All operations using the Custom LSTM run slower than corresponding operations using the PyTorch LSTM.

### Fine Tuning

Since training with hard activation functions is slow we will in general train using the PyTorch LSTM 
as described in Section 3 above, and then fine-tune that model to use hard activation functions.  The 
following command, using a lower peak learning rate, a faster warmup, and a shorter hold is recommended 
for fine-tuning LibriSpeech models on two TITAN RTX GPUs.  The FINE_TUNE=true option ensures that training 
starts anew, with the new learning rate schedule, from the specified checkpoint.  The config file should of 
course specify custom_lstm and hard_activation_functions as described above.  You may wish to backup your 
PyTorch LSTM checkpoint under a different name before running this command since it will overwrite:

```
AMP=true LEARNING_RATE=0.001 WARMUP_EPOCHS=2 HOLD_EPOCHS=16 EPOCHS=36 NUM_GPUS=2 GLOBAL_BATCH_SIZE=1008 GRAD_ACCUMULATION_STEPS=21 SAVE_BEST=1 SAVE_AT_THE_END=true FINE_TUNE=true CHECKPOINT=/results/RNN-T_best_checkpoint.pt ./scripts/train.sh
```

Fine-tuned models should use the Custom LSTM config settings during PyTorch validation and inference.


### Streaming Normalization

Audio processing is carried out during training using NVIDIA's DALI in a Python environment.  For the
hardware-accelerated inference server Myrtle has implemented most of the same algorithms in Rust.

Feature normalization is an exception since the DALI algorithm computes the mean and standard deviation used
to normalize each mel-bin over the full length of each utterance, which is incompatible with streaming.  An
adaptive streaming normalization algorithm is implemented in [./common/stream_norm.py](./common/stream_norm.py)
which can be run using valCPU.\* as shown below.  The same algorithm is implemented in Rust in the inference
server.  

The adaptive streaming normalization algorithm uses exponentially weighted moving averages and variances to
perform normalization of each new frame of mel-bin values.  The algorithm requires an initial set of mel-bin
mean and variance values as a starting point and these can be obtained by running the training script with
DUMP_MEL_STATS set to true  

```
DUMP_MEL_STATS=true NUM_GPUS=2 GRAD_ACCUMULATION_STEPS=21 EPOCHS=1 ./scripts/train.sh
```

The training data statistics are written to `<RESULTS>` as melmeans.\* and melvars.\* as both PyTorch
tensors and Numpy arrays.  These arrays can be transfered to the Rust inference server code and read by
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
python ./utils/hardware_ckpt.py \
    --fine_tuned_ckpt /results/RNN-T_best_checkpoint.pt \
    --config configs/baseline_v3-1023sp.yaml \
    --melmeans /results/melmeans.pt \
    --melvars /results/melvars.pt \
    --melalpha 0.001 \
    --output_ckpt /results/hardware_ckpt.pt
```

where `/results/RNN-T_best_checkpoint.pt` is your best fine-tuned hard-activation-function checkpoint. 
The hardware checkpoint also contains the sentencepiece model specified in the config file, the dataset mel
means and variances as dumped above, and the specified alpha value for mel statistics decay in streaming
normalization.  

This checkpoint will load into val.py and valCPU.py with "EMA" warnings that can be ignored.


# 7. Python Inference <a name="inference"></a> 

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

To get predictions for one or more wav files without manually creating a json file, run:

```
python ./utils/get_predictions.py /results/path/to/file1.wav /results/path/to/file2.wav ...
```

This script internally sets `$DATA_DIR=/`, so it takes absolute paths as input.  The paths must be visible
from inside the container, so it's recommended to place the wav files under /results or /datasets. 
You can specify which model checkpoint to use with `--checkpoint /results/custom_checkpoint.pt`.

# 8. Test Sets <a name="test"></a> 

To test a model against a test set, the following commands will use the best validation-set checkpoint saved
during training (to be found in /results/RNN-T_best_checkpoint.pt).  

To test on the LibriSpeech test-clean partition, inside the container run  

```
NUM_GPUS=2 VAL_MANIFESTS=/datasets/LibriSpeech/librispeech-test-clean-wav.json ./scripts/val.sh
```

To test on the (noisy-speech) LibriSpeech test-other partition, run  

```
NUM_GPUS=2 VAL_MANIFESTS=/datasets/LibriSpeech/librispeech-test-other-wav.json ./scripts/val.sh
```


# 9. Acknowledgement <a name="ack"></a> 

This repository is based on the [MLCommons MLPerf RNN-T training benchmark.](https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch)

