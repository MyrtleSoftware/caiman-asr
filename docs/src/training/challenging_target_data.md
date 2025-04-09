# Challenging target data

This page describes data augmentations that may help with these problems:

- Problem: Your target audio has non-speech background noise
  - Solution: Train with background noise
- Problem: Speakers in your target audio talk over each other
  - Solution: Train with babble noise
- Problem: Your target audio was recorded at 8 kHz, e.g. a narrowband telephone connection
  - Solution: Train with narrowband conversion

### Page contents

- [Background Noise](#background_noise) for training with background noise
- [Babble Noise](#babble_noise) for training with babble noise
- [Narrowband](#narrowband) for training with narrowband conversion
- [Inspecting Augmentations](#inspect_audio) to listen to the effects of augmentations
- [Random State Passing](#random_state_passing) for training on long sequences
- [Tokens Sampling](#tokens_sampling) for training with random tokens sampling
- [Gradient Noise](#gradient_noise) for training with gradient noise

## Example Command

The following command will train the base model on the LibriSpeech dataset on an `8 x A100 (80GB)` system with these settings:

- applying background noise to 25% of samples
- applying babble noise to 10% of samples
- downsampling 50% of samples to 8 kHz
- using the default noise schedule
  - initial values 30–60dB
  - noise delay of 4896 steps
  - noise ramp of 4896 steps

```bash
./scripts/train.sh --model_config=configs/base-8703sp_run.yaml --num_gpus=8 \
    --grad_accumulation_batches=1 --batch_split_factor=8 \
    --training_steps=42000 --prob_background_noise=0.25 \
    --prob_babble_noise=0.1 --prob_train_narrowband=0.5 \
    --val_manifests=/datasets/LibriSpeech/librispeech-dev-other-flac.json
```

These augmentations are applied independently,
so some samples will have all augmentation types applied.

## Background noise training <a name="background_noise"></a>

Background noise is set via the `--prob_background_noise` argument.
By default, `prob_background_noise` is `0.25`.
Background noise takes a non-speech noise file and mixes it with the speech.

On an `8 x A100 (80GB)` system, turning off background noise augmentation increases the base model's training throughput by ~17% and the large model's throughput by ~11%.

### Implementation

The noise data is combined with speech data on-the-fly during training, using a
signal to noise ratio (SNR) randomly chosen between internal variables `low` and `high`.

The initial values for `low` and `high` can be specified (in dB) using the `--noise_initial_low` and
`--noise_initial_high` arguments when calling `train.sh`. This range is then maintained for the number of
steps specified by the `--noise_delay_steps` argument after which the noise level is ramped up over
`--noise_ramp_steps` to its final range.
The final range for background noise is 0–30dB (taken from the Google paper "Streaming
end-to-end speech recognition for mobile devices", [He et al., 2018](https://arxiv.org/abs/1811.06621)).

Before combination, the noise audio will be duplicated to become at least as long as the speech utterance.

### Background noise dataset

By default, background noise will use [Myrtle/CAIMAN-ASR-BackgroundNoise](https://huggingface.co/datasets/Myrtle/CAIMAN-ASR-BackgroundNoise) from the [Hugging Face Hub](https://huggingface.co/docs/hub/en/datasets-overview).

Note that this dataset will be cached in `~/.cache/huggingface/` in order to persist between containers.
You can change this location like so: `HF_CACHE=[path] ./scripts/docker/launch.sh ...`.

To change the default noise dataset, set `--noise_dataset` to an audio dataset on the Hugging Face Hub.
The training script will use all the audios in the noise dataset's `train` split.

If you instead wish to train with local noise files, make sure your noise is organized in the Hugging Face [AudioFolder](https://huggingface.co/docs/datasets/en/audio_dataset#audiofolder) format.
Then set `--noise_dataset` to be the path to the directory containing your noise data (i.e. the parent of the `data` directory), and pass `--use_noise_audio_folder`.

## Babble noise training <a name="babble_noise"></a>

Babble noise is set via the `--prob_babble_noise` argument.
By default, `prob_babble_noise` is `0.0`.
Babble is applied by taking other utterances from the same batch and mixing them with the speech.

### Implementation

Babble noise is combined with speech in the same way that background noise is.
The `--noise_initial_low`, `--noise_initial_high`, `--noise_delay_steps`, and `--noise_ramp_steps`
arguments are shared between background noise and babble noise.

The only difference is that the final range of babble noise is 15–30dB.

## Narrowband training <a name="narrowband"></a>

For some target domains, data is recorded at (or compressed to) 8 kHz (narrowband). For models trained with audio >8 kHz (16 kHz is the default) the audio will be upsampled to the higher sample rate before inference. This creates a mismatch between training and inference, since the model will partly rely on information from the higher frequency bands.

This can be partly mitigated by resampling a part of the training data to narrowband and back to higher frequencies, so the model is trained on audio that more closely resembles the validation data.

To apply this downsampling on-the-fly to a random half of batches, set `--prob_train_narrowband=0.5` in your training command.

## Inspecting augmentations <a name="inspect_audio"></a>

To listen to the effects of augmentations, pass `--inspect_audio`. All audios will then be saved to `/results/augmented_audios` after augmentations have been applied. This is intended for debugging only—DALI is slower with this option, and a full epoch of saved audios will use as much disk space as the training dataset.

## Random State Passing <a name="random_state_passing"></a>

RNN-Ts can find it difficult to generalise to sequences longer than those seen during training, as described in [Chiu et al, 2020](https://arxiv.org/abs/2005.03271).

Random State Passing (RSP) ([Narayanan et al., 2019](https://arxiv.org/abs/1910.11455)) reduces this issue by simulating longer sequences during training. It does this by initialising the model with states from the previous batch with some probability.
On in-house validation data, this reduces WERs on long (~1 hour) utterances by roughly 40% relative.

### Further details

Experiments indicated:

- It was better to apply RSP 1% of the time, instead of 50% as in the paper.
- Applying RSP from the beginning of training raised WERs, so RSP is only applied after `--rsp_delay` steps
  - `--rsp_delay` can be set on the command line but, by default, is set to the step at which the learning rate has decayed to 1/8 of its initial value (i.e. after x3 `half_life_steps` have elapsed). To see the benefits from RSP, it is recommended that >=5k updates are done after the RSP is switched on, so this heuristic will not be appropriate if you intend to cancel training much sooner than this. See [docstring of `set_rsp_delay_default` function](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/train_utils/rsp.py) for more details.

RSP is on by default, and can be modified via the `--rsp_seq_len_freq` argument, e.g. `--rsp_seq_len_freq 99 0 1`.
This parameter controls RSP's frequency and amount; see the `--rsp_seq_len_freq` docstring in [`args/train.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/train.py).

RSP requires Myrtle.ai's custom LSTM which is why `custom_lstm: true` is set by default in the yaml configs.

### See also

RSP is applied at training-time. An inference-time feature, [state resets](./state_resets.md) can be used in conjunction with RSP to further reduce WERs on long utterances.

## Tokens Sampling <a name="tokens_sampling"></a>

Text needs to be in the form of tokens before it is processed by the RNNT. These tokens can represent words, characters, or
subwords. CAIMAN-ASR uses subwords which are formed out of 28 characters, namely the lower-case english alphabet
letters, along with the space and apostrophe characters. The tokens are derived from the tokenizer model
[SentencePiece](https://github.com/google/sentencepiece).
A SentencePiece tokenizer model can be trained on raw text, and produces a vocabulary with the most probable subwords that
emerge in the text. These derived vocabulary entries (i.e. the tokens) are scored according to the (negative log) probability
of occurring in the text that the tokenizer was trained on. The tokenizer entries include all the individual characters of the
text, in order to avoid out-of-vocabulary error when tokenizing any text. When using the tokenizer model to convert text into
tokens the user has the option of tokenizing not with the most probable tokens (subwords), but with a combination of tokens
that have lower score.

Utilising the random tokens sampling is a form of data augmentation and it is applied on a percentage of the
training data, and not on the validation data.
This can be done with setting the sampling parameter into a real value in the range [0.0, 1.0]
in the configuration file, e.g.:

```yaml
sampling: 0.05
```

A value of 0.05 (default) means that 5% of the training data will be tokenized with random tokens sampling.
A value of 0.0 means no use of tokens sampling, whereas a value of 1.0 applies random
tokens sampling in the whole text.

## Gradient Noise <a name="gradient_noise"></a>

Adding Gaussian noise to the network gradients improves generalization to out-of-domain datasets by not over-fitting
on the datasets it is trained on. Inspired by the research paper by [Neelakantan et. al.](https://openreview.net/pdf?id=rkjZ2Pcxe),
the noise level is sampled from a Gaussian distribution with \\(mean=0.0\\) and standard deviation that decays according to
the following formula:

$$
\\sigma(t)=\\frac{noise}{{(1 + t - t\_{start})}^{decay}},
$$

\\(noise\\) is the initial noise level, \\(decay=0.55\\) is the decay constant, \\(t\\) is the step,
and \\(t\_\{start}\\) is the step when the gradient noise is switched on.

Training with gradient noise is switched off by default. It can be switched on by setting the noise level
to be a positive value in the config file.

Experiments indicate that the best time to switch on the gradient noise is after the warm-up period
(i.e. after `warmup_steps`). Moreover, the noise is only added in the gradients of the encoder components,
hence if during training the user chooses to freeze the encoder, adding gradient noise will be off by default.
