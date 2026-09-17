# Challenging target data

This page describes data augmentations that may help with these problems:

- Problem: Your target audio has non-speech background noise
  - Solution: Train with background noise
- Problem: Speakers in your target audio talk over each other
  - Solution: Train with babble noise
- Problem: Your target audio is recorded on a poor-quality microphone
  - Solution: Train with microphone augmentations
- Problem: Your target audio is in a different room environment to your training data
  - Solution: Train with room augmentations

## Page contents

- [Background Noise](#background_noise) for training with background noise
- [Babble Noise](#babble_noise) for training with babble noise
- [Microphone and Room Augmentations](#mic-room-aug)
- [Inspecting Augmentations](#inspect_audio) to listen to the effects of augmentations
- [Random State Passing](#random_state_passing) for training on long sequences
- [Tokens Sampling](#tokens_sampling) for training with random tokens sampling
- [Label Smoothing](#label_smoothing) for training with label smoothing
- [Gradient Noise](#gradient_noise) for training with gradient noise

## Background noise training <a name="background_noise"></a>

Background noise is set via the `--prob_background_noise` argument.
By default, `prob_background_noise` is `0.25`.
Background noise takes a noise file and mixes it with the speech.

### Implementation

The noise data is combined with speech data on-the-fly during training, using a
signal to noise ratio (SNR) randomly chosen between internal variables `low`
and `high`. The values for `low` and `high` can be specified (in dB) using the
`--noise_snr_low` and `--noise_snr_high` arguments when calling `train.sh`.

Before combination, the noise audio will be duplicated to become at least as
long as the speech utterance.

### Background noise dataset

By default, background noise will use
[Myrtle/CAIMAN-ASR-BackgroundNoise](https://huggingface.co/datasets/Myrtle/CAIMAN-ASR-BackgroundNoise)
from the [Hugging Face
Hub](https://huggingface.co/docs/hub/en/datasets-overview).

Note that this dataset will be cached in `~/.cache/huggingface/` in order to persist between containers.
You can change this location like so: `HF_CACHE=[path] ./scripts/docker/launch.sh ...`.

To change the default noise dataset, set `--noise_dataset` to an audio dataset on the Hugging Face Hub.
The training script will use all the audios in the noise dataset's `train` split.

## Babble noise training <a name="babble_noise"></a>

Babble noise can be emulated using the background noise mechanism whilst
supplying noise data that includes human speech.

## Microphone and room augmentations <a name="mic-room-aug"></a>

These are both created by convolving impulse response functions with the audio
signal. Set `--prob_room_aug` to a non zero value to enable convolution with
simulated room impulse responses. Alternatively, set `--room_dataset` to a path
to a lhotse recording-set to use your own room impulse responses.

We recommend using the room impulse response collection introduced by "A Study
on Data Augmentation of Reverberant Speech for Robust Speech Recognition"

For microphone augmentations, we recommend using the impulse responses from the
["Microphone impulse response project"](http://micirp.blogspot.com/)

## Inspecting augmentations <a name="inspect_audio"></a>

To listen to the effects of augmentations, pass `--inspect_audio`. All audios will then be saved to `/results/augmented_audios` after augmentations have been applied. This is intended for debugging only—the data pipeline is slower with this option, and a full epoch of saved audios will use as much disk space as the training dataset.

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

### Implementation notes

Unintuitively, random state passing requires the penultimate hidden state and
the last embedding token of the previous batch. Consider two utterances, with
token sequences `[I, like, cats]` and `[You, love, dogs]`

By training on the concatenation of these two utterances,
the prediction net calculation looks like:

```txt
(A)
SOS I like cats You love dogs
  ↘  ↘    ↘   ↘   ↘   ↘   ↘
0 ➡h1➡h2 ➡h3 ➡h4 ➡h5 ➡h6 ➡h7
```

Here 0 is the zero embedding vector which LSTMs use as the initial hidden state
for SOS.

By training on these utterances individually, the pred net calculations are:

```txt
(B)
SOS I like cats
  ↘  ↘   ↘    ↘
0 ➡h1➡h2 ➡h3 ➡h4
```

And:

```txt
(C)
SOS You love dogs
  ↘    ↘   ↘   ↘
0 ➡h1 ➡H2 ➡H3 ➡H4
```

Where H_i is a different state from h_i. Note that for the second utterance (C)
the hidden states are different than in A, because the context of the first
utterance is lost. This can be fixed by replacing (SOS, 0) in C with (cats, h3)
from (B):

```txt
(D)
cats You love dogs
   ↘    ↘   ↘   ↘
h3 ➡h4 ➡h5 ➡h6 ➡h7
```

Hence, (D) has the same hidden states as A since combining the tokens and states
is done the same way as in (A).

Thus, it's required to save the embedding of last token (e.g. cats) and the
next-to-last state (e.g. h3) to make this work.

For full details see the implementation in `training/caiman_asr_train/core/model.py`

## Tokens Sampling <a name="tokens_sampling"></a>

Text needs to be in the form of tokens before it is processed by the RNNT. These tokens can represent words, characters, or
subwords. CAIMAN-ASR uses subwords, formed out of the character set configured for the model
(see [Changing the character set](changing_the_character_set.md)). The tokens are derived from the tokenizer model
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

## Label smoothing <a name="label_smoothing"></a>

Label smoothing is the process of softening target labels in a classification
problem. Normally this is achieved by transforming a delta-function target to a
delta-function plus uniform distribution. However, the extension of this to
RNNT is not straightforward if non-local smoothing is desired. Myrtle uses a
doubly-stochastic approach: alternative labels are synthesized on the fly and
used a small percentage of the time. This smooths the averaged target over
multiple optimizer steps. The labels are generated by replacing words in the
ground-truth label with "close" words (as determined by the Levenshtein
distance).

Label smoothing is on by default, it is controlled by the `label_smoothing_kwargs`
sub-key of the `tokenizer` configuration in the config file:

```yaml
label_smoothing_kwargs:
  dict_path: "<path/to/words.txt>"
  levenshtein_threshold: <float>
  prob_smooth_word: <float>
```

Where `dict_path` is the path to the dictionary file, this should be formatted
the same as a standard unix dictionary file (see `/usr/share/dict/words` or
`/usr/dict/words`). `levenshtein_threshold * len(word)` is the maximum Levenshtein distance
for words to be considered "close". `prob_smooth_word` is the probability of a word
being replaced by a "close" word (the "close" words are sampled uniformly).
Recommended values are `levenshtein_threshold=0.34` and `prob_smooth_word=0.01`.

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
