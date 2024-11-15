# Endpointing

End-pointing (EP) is a feature that allows a model to predict when a speaker has arrived at the end of a sentence/utterance. Endpointing works by augmenting the token-set with a special control token - `<EOS>`. When model is trained on ‘segmented’ transcripts, separated with the `<EOS>` token, it learns to predict the token.

## Training an EP model

Add an EOS token definition to your model config, we suggest using `<EOS>`, see the examples in the `config` folder.

### Generating segmented transcripts

To convert a regular dataset to a segmented dataset use the `eos_add.py` script:

```sh
python scripts/eos_add.py \
    --data_dir=/datasets/Unsegmented \
    --output_dir=/datasets/Segmented \
    --manifests man1.json man2.json etc.json
```

This will use the [wtpsplit](https://github.com/segment-any-text/wtpsplit) model family to add EOS tokens to your training manifest. This may take several hours depending on the size of your dataset and hardware.

### Tokenizer with EOS support

Generate a new tokenizer from your segmented transcripts following the [documentation](./changing_the_character_set.md)

### N-gram language model

Generate a new N-gram model with your segmented transcripts and EOS-enabled tokenizer following the [documentation](./ngram_lm.md)

## Controlling endpointing during decoding

Endpointing is enabled when an EOS token is specified in the config file. If endpointing is enabled the decoding of the EOS token has special handling and the `--eos_decoding` strategy must be specified. The options available are:

1. `ignore` - The model's output probabilities are adjusted such that the probability of predicting the EOS token is zero i.e. `p(EOS) <- 0`.
2. `blank` - The probability of predicting the EOS token is moved to the blank token, i.e. `prob_blank = prob_EOS + prob_blank` and `prob_EOS = 0`
3. `none` - This is for compatibility with checkpoints whose tokenizers do not have an EOS token.
4. `predict` -  EOS prediction is on. The model predicts the EOS token as it would for any other token. Predict mode allows for controlling the posterior of the EOS token to account for the disproportionate impact that an early EOS token could have on WER if decoding in [terminal EOS mode](#early-termination-at-eos-and-ep-latency). Hence, the probability of predicting the EOS token is modified according to:

```python
def modify(logit_eos: float, eos_alpha: float, eos_beta: float) -> float:

	logit_eos = eos_alpha * logit_eos

	if eos_beta > 0 and logit_eos < log(eos_beta):
	   return -float("inf")

	return logit_eos
```

In predict mode, when `eos_alpha == 1 and eos_beta == 0` no modification is made to the output probabilities (hence these will be referred to as the _identity parameters_) which corresponds to the training environment.

### Early termination at EOS and EP latency

One of the use cases for endpointing is triggering some kind of action once a speaker has completed a sentence, e.g. dispatching a user's request from a voice assistant. If the _termination_ of the ASR is too eager, it may cut off the user. EP latency, the time between the user finishing speaking and model detecting the EOS, must be minimized without harming WER. To simulate the WER effect of cutting off the user, the model has an `--eos_is_terminal` option. This will trim transcripts after an EOS token.

If you run a validation with `--calculate_emission_latency` you will see the EOS latency statistics calculated as well. This includes an `eos_frac` which is the proportion of utterances that are terminated by the EOS token.

## Silence termination (VAD)

There is a non-speech detection fallback if the model fails to endpoint. This is off by default - to turn on set `--eos_vad_threshold=<value in seconds>`. This will trim decodings after a silence longer than the threshold is detected.

For the greedy decoder silence termination is triggered by `n`-consecutive seconds of blank tokens.

For the beam decoder silence termination is triggered if the difference between the timestamp of the most recent non-blank and the current decoding timestamp is greater than `n`. This check is performed when a final is emitted, such that there is only one beam.
