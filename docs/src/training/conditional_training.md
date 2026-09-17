# Conditional Training

Conditional training allows a single ASR model to generate transcripts in different formats by conditioning its output on a given prompt. Currently, this is used to switch between:

- Producing text with casing and punctuation versus lowercased, un-punctuated text.
- Producing text from different languages (e.g., English vs. Spanish) in a multilingual ASR model.

## How conditional training works

Conditional training works by prepending a sequence of control tokens to the
beginning of the target transcript during training.

The training process is as follows:

1. The full sequence, `prompt (control tokens) + transcript`, is fed into the model's prediction network.
2. The prediction network outputs hidden states for the entire sequence.
3. The hidden states corresponding to the initial `prompt` are stripped away.
4. The remaining hidden states, corresponding only to the `transcript`, are passed to the joint network.
5. (Optional) A similar conditional scheme can be applied to the encoder, as discussed in [Conditional Encoder](#conditional-encoder).

This method, a form of teacher-forcing, conditions the model’s output without teaching it to predict the prompt tokens themselves.

## Punctuation and casing

### Motivation

For ASR, generating normalized text (lowercase, no punctuation) is an easier task than generating fully formatted text. By collapsing multiple written forms (e.g., "hello.", "Hello?", "hello!") into a single target ("hello"), the model can achieve a lower WER.

Conditional training enables training a single model that can produce both highly accurate, unformatted text and more readable, formatted text.

### How It Works

The control tokens used for punctuation and casing are:

- **`<pnc>`**: This token prompts the model to produce a transcript with punctuation and casing.
- **`<nopnc>`**: This token prompts the model to produce a lowercased transcript with no punctuation.

During training, each utterance is randomly assigned either the formatted (`<pnc>`) or unformatted (`<nopnc>`) transcript version. The probability of selecting the `<pnc>` version is controlled by the `--pnc_prob` argument.

Experimentally, a probability of `0.25` was found to yield the best overall WER for both decoding modes (`<pnc>` and `<nopnc>`), without negatively impacting the model's ability to produce formatted text (PER remained virtually unchanged). This result indicates that the model only needs to see a relatively small fraction of punctuated, cased transcripts to effectively learn formatting.

### Usage

To enable conditional training, add the `--conditional` flag to your training command.

```bash
./scripts/train.sh \
  # ... other args ...
  --conditional
```

Adjust the sampling probability between formatted and unformatted transcripts with the `--pnc_prob` argument (default `0.25`):

```bash
./scripts/train.sh \
  # ... other args ...
  --conditional \
  --pnc_prob 0.5
```

You must add the following to `user_tokens` in the model config (.yaml) before training:

```yaml
pnc: "<pnc>"
nopnc: "<nopnc>"
```

## Multilingual ASR <a name="multilingual-asr"></a>

If you supply the `--conditional_language` flag to training then, for each
language code e.g. `en:` in your multilingual shar/json dataset yaml you will
require a matching user token i.e. `lang_en: <lang_en>` in your model config
(.yaml). During training time each utterance will be prepended (but after any
pnc/nopnc prefix tokens) with its language token e.g. `<lang_en>`. For example:

```yaml
lang_en: "<lang_en>"
lang_fr: "<lang_fr>"
```

This assumes the model's character set covers all of these languages, and
that each utterance carries the right language code — see [Training on
multiple languages](changing_the_character_set.md#training-on-multiple-languages)
for the rest of the multilingual setup.

## Conditional Encoder

In multilingual settings, making the encoder *prefix-aware* (or language-aware) boosts performance. By default, the model passes `<pnc>` and/or `<lang_xx>` prefixes to the encoder, mirroring the behavior of the prediction network.

### Implementation Details

When enabled, the model uses a dedicated learnable embedding table. Each prefix is assigned an embedding vector that functions as "pseudo-log-mel" features, which are then fed directly into the encoder network.

To facilitate this, a `prefix_token_map` maps specific prefix `token_ids` to their corresponding indices in the embedding table.

### Configuration

Toggle this feature via the model config (e.g. testing-1023sp.yaml):

```yaml
rnnt:
  # ... other configs ...
  conditional_encoder: true
```

## Next Steps

Having trained a conditional model, go to the [conditional decoding docs](conditional_decoding.md)
to select the output format at validation/inference time using a prefix.
