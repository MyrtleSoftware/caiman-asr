# Conditional Training

Conditional training allows a single ASR model to generate transcripts in different formats by conditioning its output on a given prompt. Currently, this is used to switch between producing text with casing and punctuation versus lowercased, un-punctuated text.

## Motivation

For ASR, generating normalized text (lowercase, no punctuation) is an easier task than generating fully formatted text. By collapsing multiple written forms (e.g., "hello.", "Hello?", "hello!") into a single target ("hello"), the model can achieve a lower WER.

Conditional training enables training a single model that can produce both highly accurate, unformatted text and more readable, formatted text.

## How It Works

Conditional training works by prepending a control token to the beginning of the target transcript during training.

- **`<pnc>`**: This token prompts the model to produce a transcript with punctuation and casing.
- **`<nopnc>`**: This token prompts the model to produce a lowercased transcript with no punctuation.

During training, each utterance is randomly assigned either the formatted (`<pnc>`) or unformatted (`<nopnc>`) transcript version. The probability of selecting the `<pnc>` version is controlled by the `--pnc_prob` argument.

Experimentally, a probability of `0.25` was found to yield the best overall WER for both decoding modes (`<pnc>` and `<nopnc>`), without negatively impacting the model's ability to produce formatted text (PER remained virtually unchanged). This result indicates that the model only needs to see a relatively small fraction of punctuated, cased transcripts to effectively learn formatting.

The training process is as follows:

1. The full sequence, `prompt (control token) + transcript`, is fed into the model's prediction network.
2. The prediction network outputs hidden states for the entire sequence.
3. The hidden states corresponding to the initial `prompt` are stripped away.
4. The remaining hidden states, corresponding only to the `transcript`, are passed to the joint network.

This method, a form of teacher-forcing, conditions the model’s output without teaching it to predict the prompt tokens themselves.

## Usage

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

```admonish
Conditional training requires `batch_split_factor > 1`.
```

## Next Steps

Having trained a conditional model, go to the [conditional decoding docs](conditional_decoding.md)
to select the output format at validation/inference time using a prefix.
