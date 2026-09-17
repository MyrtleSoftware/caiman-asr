# Language Detection

Building on [Conditional Training](conditional_training.md) and [Conditional Decoding](conditional_decoding.md), **Language Detection Mode** allows a multilingual model to automatically identify and transcribe the correct language without requiring a specific language prefix at inference time.

## Motivation

In many real-world scenarios, the language of the input audio is unknown beforehand. While standard multilingual models require a specific prefix (e.g., `<lang_en>`, `<lang_fr>`) to condition their output, a model trained with language detection can operate in a "zero-knowledge" state. By using a generic `<lang_detect>` prefix, the model identifies the language from the audio features and proceeds with transcription.

## How it Works

Language detection is achieved by "masking" specific language tokens during training:

1. **Training**: Instead of always prepending the ground-truth language token (e.g., `<lang_en>`), the system randomly replaces it with a special `<lang_detect>` token based on a defined probability (`--lang_mask_prob`). This forces the model to learn language-discriminative features from the audio itself when the explicit language hint is missing.
2. **Inference**: At decoding time, providing the `lang_detect` prefix signals the model to perform automatic language identification.

## Training

To enable language detection during training, use the `--lang_mask_prob` argument.

```bash
./scripts/train.sh \
  --conditional \
  --conditional_language \
  --lang_mask_prob 0.35 \
  # ... other args ...
```

### Hyperparameters

Through empirical testing, a masking probability of **0.35** (35%) was found to be the optimal balance. This allows the model to:

- Maintain high accuracy when the language is explicitly provided.
- Enable automatic language detection when the language is unknown.

### Configuration

You must add the `lang_detect` token to the `user_tokens` section of your model configuration (`.yaml`):

```yaml
user_tokens:
  # ... other tokens ...
  pnc: "<pnc>"
  nopnc: "<nopnc>"
  lang_detect: "<lang_detect>"
```

## Inference (Decoding)

To use language detection during validation or inference, set the `--val_lang_prefix` to `lang_detect`. This tells the decoder to prepend the `<lang_detect>` control token instead of a specific language code.

```bash
./scripts/val.sh \
  --pnc_prefix=pnc \
  --val_lang_prefix=lang_detect
```

```admonish
The model must have been trained with a non-zero `--lang_mask_prob` for this mode to function correctly.
```
