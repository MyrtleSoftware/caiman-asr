# Changing the character set

With default training settings, the CAIMAN-ASR model will output
punctuated (space, full stop, comma, apostrophe, question mark, digits,
and a few currency symbols) and cased characters specific to the languages
that training is performed on.

This page describes how to change the settings to support additional characters,
different languages, or multiple languages at once.

The code has been tested with English and other european language training.
If you would like additional support for a specific language,
please contact [caiman-asr@myrtle.ai](mailto:caiman-asr@myrtle.ai)

## Guidelines

### Step 1: Choose a character set

The character set is not written out by hand. Instead, it is derived
automatically from the `langs` field of the model config, which lists the
ISO language codes you want the model to support, for example:

```yaml
langs: [en]
```

For each language code, [ICU](https://icu.unicode.org/) supplies the set of
characters typically used to write that language (both lower and upper case).
These per-language characters are combined with a fixed set of shared
punctuation, digits, and currency symbols
(`' , . ? 0123456789 % $ ¢ € £ -`) and space, to make the final character
set (check the file
[`data/text/allowed_charset.py`](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/data/text/allowed_charset.py)).
So for `langs: [en]` the character set is:

- 26 lowercase English letters,
- 26 uppercase English letters,
- space,
- full stop (period),
- comma,
- apostrophe,
- question mark,
- digits 0-9,
- `%`, `$`, `¢`, `€`, `£`, `-`.

If you list more than one language in `langs`, the character sets for all of
the languages are unioned together, for example `langs: [en, fr]` produces a
character set containing both the English and French alphabets. See
[Training on multiple languages](#training-on-multiple-languages) below for
how such a model is trained and used.

```admonish
`langs` and `tokenizer.labels` are mutually exclusive. If ICU's exemplar set
for a language doesn't give you the character set you want, you can instead
write the character set out by hand as `tokenizer.labels` in the model config,
skipping `langs` entirely. The rest of this page still applies — the
per-utterance `lang_code` used for on-the-fly normalization in Step 2 still
comes from the language codes in your dataset yaml either way.
```

The maximum size of your character set is the sentencepiece vocabulary size,
as each character in the character set receives a unique token
in the sentencepiece vocabulary.
See [here](model_yaml_configurations.md) for the vocabulary size
for each model configuration.

We recommend keeping the character set at least an order of magnitude smaller
than the sentencepiece vocabulary size.
Otherwise there may be too few multi-character subwords in the vocabulary,
which might make the model less effective.

### Step 2: Configure normalization

It's possible for the raw training data to contain characters
other than those in the character set.
For instance, an English dataset might contain "café",
even if the character set is only ASCII.

```admonish
Training will crash if there are characters in the dataset
that are not in the character set, after normalization has run.
```

Unlike in previous versions of this pipeline, there is no longer a single
`normalize_transcripts` setting that picks between a handful of named levels
(`identity`/`scrub`/`ascii`/`digit_to_word`/`lowercase`). Instead, every
transcript is passed through a fixed pipeline of steps, some of which are
optional and configurable:

1. **Transliteration** (always on): characters not in the character set are
   transliterated to their closest ASCII equivalent, for example "café"
   becomes "cafe" and "straße" becomes "strasse". This uses the transcript's
   `lang_code` to decide, case-insensitively, which characters count as
   "in the character set".
2. **Standardization** (optional, on by default for training): applies
   Whisper-style normalization to English transcripts: stripping filler words
   (e.g. "um", "uh"), expanding contractions, and standardizing British/American English
   spelling. This step is controlled by `standardize_text` in the
   yaml config, and is disabled for validation.
3. **Replacements** (optional, see Step 3 below): your own custom string
   substitutions.
4. **Tag removal** (always on): tags such as `<silence>` or `<foreign_word>`
   are stripped out. This is no longer configurable.
5. **Scrubbing** (always on): any character still not in the character set is
   replaced with a space.

Note that digit-to-word conversion (e.g. "123rd" becoming
"one hundred and twenty-third") is no longer part of the pipeline,
so a `<pnc>` transcript (or a non-conditionally-trained model) keeps
digits as digits. It still happens on a separate lowercase-normalization path
used for the `<nopnc>` variant of a transcript under [conditional
training](conditional_training.md#punctuation-and-casing), and for WER
computation during validation (see [WER
standardization](wer_calculation.md#wer-standardization)). If you want digits
spelled out for `<pnc>` or non-conditional output too, include the conversion
in your own custom `replacements` (Step 3) or pre-process your transcripts
before training.

### Step 3: Custom replacements

You may want to tweak how text is normalized,
beyond what the pipeline above does automatically.
For example, you might want to make the following changes
to your training transcripts:

- Replace ";" with ","
- Replace "-" with " " if "-" isn't in your character set,
  so that "twenty-one" becomes "twenty one" instead of "twentyone"

You can make these changes
by adding custom replacement instructions to the yaml file. Example:

```yaml
replacements:
  - old: ";"
    new: ","
  - old: "-"
    new: " "
```

In the normalization pipeline, these replacements are applied
after standardization and before the transcripts are scrubbed
of characters not in the character set.
The replacements are applied even if `standardize_text` is disabled,
although by default there are no replacements.

### Step 4: Tag removal

Some datasets contain tags, such as `<silence>` or `<affirmative>`.
These tags are always removed from the training transcripts
during on-the-fly text normalization, before the text is tokenized.
Hence the model will not predict these tags during inference.

```admonish
If you want the model to be trained with tags and possibly predict tags
during inference, you'll need to add them as user symbols in the model config,
(wrapped in angle brackets, e.g. `<silence>`),
and train your tokenizer on a dataset containing them.
```

### Step 5: Update fields in the model configuration

You'll want to update:

- the character set by
  - updating the languages under `langs` to the languages included in your model
  - or, for a fully custom character set, updating `tokenizer.labels`
- whether to standardize transcripts on the fly, under `standardize_text`
- whether standardization preserves casing and punctuation, under
  `preserve_case` and `preserve_punctuation`
- the replacements under `replacements`

```admonish
Note that only one of `labels` or `langs` should be present in the model config.
The languages provided in the langs should be ISO language codes, for example:
`en` for `English`, `es` for Spanish. If using country codes, please use an underscore
instead of a hyphen as normally used in ISO standards. For example, for US English, use
`en_US`
```

### Step 6: Train a sentencepiece model

The following command is used to train
the Librispeech sentencepiece model
using the default character set,
as happens [here](jsonl_format.md#quick-start):

```bash
{{#include ../../../training/scripts/make_json_artifacts.sh:spm_in_mdbook}}
```

This script reads the config file,
so it will train the correct sentencepiece model
for your character set and replacements.

You may also wish to run some other scripts in
`scripts/make_json_artifacts.sh`,
such as the scripts that prepare the LM data
and train the n-gram LM using your new tokenizer.

### Step 7: Finish filling out the model configuration

If you haven't filled out the standard
[missing fields](model_yaml_configurations.md#missing-yaml-fields)
in the yaml config file, be sure to update them,
especially the `sentpiece_model` you trained in Step 6.

### Step 8: Large character sets

If you are training on a language like Chinese
that has a large character set, be sure to train
a sentencepiece model with at least as many tokens
as there are unique characters.

```admonish
You may also want to
[use character error rate](./wer_calculation.md#character-error-rate-and-mixture-error-rate)
```

The sentencepiece model will not include characters
in its vocabulary if they are exceptionally rare in the training data.
This is not an issue when training on English, since no character is very rare.
But for other languages, this can cause training to crash during the token cache generation.

To prevent this, you can change the error to a warning:

1. If your sentencepiece model is `/path/to/sentencepiece.model`, create a file called `/path/to/sentencepiece.yaml`. This is a configuration file that controls global settings for the sentencepiece model.

2. Add the following line:

   ```yaml
   unk_handling: WARN
   ```

   (The default is `FAIL`.)

3. If you still see many (>100) warnings about unknown tokens
   during training, there likely is a true problem with your sentencepiece model.
   Please contact [caiman-asr@myrtle.ai](mailto:caiman-asr@myrtle.ai) so Myrtle
   can add support for your language.

## Training on multiple languages <a name="training-on-multiple-languages"></a>

Listing more than one code in `langs`, e.g. `langs: [en, fr]`, builds a single
tokenizer whose character set is the union of each language's characters, as
described in [Step 1](#step-1-choose-a-character-set). This is the basis for
training one multilingual model rather than one model per language.

A few things to know when training multilingual models:

- Each utterance still carries its own `lang_code` (assigned per-manifest in
  your dataset yaml — see
  [multilingual dataset balancing](training.md#controlling-the-proportion-of-data-from-each-manifest)),
  and the transliteration and standardization steps from
  [Step 2](#step-2-configure-normalization) above use that `lang_code` to pick
  the right rules for that utterance, even though every utterance is scrubbed
  against the same, combined character set.
- To make the model aware of which language it's transcribing (rather than
  relying purely on the audio), use `--conditional_language` to prepend a
  per-language control token to each transcript — see
  [Multilingual ASR](conditional_training.md#multilingual-asr).
- If the language of the input audio isn't known ahead of time at inference,
  [Language Detection](language_detection.md) lets the model infer it
  automatically instead of requiring a language prefix.
- When combining datasets of very different sizes across languages, or across
  sources within a language, [canary-manifest
  balancing](training.md#canary-manifest-balancing) controls how much of each
  one the model sees per epoch.
- In order to fine-tune the trained multilingual model on only a subset
  of `langs`, use `ft_langs` — see [Fine-tuning a multilingual model on a subset of languages](resuming_and_fine_tuning.md#fine-tuning-a-multilingual-model-on-a-subset-of-languages).

## Inspecting character errors

The [WER calculation](wer_calculation.md#wer-standardization) ignores
capitalization and punctuation errors, and there is no longer a way to ask it not
to, so no breakdown of those errors is available. See the note in that document.
