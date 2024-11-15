# Changing the character set

With default training settings, the CAIMAN-ASR model will only output
lowercase ASCII characters, space, and `'`.
This page describes how to change the settings to support additional characters
or different languages.

The code has been tested with English language training,
but it provides basic support for other languages.
If you would like additional support for a specific language,
please contact [caiman-asr@myrtle.ai](mailto:caiman-asr@myrtle.ai)

## Guidelines

### Step 1: Choose a character set

As described above, the default character set is `abcdefghijklmnopqrstuvwxyz '`.

The maximum size of your character set is the sentencepiece vocabulary size,
as each character in the character set receives a unique token
in the sentencepiece vocabulary.
See [here](model_yaml_configurations.md) for the vocabulary size
for each model configuration.

We recommend keeping the character set at least an order of magnitude smaller
than the sentencepiece vocabulary size.
Otherwise there may be too few multi-character subwords in the vocabulary,
which might make the model less effective.

### Step 2: Choose a normalizer

It's possible for the raw training data to contain characters
other than those in the character set.
For instance, an English dataset might contain "café",
even if the character set is only ASCII.

```admonish
Training will crash if there are characters in the dataset
that are not in the character set.
```

To handle these rare characters, you can select a normalizer
in the yaml config file. The options, in order of least to most interference, are:

- `identity`
  - Does not transform the input text
- `scrub`
  - Removes characters that are not in the config file's character set
  - Recommended for languages that use a character set different than ASCII
- `ascii`
  - Replaces non-ASCII characters with ASCII equivalents
  - For example, "café" becomes "cafe"
  - Recommended if model is predicting English with digits
  - Also applies `scrub`
- `digit_to_word`
  - Replaces digits with their word equivalents
  - For example, "123rd" becomes "one hundred and twenty-third"
  - Assumes English names for numbers
  - Also applies `ascii` and `scrub`
- `lowercase`
  - Lowercases text and expands abbreviations
  - For example, "Mr." becomes "mister"
  - This is the default normalizer
  - Recommended for predicting lowercase English
    without digits
  - Also applies `digit_to_word`, `ascii`, and `scrub`

### Step 3: Custom replacements

You may want to tweak how text is normalized,
beyond the five normalizers listed above.
For example, you might want to make the following changes
to your training transcripts:

- Replace ";" with ","
- Replace "-" with " " if normalization is on
  and "-" isn't in your character set,
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

In the normalization pipeline, these replacements will be applied
just before the transcripts are scrubbed of characters not in the character set.
The replacements will still be applied even if the normalizer is `identity`,
although by default there are no replacements.

### Step 4: Tag removal

Some datasets contain tags, such as `<silence>` or `<affirmative>`.
By default, these tags are removed from the training transcripts
during on-the-fly text normalization, before the text is tokenized.
Hence the model will not predict these tags during inference.
If you want the model to be trained with tags
and possibly predict tags during inference,
set `remove_tags: false` in the yaml file.

```admonish
If you set `remove_tags: false`
but do not train your tokenizer on a dataset with tags,
the tokenizer will crash
if it sees tags during model training or validation.
```

### Step 5: Update fields in the model configuration

You'll want to update:

- the character set under `labels` to your custom character set
- the normalizer under `normalize_transcripts`
- the replacements under `replacements`
- Whether to remove tags, under `remove_tags`

### Step 6: Train a sentencepiece model

The following command is used to train
the Librispeech sentencepiece model
using the default character set,
as happens [here](json_format.md#quick-start):

```bash
{{#include ../../../training/scripts/make_json_artifacts.sh:spm_in_mdbook}}
```

This script reads the config file,
so it will train the correct sentencepiece model
for your character set, normalizer, and replacements.

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

## Inspecting character errors

By default, the [WER calculation](wer_calculation.md#wer-standardization)
ignores capitalization or punctuation errors.
If you would like to see an analysis of these errors,
you can use the flag `--breakdown_wer`.
