# Data preparation <a name="data_preparation"></a>

Having chosen which [model configuration](model_yaml_configurations.md) to train, you will need to complete the following preprocessing steps:

1. Prepare your data in one of the supported training formats: `JSONL` or `SHAR`.
2. Create a sentencepiece model from your training data.
3. Record your training data log-mel stats for input feature normalization.
4. Populate a YAML configuration file with the [missing fields](model_yaml_configurations.md#missing_yaml_fields).
5. Generate an n-gram language model from your training data.
6. Optionally [standardize transcripts](jsonl_format.md#standardizing-transcripts)

### Text normalization <a name="text_norm"></a>

The examples assume a character set of size 73 (English, i.e. `langs: [en]`):

- 26 lowercase letters,
- 26 uppercase letters,
- space,
- full stop (period),
- comma,
- apostrophe,
- question mark,
- digits 0-9,
- `%`, `$`, `¢`, `€`, `£`, `-`.

Transcripts will be normalized on the fly during training, as configured by
the `standardize_text`, `preserve_case`, and `preserve_punctuation` fields
in the YAML config templates.
See [Changing the character set](changing_the_character_set.md)
for how to configure the character set and normalization,
including for models trained on [multiple languages](changing_the_character_set.md#training-on-multiple-languages).

During validation, the predictions and reference transcripts
will be [standardized](wer_calculation.md#wer-standardization).

### See also

- [Prepare LibriSpeech in `JSONL` format](jsonl_format.md#librispeech_json)
- [Supported dataset formats](supported_dataset_formats.md)
- [Input activation normalization](log_mel_feature_normalization.md)
- [Controlling the proportion of data from each manifest](training.md#controlling-the-proportion-of-data-from-each-manifest) for multilingual/multi-manifest training
