# Data preparation <a name="data_preparation"></a>

Having chosen which [model configuration](model_yaml_configurations.md) to train, you will need to complete the following preprocessing steps:

1. Prepare your data in one of the supported training formats: `JSON` or `WebDataset`.
2. Create a sentencepiece model from your training data.
3. Record your training data log-mel stats for input feature normalization.
4. Populate a YAML configuration file with the [missing fields](model_yaml_configurations.md#missing_yaml_fields).
5. Generate an n-gram language model from your training data.

### Text normalization <a name="text_norm"></a>

```admonish
The examples assume a character set of size 28: a space, an apostrophe and 26 lower case letters.
Transcripts will be normalized on the fly during training,
as set in the YAML config templates, `normalize_transcripts: lowercase`.
See [Changing the character set](changing_the_character_set.md)
for how to configure the character set and normalization.
During validation, the predictions and reference transcripts
will be [standardized](wer_calculation.md#wer-standardization).
```

### Text standardization <a name="text_standard"></a>

```admonish
Training on multiple datasets can negatively affect WER because the same word
is transcribed with different conventions across the datasets. It is possible to
make the training transcripts consistent
by setting `standardize_text: true` in the YAML config (this is the default).
This will apply the same standardization rules as used in validation
as described in the WER Standardization section of the [WER calculation docs](./wer_calculation.md)
- but in this case, to the training transcripts.

```

### See also

- [Prepare LibriSpeech in `JSON` format](json_format.md#librispeech_json)
- [Supported dataset formats](supported_dataset_formats.md)
- [Input activation normalization](log_mel_feature_normalization.md)
