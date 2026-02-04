# Data preparation <a name="data_preparation"></a>

Having chosen which [model configuration](model_yaml_configurations.md) to train, you will need to complete the following preprocessing steps:

1. Prepare your data in one of the supported training formats: `JSON` or `WebDataset`.
2. Create a sentencepiece model from your training data.
3. Record your training data log-mel stats for input feature normalization.
4. Populate a YAML configuration file with the [missing fields](model_yaml_configurations.md#missing_yaml_fields).
5. Generate an n-gram language model from your training data.
6. Optionally [standardize transcripts](json_format.md#standardizing-transcripts)

### Text normalization <a name="text_norm"></a>

The examples assume a character set of size 57:

- 26 lowercase letters,
- 26 uppercase letters,
- space,
- full stop (period),
- comma,
- apostrophe,
- question mark.

Transcripts will be normalized on the fly during training,
as set in the YAML config templates, `normalize_transcripts: digit_to_word`.
See [Changing the character set](changing_the_character_set.md)
for how to configure the character set and normalization.

During validation, the predictions and reference transcripts
will be [standardized](wer_calculation.md#wer-standardization).

### See also

- [Prepare LibriSpeech in `JSON` format](json_format.md#librispeech_json)
- [Supported dataset formats](supported_dataset_formats.md)
- [Input activation normalization](log_mel_feature_normalization.md)
