# Data preparation <a name="data_preparation"></a>

Having chosen which [model configuration](model_yaml_configurations.md) to train, you will need to complete the following preprocessing steps:

1. Prepare your data in one of the supported training formats: `JSON` or `WebDataset`.
2. Create a sentencepiece model from your training data.
3. Record your training data log-mel stats for input feature normalization.
4. Populate a YAML configuration file with the [missing fields](model_yaml_configurations.md#missing_yaml_fields).

### Text normalization <a name="text_norm"></a>

```admonish
The examples assume a character set of size 28: a space, an apostrophe and 26 lower case letters. If transcripts aren't normalized during this preprocessing stage, they will be normalized on the fly during training (and validation) as by default in the YAML config templates, `normalize_transcripts: true`.
```

### See also

* [Prepare LibriSpeech in `JSON` format](json_format.md#librispeech_json)
* [Supported dataset formats](supported_dataset_formats.md)
* [Input activation normalization](log_mel_feature_normalization.md)
