# Model YAML configurations

Before training, you must select the model configuration you wish to train.
Please refer to the [key features](../key_features.md#model-configurations) for a description of the options available,
as well as the [training times](training_times.md).
Having selected a configuration it is necessary to note the config path and sentencepiece vocabulary size ("spm size")
of your chosen config from the following table as these will be needed in the subsequent [data preparation steps](data_preparation.md):

|   Name    | Parameters | spm size |                                                       config                                                       | Acceleration supported? |
| :-------: | :--------: | :------: | :----------------------------------------------------------------------------------------------------------------: | :---------------------: |
| `testing` |    49M     |   1023   | [testing-1023sp.yaml](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/configs/testing-1023sp.yaml) |           ❌            |
|  `base`   |    85M     |   8703   |    [base-8703sp.yaml](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/configs/base-8703sp.yaml)    |           ✅            |
|  `large`  |    196M    |  17407   |  [large-17407sp.yaml](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/configs/large-17407sp.yaml)  |           ✅            |

It is recommended to train the `base` model on LibriSpeech as described [here](training.md) before training `base` or `large` on your own data.

```admonish
The `base` and `large` architectures were optimized to provide a good tradeoff between WER and throughput on the accelerator.
Other architectures will not run on the accelerator.
```

`train.sh` will verify that you are training a model
that is supported by the accelerator.
If you want to skip this check so you can
train the testing model for more rapid iteration,
pass the flag `--skip_state_dict_check` to `train.sh`.

## Missing YAML fields <a name="missing_yaml_fields"></a>

The configs referenced above are not intended to be edited directly. Instead, they are used as templates to create `<config-name>_run.yaml` files. The `_run.yaml` file is a copy of the chosen config with the following fields populated:

- `sentpiece_model: /datasets/sentencepieces/SENTENCEPIECE.model`
- `stats_path: /datasets/stats/STATS_SUBDIR`
- `max_duration: MAX_DURATION`
- `ngram_path: /datasets/ngrams/NGRAM_SUBDIR`

Populating these fields can be performed by the `training/scripts/create_config_set_env.sh` script.

For example usage, see the following documentation: [Prepare LibriSpeech in the `JSON` format](json_format.md#librispeech_json).
