# Export inference checkpoint

To run your model on Myrtle.ai's hardware-accelerated inference server you will need to create a hardware checkpoint to enable transfer of this and other data.

This requires mel-bin mean and variances as described [here](./log_mel_feature_normalization.md).

To create a hardware checkpoint run:

```
python ./caiman_asr_train/export/hardware_ckpt.py \
    --ckpt /results/RNN-T_best_checkpoint.pt \
    --config <path/to/config.yaml> \
    --output_ckpt /results/hardware_checkpoint.testing.example.pt
```

where `/results/RNN-T_best_checkpoint.pt` is your best checkpoint.

The script should take a few seconds to run.

The generated hardware checkpoint will contain the sentencepiece model specified in the config file and the dataset mel stats.

The hardware checkpoint will also include the binary n-gram generated during preprocessing, as specified by the `ngram_path` field in the config file.
However, this is optional, and can be skipped by passing the `--skip_ngram` flag:

```
python ./caiman_asr_train/export/hardware_ckpt.py \
    --ckpt /results/RNN-T_best_checkpoint.pt \
    --config <path/to/config.yaml> \
    --output_ckpt /results/hardware_checkpoint.testing.example.pt
    --skip_ngram
```

To include an n-gram that was generated on a different dataset, use the `--override_ngram_path` argument:

```
python ./caiman_asr_train/export/hardware_ckpt.py \
    --ckpt /results/RNN-T_best_checkpoint.pt \
    --config <path/to/config.yaml> \
    --output_ckpt /results/hardware_checkpoint.testing.example.pt \
    --override_ngram_path /path/to/ngram.binary
```

```admonish
The inference checkpoint will load into val.py with "EMA" warnings that can be ignored.
```
