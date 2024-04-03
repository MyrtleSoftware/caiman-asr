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

This checkpoint will load into val.py with "EMA" warnings that can be ignored.
