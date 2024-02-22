# Training: Resuming and Finetuning

The `--resume` option to the [train.sh script](../scripts/train.sh) enables you to resume training from a `--checkpoint=/path/to/checkpoint.pt` file including the optimizer state.

The `--fine_tune` option ensures that training starts anew, with a new learning rate schedule and optimizer state from the specified checkpoint.

To freeze the encoder weights during training change the `enc_freeze` option in the config file to:

```yaml
enc_freeze: true
```
