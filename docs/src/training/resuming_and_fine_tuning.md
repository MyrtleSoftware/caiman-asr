# Resuming and Fine-tuning

The `--resume` option to the `train.sh` script enables you to resume training from a `--checkpoint=/path/to/checkpoint.pt`
file including the optimizer state.
Resuming from a checkpoint will continue training from the last step recorded in the checkpoint, and the files that will be seen by
the model will be the ones that would be seen if the model training was not interrupted.
In the case of resuming training when using tar files, the order of the files that will be seen by the model is the same as the order that the model saw when
the training started from scratch, i.e. not the same as if training had not been interrupted.

The `--fine_tune` option ensures that training starts anew, with a new learning rate schedule and optimizer state from the specified checkpoint.

To freeze the encoder weights during training change the `enc_freeze` option in the config file to:

```yaml
enc_freeze: true
```
