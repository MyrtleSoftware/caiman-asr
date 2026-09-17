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

## Fine-tuning a multilingual model on a subset of languages

If you trained a multilingual model with, say, `langs: [en, fr, de]` (see
[Training on multiple languages](changing_the_character_set.md#training-on-multiple-languages)),
you can resume its training on only some of those languages, for example
just French, by adding `ft_langs` to the model config:

```yaml
langs: [en, fr, de]
ft_langs: [fr]
```

`ft_langs` is optional and, when present, selects which of the manifest's
languages are loaded as training data, in place of `langs`. Crucially,
`langs` itself is left untouched, so the sentencepiece tokenizer, ngram
model, and mel stats are unaffected and stay identical to the original
model - only the training data changes.

Arguments you should tune in order to resume the train successfully:

```bash
--ft_langs <ft-langs>  # list of languages to fine-tune on
--checkpoint <path>  # path to the checkpoint file
--training_steps <steps>  # number of training steps to run
```

```admonish
Make sure that you do not change the hyperparameters that you used for training
the multilingual model, otherwise the creation of new artifacts may be triggered
and the training may be inaccurate. Such hyperparameters include the --multingual
set to true, and the same --dataset_name.
```

`ft_langs` has no effect on validation, which continues to use whatever
`--val_manifests`/`--val_standardizer_lang` are passed on the command line.
