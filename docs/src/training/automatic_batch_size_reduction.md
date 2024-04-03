# Automatic batch size reduction

When validating on long utterances with the large model, the encoder
may run out of memory even with a batch size of 1.

State resets are implemented by splitting one utterance into a batch
of smaller utterances, even when `--val_batch_size=1`.
This creates an opportunity to reduce the VRAM usage
further, by processing the 'batch' created from one long utterance in smaller
batches, instead of all at once.

The validation script will automatically reduce the batch size if the number
of inputs to the encoder is greater than `--max_inputs_per_batch`. The default
value of `--max_inputs_per_batch` is 1e7, which was calibrated to let the
large model validate on a 2-hour-long utterance on an 11 GB GPU.

Note that this option can't reduce memory usage on a long utterance if state resets
is turned off, since the batch size can't go below 1.

You may wish to reduce the default `--max_inputs_per_batch` if you have a smaller GPU/longer utterances.
Increasing the default is probably unnecessary, since validation on an `8 x A100 (80GB)` system
is not slowed down by the default `--max_inputs_per_batch`.
