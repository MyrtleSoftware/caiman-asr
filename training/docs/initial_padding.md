# Initial padding

The ASR server inserts `window_size - window_stride` seconds of silence at the beginning of all audios. This is 10ms for the testing model, and 15ms for the base/large models.

To match this behavior, this repo inserts the same silence during training and validation. Experiments indicated that this does not affect WER.

This feature can be turned off by passing `--turn_off_initial_padding`.
