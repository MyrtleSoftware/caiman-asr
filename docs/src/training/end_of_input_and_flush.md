# End of input (EOI) 'token'

During training/validation if the `append_EOI` option is set to `true` in the
configuration file, like:

```yaml
input_val: # or input_train
  audio_dataset:
    append_EOI: true
```

Then a special EOI mel-spectrogram is appended to the end of each audio sample.
This communicates to the model that the audio sample is about to terminate and
allows the model to 'dump' any internally buffered predictions. Empirically
this has proven to reduce WER.

During inference on the FPGA this token is appended at the close of stream
automatically.

## EOI-drop regularization

Previously, the un-warned termination incentivised early emissions to minimise
the risk of not emitting in time. Hence, the EOI token increases emission
latency. To counter this, a fraction of the training utterances randomly have
their EOI tokens removed. This is controlled via the `--eoi_drop_fraction`
argument and defaults to `0.5` (50%).
