# State Resets

State Resets is a streaming-compatible version of the 'Dynamic Overlapping Inference' proposed in
[this paper](https://arxiv.org/pdf/1911.02242.pdf).
It is a technique that can be used during inference, where the hidden state of the
model is reset after a fixed duration. This is achieved by splitting long utterances into shorter
segments, and evaluating each segment independently of the previous ones.

State Resets can be amended to include an overlapping region, where each of the
segments have prepended audio from their previous segments. The overlapping region of the next segment is
used as a warm-up for the decoder between the state resets and tokens emitted in the overlapping
region are always from the first segment.

Evaluation with State Resets is on by default, with the following arguments:

```bash
--sr_segment=15 --sr_overlap=3
```

With these arguments, the utterances longer than 15 seconds will be split into segments of 15 seconds each,
where, other than the first
segment, all segments include the final 3 seconds of the previous segment.

Experiments indicate that the above defaults show a 10% relative reduction in the WER for
long-utterances, and do not deteriorate the short utterance performance.

To turn off state resets, set `--sr_segment=0`.

```admonish
In order to use state resets it is required that the `--val_batch_size` is kept to the default value of 1.
```

## At inference time

The user can configure whether to use state resets on the CAIMAN-ASR server.
State resets are off by default, and enabling them will reduce RTS by 20–25%.

## See also

State resets is applied at inference-time. A training-time feature, [RSP](challenging_target_data.md#random_state_passing)
can be used in conjunction with state-resets to further reduce WERs on long utterances.
