# Word timestamps

In the following discussion _timestamps_ refers only to the timestamps of the
__finals__.

The model emits timestamps for each token, these can be grouped into _word
timestamps_ which define the start/end time of each word.

## Average accumulated shift

The accuracy of the word timestamps is quantified with the _average accumulated
shift_ (AAS) metric (see <https://arxiv.org/pdf/2301.12343>). This is defined as:

$$
\text{AAS} = \frac{1}{N}
  \sum_{i=1}^N \frac{
  \left| t_i^{\text{predStart}} - t_i^{\text{refStart}}  \right| +
  \left| t_i^{\text{predEnd}} - t_i^{\text{refEnd}}  \right|
  }
  {2}
$$

This can be understood as the average absolute difference (in time) between the
true start/end time and the model's predicted start/end time for each word.

## Latency

The model's token level timestamps are known to lag behind real-time (see
[Emission Latency](./emission_latency.md)). To correct for this when
estimating word timestamps a latency offset should be subtracted from the token
timestamp at the beginning and end of each word. This is supported via the
following flags:

```sh
--latency_head_offset <value in seconds>
--latency_tail_offset <value in seconds>
```

In general these require a model/domain/decoder specific calibration.

## Measuring AAS

If the `--calculate_emission_latency` flag is passed to the
[Validation](./validation.md) script then several AAS related metrics are
measured these include:

- `"optimal_head_offset"`:

$$
  \text{median} \left\lbrace t_i^{\text{predStart}} - t_i^{\text{refStart}} \mid i \in 1\ldots N \right\rbrace
$$

- `"optimal_tail_offset"`:

$$
  \text{median} \left\lbrace t_i^{\text{predEnd}} - t_i^{\text{refEnd}} \mid i \in 1\ldots N \right\rbrace
$$

- `"raw_AAS"`: The AAS calculated without any latency correction
- `"fixed_AAS"`: The AAS calculated with the head/tail offset supplied via the
  CLI flags
- `"corrected_AAS"`: The AAS calculated using the computed optimal head/tail
  offset
