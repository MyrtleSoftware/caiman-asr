# Beam search decoder

By default validation is carried out using a greedy decoder. To instead use a beam decoder, run

```
./scripts/val.sh --decoder=beam
```

which runs with a default beam width of 4. To change the beam width, run, for example

```
./scripts/val.sh --decoder=beam --beam_width=8
```

```admonish
All of the beam decoder options described in this page are available in `train.sh` as well as `val.sh`.
```

## Adaptive beam search

The time synchronous beam decoder utilises an optimised version of beam search - adaptive beam search - which reduces decoding compute and latency by reducing the number of beam expansions to consider, without degrading WER. Three hypothesis pruning methods are employed:

1. Hypotheses with a score less than `beam_prune_score_thresh` (default 0.4) below the best hypothesis' score are pruned.
2. Tokens with a logprob score less than `beam_prune_topk_thresh` (default 1.5) below the most likely token are ignored.
3. Hypotheses are depth-pruned when their most recent common ancestor is further than `beam_final_emission_thresh` seconds older than the best hypothesis. This has the effect of forcing finals at at-least this interval, which reduces tail emission latencies.

Reducing `beam_prune_score_thresh`, `beam_prune_topk_thresh`, and `beam_final_emission_thresh` increases pruning aggressiveness; setting them \< 0 disables pruning.

Adaptive beam search dynamically adjusts computation based on model confidence, using more compute when uncertain and behaving almost greedily when confident.

## Softmax temperature

The beam decoder applies a softmax temperature to the logits. By default the `--temperature=1.4` as this was found to improve WER across a range of configurations. Increasing the temperature will increase the beam diversity and make the greedy path less likely.

## Fuzzy top-k logits

When using `--decoder=beam`, the model first calculates the logits for all classes (or tokens),
applies the log-softmax function to get log probabilities, and then selects the top beam-width
tokens to expand the beam.

However, in the hardware-accelerated solution, the I/O of sending the full vocab-size logits tensor from the
FPGA to the CPU is a bottleneck. To address this, the hardware-accelerated solution sends a reduced set of logits
to the CPU. Specifically, it sends the highest-value logits within some local block of the logits tensor.
This enables a 'fuzzy top-k' operation which is approximate to the full top-k operation with some small difference.

Our experiments show that using the reduced logits tensor (fuzzy top-k logits) does not impact
the model's WER performance.

### Using fuzzy top-k logits

To use a reduced tensor implementation similar to the accelerated version, run the following command:

```bash
./scripts/val.sh --num_gpus=2 --decoder=beam --fuzzy_topk_logits
```

Please note that the evaluation time will increase by ~30% compared to standard beam search,
so it is disabled by default.

## N-gram language models

N-gram language models are used with beam decoding to improve WER. This is on by default and described in more detail in the [N-gram language model documentation](ngram_lm.md).
