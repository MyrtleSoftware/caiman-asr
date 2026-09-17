# Batch size hyperparameters

If you are training on an `8 x A100 (80GB)` machine, the recommended batch size
hyper-parameters are [given here](training_times.md). Otherwise, this page
gives guidance on how to select them.

Batch size in CAIMAN-ASR can be fixed or dynamic, depending on the `--sampling_mode`:

- "fixed", fixed number of utterances per batch
- "duration", dynamic batches with approx equal total duration, AKA 'duration
  batches'
- "1D-bucketing", dynamic duration batches from pre-defined duration
  buckets
- "1D-bucketing-dyn", dynamic duration batches from dynamically created
  duration buckets
- "2D-bucketing", dynamic batches with approx equal VRAM usage from pre-defined
  duration/token buckets
- "2D-bucketing-fixed", fixed batches from pre-defined duration/token buckets
- "2D-bucketing-singlet", fixed size batches from a single duration/token
  bucket

## `grad_accumulation_batches (GAB)`

One optimizer step is taken every `--grad_accumulation_batches` batches. Each
GPU runs the same number of batches (the elements per batch may differ between
ranks) and the gradients are reduced across GPUs before the step, so the batch
the model sees per step is:

```python
global_batch_size = avg_per_gpu_batch_size * num_gpus * grad_accumulation_batches
```

RNN-T models need a large `global_batch_size` to reach good WERs — aim for
`>1024` utterances or `>5` hours — but the larger it is, the longer training
takes.

GAB significantly affects the VRAM usage. Take the largest per-GPU batch that
does not OOM, then set GAB to reach the global batch size you want.

## `joint_lattice_cap (JLC)`

The joint network's output needs far more VRAM than the encoder and prediction
networks do for the same batch, so each batch is split again before the joint
and loss. The encoder and prediction still run at the full `PER_GPU_BATCH_SIZE`
— they are most of the compute, so under-filling them would cost throughput.

Each split holds at most `--joint_lattice_cap` lattice cells, where an utterance
contributes `enc_len * (token_len + 1)` cells. This budgets the joint and loss
independently of the batch size and the utterance lengths, and does not affect
WER.

Set it before the batch, since it decides how much VRAM the batch gets. Too
small and the split count grows until per-split overhead dominates; past the
knee, each doubling costs twice the VRAM for under 1% more joint+loss
throughput. Typical values are 2^20-2^21 at `V=1024`, 2^18-2^19 at `V=8704` and
2^17-2^18 at `V=17408`; the measured ones live in
`myrtle-training/scripts/select_gab_jlc.bash`.

Note: JLC supersedes the older "batch_split_factor" hyperparameter

## Summary

1. Set `--joint_lattice_cap` first. On the Myrtle cluster `select_gab_jlc.bash`
   supplies it; elsewhere measure it with
   `caiman_asr_train.train_utils.jlc_bench`.
2. Tune the sampler arguments (duration, bucketing, fixed batch size) for the
   largest per-GPU batch that fits.
3. Set `--grad_accumulation_batches` so the global batch reaches your target
   duration.

Test with your **full training dataset**: the utterance length distribution
drives peak VRAM.
