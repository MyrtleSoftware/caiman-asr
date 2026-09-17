# Agent instructions

Run commands from the repository root unless a command says otherwise. Prefer
targeted checks that complete in under 5 minutes before attempting longer
training or validation runs.

## Repository orientation

This repo contains the customer-facing CAIMAN-ASR code plus Myrtle-internal
extensions. The directories agents usually need are:

- `training/`: customer-facing ASR training package. Shared model, data,
  validation, export, and tests live here, with library code under
  `training/caiman_asr_train`.
- `myrtle-training/`: Myrtle-internal training package. Use this for internal
  example training scripts, Myrtle-specific datasets, and code under
  `myrtle-training/myrtle_asr_train`.
- `myrtle-data/`: Myrtle-specific data preparation utilities and tests.
- `inference/`: inference and benchmarking utilities.
- `docs/`: mdbook documentation sources.

Prefer putting broadly applicable training changes in `training/caiman_asr_train`.
Put Myrtle-only workflow, dataset, or internal experiment changes in
`myrtle-training`. The Myrtle docker launcher mounts both `myrtle-training` and
`training`, so internal runs can exercise changes in both trees.

Docker launchers mount the repo at `/workspace/training` and, for Myrtle
training, `/workspace/myrtle-training`. They also symlink the first launch
argument to `/datasets` inside the container; use an absolute datasets path for
commands that actually read data.

## Quick `pytest` in docker

The docker launch scripts require absolute host paths. Keep iteration fast by
replacing the `pytest` target with the smallest relevant test file or directory.

```bash
cd myrtle-training
TTY=false COMMAND="python -X utf8 -m pytest tests" ./scripts/docker/launch.sh \
    "$(mktemp -d)" "$(mktemp -d)" "$(mktemp -d)"
```

For the main training tests instead of Myrtle-specific tests:

```bash
cd training
TTY=false COMMAND="python -X utf8 -m pytest tests" ./scripts/docker/launch.sh \
    "$(mktemp -d)" "$(mktemp -d)" "$(mktemp -d)"
```

## Quick example training run

Use the English example train script with only 5 training steps and skip the
initial validation pass so agents can iterate quickly.

```bash
TTY=false myrtle-training/scripts/example_train_en.bash \
    --training_steps 5 --skip_first_val_eval --quick_val
```

This script builds and launches the Myrtle training docker image, then runs the
trainer inside the container. It uses the script defaults for dataset,
checkpoint, and result locations. `--quick_val` keeps validation enabled but
bounds the in-training validation to one batch, then skips the separate
post-training validation harness. The cached path should be quick; the first
run can take longer if docker layers or training artifacts need to be built.

## Running real trains on the Jenkins cluster

Long trains (hours to days) run on the shared Jenkins GPU cluster: launch with
`myrtle-training/scripts/queue.bash`, monitor/cancel with
`myrtle-training/scripts/jenkins/monitor.py`, and read results off the shared
`/mount/cassini/checkpoints` (colo) or `/mount/moore/checkpoints` (office)
mounts. The full launch → monitor (~12h cadence) → read-WER → cancel workflow is
documented in the **`running-trains` skill**
(`.claude/skills/running-trains/SKILL.md`), which loads automatically when you
work on cluster trains.
