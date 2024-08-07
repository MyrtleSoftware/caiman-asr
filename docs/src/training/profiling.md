# Profiling

You can turn on profiling by passing `--profiler` in your training or validation command. Note that profiling will likely slow down the script and is intended as a debugging feature.

Some of the profiling results are only saved after the script completes so it is necessary to avoid killing with `Ctrl + C` if you want to record the full profiling results.
As such, when profiling training it is recommended to:

- profile a small number of `--training_steps`
- set `--n_utterances_only [N_UTTERANCES_ONLY]` to sample from the training dataset.

Similarly, when profiling validation it is recommended to use `--nth_batch_only=<batch idx>`

Profiling results will be saved in `[output_dir]/benchmark/`. This consists of:

- [yappi](https://github.com/sumerc/yappi/tree/master) logs named `program[rank]_[timestamp].prof`. These can be viewed via SnakeViz:

  Launch a container with the command `SNAKEVIZ_PORT=[an unused port] ./scripts/docker/launch.sh ...`. Inside the container, run

  ```sh
  ./scripts/profile/launch_snakeviz.bash /results/benchmark/program[rank]_[timestamp].prof
  ```

  This will print an interactive URL that you can view in a web browser.

- top logs named `top_log_[timestamp].html`. These can be viewed outside the container using a web browser.

- nvidia-smi text logs named `nvidia_smi_log_[timestamp].txt`.

- Manual timings of certain parts of the training loop for each training step constituting an epoch. These are text files named `timings_stepN_rankM_[timestamp].txt`.

- system information in `system_info_[timestamp].txt`.

## Sending results

In order to share debug information with Myrtle.ai please run the following script:

```sh
OUTPUT_DIR=/<results dir to share> TAR_FILE=logs_to_share.tar.gz ./scripts/tar_logs_exclude_ckpts.bash
```

This will compress the logs excluding any checkpoints present in `OUTPUT_DIR`. The resulting `logs_to_share.tar.gz` file can be shared with Myrtle.ai or another third-party.
