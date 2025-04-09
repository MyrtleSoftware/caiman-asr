# CAIMAN-ASR benchmark

This file describes how to run evaluation of CAIMAN-ASR. Please follow the
[installation guide](./benchmark_installation.md) for instructions on setup.

Please run all commands from the `benchmark` directory.

## Note on the datasets

The evaluation script uses the LibriSpeech dev-clean dataset.
Please refer to [benchmarking with custom datasets](benchmarking_custom_datasets.md)
for instructions on how to use a custom dataset.

## Note on using Docker

If you installed using Docker,
you must prepend `./launch.bash` to all commands in this file.

## CAIMAN-ASR

The script to run CAIMAN-ASR evaluation is `transcribe_caiman.py`. It is composed of 3 steps which are all performed in sequence. There are however flags to control the flow and skip individual steps.

1. Data preparation
2. Transcription
3. Evaluation

Running the command will perform all 3 steps.

```bash
./transcribe_caiman.py --address <ADDRESS> --port <PORT> --run_name <unique name> --append_results caiman-base
```

And if you installed using Docker, instead run:

```bash
./launch.bash ./transcribe_caiman.py --address <ADDRESS> --port <PORT> --run_name <unique name> --append_results caiman-base
```

where `--address` is the websocket address of transcription service and `--port` is the port.

The above command assumes you're using the CAIMAN-ASR base model. If you want to use the large model, replace `--append_results caiman-base` with `--append_results caiman-large`.

### Note on client speed

The client simulates a real user by
feeding the audio to the server at real time.
Since there are several hours of audio,
the client will take several hours to run.

To instead do a trial run on just five audio files,
pass the flag `--limit_to 5`.

### Data preparation

This step is essential to run the first time the service is used. It downloads the evaluation audio data, ground truth CTM file, and prepares it all into manifests. There is typically no need to run it again when the user wants to only generate new transcriptions,
so the script will automatically skip this step if the data exists.

If you want to redo the data preparation step anyway, pass the `--force_data_prep` flag.

```bash
./transcribe_caiman.py --address <ADDRESS> --port <PORT> --force_data_prep --run_name <unique name> --append_results caiman-base
```

### Transcription

This step runs the transcription step and generates `*.caiman-asr.trans` transcription files. These are saved directly to the LibriSpeech directory. For each audio file, the script generates one transcription file.

By default, the transcription is skipped for already done audio,
which will happen if you run the script another time with the same `run_name`.
This behavior can be changed by passing the `--force_transcription` flag, and all files will be transcribed from scratch and existing transcription files will be overwritten.

```bash
./transcribe_caiman.py --address <ADDRESS> --port <PORT> --force_transcription --run_name <name of previous run> --append_results caiman-base
```

The transcription step can be skipped by running the `transcribe_caiman.py` and passing the `--skip_transcription` flag.

```bash
./transcribe_caiman.py --address <ADDRESS> --port <PORT> --skip_transcription --run_name <name of previous run> --append_results caiman-base
```

### Evaluation

This step takes generated trans files, aggregates them to a single `librispeech-dev-clean.caiman-asr.ctm` file, a ground truth CTM file, and computes the word error rate (WER) and latencies. The evaluation step can be skipped by running the `transcribe_caiman.py` and passing the `--skip_evaluation` flag.

```bash
./transcribe_caiman.py --address <ADDRESS> --port <PORT> --skip_evaluation --run_name <unique name> --append_results caiman-base
```

Results will also be saved in `~/.cache/myrtle/benchmark/results/[append_results].csv`.

### List of arguments

The arguments for `transcribe_caiman` are as follows:

```bash
--address <STRING_VALUE>: Server address the transcription service is running on, default=''.
--port <INT_VALUE>: Port number the transcription service is running on, default=3030.
--force_data_prep: Do the data preparation step even if the data already has been downloaded, default=False.
--force_transcription: Re-transcribe all audio files from scratch, default=False.
--skip_transcription: Skip transcription step and go directly to the evaluation step, default=False.
--skip_evaluation: Skip evaluation step, default=False.
--limit_to <INT_VALUE>: Limit transcription to a number of files. Suitable for testing purposes, default=None.
--run_name <STRING_VALUE>: Name used to identify results.
  Individual transcription files will be saved to `$HOME/.cache/myrtle/benchmark/<run name>`.
--append_results <STRING_VALUE>: Results will be appended to `~/.cache/myrtle/benchmark/results/[append_results].csv`
  Must be one of caiman-base, caiman-large.
```
