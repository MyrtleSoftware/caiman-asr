# Benchmarking CAIMAN_ASR on custom datasets

## Data format in JSON

CAIMAN_ASR evaluation runs by default on LibriSpeech dev-clean dataset.

If the user wants to run an evaluation on a custom dataset, they need to generate a JSON manifest
with the transcripts and the paths to the audio files in the following format:

```bash
    [
        {
        "transcript": "BLA BLA BLA ...",
        "files": [
            {
            "channels": 1,
            "sample_rate": 16000.0,
            "bitdepth": 16,
            "bitrate": 155000.0,
            "duration": 11.21,
            "num_samples": 179360,
            "encoding": "WAV",
            "silent": false,
            "fname": "test-clean/5683/32879/5683-32879-0004.wav"
            }
        ],
        "original_duration": 11.21,
        "original_num_samples": 179360
        },
        ...
    ]
```

Please refer to the documentation [here](../training/json_format.md),
specifically the section [Convert your dataset to the JSON format](../training/json_format.md#other_datasets_json)
for more information.

## CTM file

In order to evaluate user-perceived latency, CAIMAN_ASR requires a CTM file, which contains the ground truth of when the speaker finished words. This can be generated according to the instructions
[here](../training/emission_latency.md).

See the instructions regarding launching the docker container [here](../training/ml_training_flow.md#environment-setup),
and run the above command to generate the CTM file with the model config argument as: `--model_config configs/testing-1023sp_run.yaml`.

## Notes on the custom dataset format

- The audio files should be in `WAV` format
- The audio files, the JSON manifest and the CTM file should be copied under `$HOME/.cache/myrtle/benchmark/\<custom_dataset_dir>/`.
- The `JSON` manifest should be named `<custom_dataset_name>-wav.json`.
- The `CTM` file should be named `<custom_dataset_name>.wav.ctm`.
- Please make sure that the audio file paths inside the JSON manifest and the CTM file are relative to the directory where the JSON manifest and the CTM file are stored.

## Running the evaluation on custom data

Run the evaluation script according to the instructions in [CAIMAN-ASR benchmark](caiman-asr_benchmark.md)
with the additional flags:

```bash
--data_dir <custom_dataset_dir> --dset <custom_dataset_name>
```
