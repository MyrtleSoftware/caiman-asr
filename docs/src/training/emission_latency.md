# Emission Latency

Emission latency (EL) is defined by the time difference between the end of a spoken word in an audio file and when the model outputs the final token for the corresponding word, minus the mean frame latency.
After the model receives the final audio frame of a word, it might not predict the word until it has heard a few more frames of audio.
EL measures this delay.

To calculate the model's EL during validation, pass the `--calculate_emission_latency` flag, e.g.

```bash
./scripts/val.sh --calculate_emission_latency
```

When this flag is enabled, CTM files containing model timestamps are exported to `--output_dir`.

Emission latencies are calculated by aligning the model-exported CTM files with corresponding ground truth CTM files.
Ground truth CTM files are expected to be located in the same directory as the validation manifest or tar files and should share the same base name e.g.
if `--data_dir=/path/to/dataset` and `--val_manifests=data.json`, then the assumed filepath of the ground truth CTM file is `/path/to/dataset/data.ctm`.
See [Forced Alignment](#forced-alignment) for details on producing ground truth CTM files.

The script outputs the mean latency, as well as the 50th, 90th, and 99th percentile latencies.

Moreover, the Token Usage Rate is reported. This is the proportion of words' timestamps that are used in the emission latency calculation.

If one already has model-exported CTM files and corresponding ground truth files, the [measure_latency.py](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/latency/measure_latency.py)
script can be used instead of running a complete validation run. To do so, run the script with paths to the ground truth and model CTM files:

```bash
python caiman_asr_train/latency/measure_latency.py --gt_ctm /path/to/ground_truth.ctm --model_ctm /path/to/model.ctm
```

To include substitution errors in latency calculations, add the `--include_subs` flag:

```bash
python caiman_asr_train/latency/measure_latency.py --gt_ctm /path/to/ground_truth.ctm --model_ctm /path/to/model.ctm --include_subs
```

To export a scatter plot of EL against time from the start of the sequence, pass a filepath to the optional `--output_img_path` argument e.g.

```bash
python caiman_asr_train/latency/measure_latency.py --gt_ctm /path/to/ground_truth.ctm --model_ctm /path/to/model.ctm --output_img_path /path/to/img.png
```

EL logging is also compatible with `val_multiple.sh`.

## Forced Alignment

The script [forced_align.py](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/latency/forced_align.py) is used to align audio recordings with their corresponding transcripts, producing ground truth timestamps for each word.

To perform forced alignment, execute the script with the required arguments e.g.

```bash
python caiman_asr_train/latency/forced_align.py --dataset_dir /path/to/dataset --manifests data.json --model_config /path/to/model/config.yaml
```

Please note that the config file should be provided, as it contains information on
the transcription character set and normalization.
By default, CTM files are exported to the same location as the manifest files and share the same base name e.g.
if `--dataset_dir /path/to/dataset` and `--manifests data.json`, then the default filepath of the CTM file is `/path/to/dataset/data.ctm`.
The output directory to which CTM files are saved can be changed as follows:

```bash
python caiman_asr_train/latency/forced_align.py --dataset_dir /path/to/dataset --manifests manifest.json --output_dir /custom/output/directory --model_config /path/to/model/config.yaml
```

Multiple manifest files can be passed to the script e.g.

```bash
python caiman_asr_train/latency/forced_align.py --dataset_dir /path/to/dataset --manifests manifest1.json manifest2.json --model_config /path/to/model/config.yaml
```

The script also supports (multiple) tar files:

```bash
python caiman_asr_train/latency/forced_align.py --read_from_tar --tar_files data1.tar data2.tar --dataset_dir /path/to/dataset --model_config /path/to/model/config.yaml
```

Both absolute and relative paths are accepted for `--manifests` and `--tar_files`.

By default, utterances are split into 5 minute segments. This allows us to perform forced alignment on datasets with very long utterances (e.g. Earnings21) without encountering memory issues.
Most datasets have utterances shorter than 5 minutes and are therefore unaffected by this.
To change the segment length, pass the optional `--segment_len` argument with an integer number of minutes e.g.

```bash
python caiman_asr_train/latency/forced_align.py --segment_len 15 --dataset_dir /path/to/dataset --manifests data.json --model_config /path/to/model/config.yaml
```

There is also a CPU option:

```bash
python caiman_asr_train/latency/forced_align.py --cpu --dataset_dir /path/to/dataset --manifests data.json --model_config /path/to/model/config.yaml
```

## CTM

CTM (Conversation Time Mark) format is space separated file with entries:

`<recording_id> <channel_id> <token_start_time> <token_duration> <token_value>`

and an optional sixth entry `<confidence_score>`.
CTM allows either token-level or word-level timestamps.

### Next Steps

To improve the emission latency of your model, consider training with a [Delay Penalty](./delay_penalty.md).
