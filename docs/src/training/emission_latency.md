# Emission Latency

Emission latency (EL) is defined by the time difference between the end of a spoken word in an audio file and when the model outputs the final token for the corresponding word, minus the mean frame latency.
After the model receives the final audio frame of a word, it might not predict the word until it has heard a few more frames of audio.
EL measures this delay.

To calculate the model's EL during validation, pass the `--calculate_emission_latency` flag, e.g.

```bash
./scripts/val.sh --calculate_emission_latency
```

When this flag is enabled, CTM files containing model timestamps are exported to `--output_dir`.

Emission latencies are calculated by aligning the model-exported CTM files with corresponding ground truth alignments.
Ground truth alignments are expected to be in the manifest files for the dataset.

See [Forced Alignment](#forced-alignment) for details on producing ground truth alignments.

The script outputs the mean latency, as well as the 50th, 90th, and 99th percentile latencies.

Moreover, the Token Usage Rate is reported. This is the proportion of words' timestamps that are used in the emission latency calculation.

If one already has model-exported CTM files and corresponding ground truth files, the [measure_latency.py](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/latency/measure_latency.py)
script can be used instead of running a complete validation run. To do so, run the script with paths to the ground truth and model CTM files:

```bash
python caiman_asr_train/latency/measure_latency.py --gt_ctm /path/to/dataset_manifest.jsonl.gz --model_ctm /path/to/model.ctm
```

To include substitution errors in latency calculations, add the `--include_subs` flag:

```bash
python caiman_asr_train/latency/measure_latency.py --gt_ctm /path/to/dataset_manifest.jsonl.gz --model_ctm /path/to/model.ctm --include_subs
```

To export a scatter plot of EL against time from the start of the sequence, pass a filepath to the optional `--output_img_path` argument e.g.

```bash
python caiman_asr_train/latency/measure_latency.py --gt_ctm /path/to/dataset_manifest.jsonl.gz --model_ctm /path/to/model.ctm --output_img_path /path/to/img.png
```

EL logging is also compatible with `val_multiple.sh`.

## Forced Alignment

The script [forced_align.py](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/latency/forced_align.py) is used to align audio recordings with their corresponding transcripts, producing ground truth timestamps for each word.

To perform forced alignment, execute the script with the required arguments e.g.

```bash
python caiman_asr_train/latency/forced_align.py --dataset_dir /path/to/dataset --manifests data.jsonl.gz --output_dir /pat/to/output/dir --lang_code lang_code
```

By default, alignments would be appended to each supervision segment in the cutset and the resulting JSONL will be saved to the output_dir.
To overwrite existing JSONL use `--overwrite`.

Multiple manifest files can be passed to the script e.g.

```bash
python caiman_asr_train/latency/forced_align.py --dataset_dir /path/to/dataset --manifests manifest1.jsonl manifest2.jsonl --output_dir /pat/to/output/dir --lang_code lang_code
```

By default, utterances are split into 5 minute segments. This allows us to perform forced alignment on datasets with very long utterances (e.g. Earnings21) without encountering memory issues.
Most datasets have utterances shorter than 5 minutes and are therefore unaffected by this.
To change the segment length, pass the optional `--segment_len` argument with an integer number of minutes e.g.

```bash
python caiman_asr_train/latency/forced_align.py --segment_len 15 --dataset_dir /path/to/dataset --manifests data.jsonl.gz --output_dir /pat/to/output/dir --lang_code lang_code
```

There is also a CPU option:

```bash
python caiman_asr_train/latency/forced_align.py --cpu --dataset_dir /path/to/dataset --manifests data.json
```

## Alignment

The alignment uses the Lhotse Alignmnent format where each token has an associated AlignmentItem.
Each AlignmentItem consists of the token, start time, duration and confidence score, for example:

```
    "word": [
        [
        "Thank",
        0,
        0.32,
        0.49
        ],
        [
        "you",
        0.32,
        0.08,
        0.97
        ]
    ]
```

### Next Steps

To improve the emission latency of your model, consider training with a [Delay Penalty](./delay_penalty.md).
