# Saving Predictions

To dump the predicted text for a list of input wav files, pass the `--dump_preds` argument and call `val.sh`:

```
./scripts/val.sh --dump_preds --val_manifests=/results/your-inference-list.json
```

Predicted text will be written to `/results/preds[rank].txt`

The argument `--dump_preds` can be used whether or not there are ground-truth transcripts in the json file.  If there are,
then the word error rate reported by val will be accurate; if not, then it will be nonsense and should
be ignored.  The minimal json file for inference (with 2 wav files) looks like this:

```
[
  {
    "transcript": "dummy",
    "files": [
      {
        "fname": "relative-path/to/stem1.wav"
      }
    ],
    "original_duration": 0.0
  },
  {
    "transcript": "dummy",
    "files": [
      {
        "fname": "relative-path/to/stem2.wav"
      }
    ],
    "original_duration": 0.0
  }
]
```

where "dummy" can be replaced by the ground-truth transcript for accurate word error rate calculation,
where the filenames are relative to the `--data_dir` argument fed to (or defaulted to by) `val.sh`, and where
the original_duration values are effectively ignored (compared to infinity) but must be present.
Predictions can be generated using other checkpoints by specifying the `--checkpoint` argument.
