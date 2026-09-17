# SHAR format

The SHAR format is a sharded tarred archive format developed as part of
[Lhotse](https://lhotse.readthedocs.io/en/latest/getting-started.html).
CAIMAN-ASR's training is optimized to work with SHAR archives, which
provide efficient storage and retrieval of large datasets. SHAR allows
efficient randomization of data order as well as de-duplication across data
workers. To convert your data into the SHAR format use:

```bash
python ./training/caiman_asr_train/lhotse/scripts/convert.py cuts2shar \
  /path/to/data.cuts.jsonl.gz \
  /path/to/out/dir/ \
  --compression flac
```

These SHAR datasets can then be used in your train_dataset.yaml as follows:

```yaml
format: multilingual-shar
languages:
  lang_code:
    canary:
      alpha: 0.5
    datasets:
      - /path/to/out/dir/:
      - /some/other/shar/dir/:
```

See the convert script's `--help` for further options
