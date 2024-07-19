# N-gram Language Model

An external language model can improve ASR accuracy, especially in out-of-domain contexts with rare or specialised vocabulary.
CAIMAN-ASR supports the use of [KenLM](https://github.com/kpu/kenlm) n-gram language model shallow fusion corrections when using `--decoder=beam`.
We have seen a consistent WER improvement even when the N-gram is trained on ASR data transcripts.
As such we automatically generate this N-gram during preprocessing and use it during validation by default.
See the [Validation with an N-gram](#validation) section for more details.

## Build an N-gram Language Model

When adapting the preprocessing steps detailed [here](./json_format.md) for your own dataset, you should have generated an n-gram language model trained on your transcripts.
To generate an n-gram from a different dataset, see the following steps.

### Preparing Data

To train an n-gram with KenLM on transcripts from ASR datasets, the data must first be prepared into the correct format - a `.txt` file where tokens within a sentence are space-separated and each sentence appears on a new line.

To gather the transcripts from json manifest files, run the following command inside a running container:

```bash
python caiman_asr_train/lm/prep_kenlm_data.py --data_dir /path/to/dataset/ --manifests manifest1.json manifest2.json --output_path /path/to/transcripts.txt --model_config configs/config.yaml
```

To instead gather the transcripts from data in the WebDataset format, run the following command:

```bash
python caiman_asr_train/lm/prep_kenlm_data.py --data_dir /path/to/dataset/ --read_from_tar --tar_files file1.tar file2.tar --output_path /path/to/transcripts.txt --model_config configs/config.yaml
```

```admonish
Use the same model configuration file that was used for RNN-T. If the n-gram is not trained on data tokenized by the same SentencePiece model, using an ngram language model is likely to degrade WER.
```

### Training an N-gram

To train an n-gram, run the `generate_ngram.sh` script as follows:

```bash
./scripts/generate_ngram.sh [NGRAM_ORDER] /path/to/transcripts.txt /path/to/ngram.arpa /path/to/ngram.binary
```

For example, to generate a 4-gram, set `[NGRAM_ORDER]` to 4 as follows:

```bash
./scripts/generate_ngram.sh 4 /path/to/transcripts.txt /path/to/ngram.arpa /path/to/ngram.binary
```

The script will produce an ARPA file, which is a human-readable version of the language model, and a binary file, which allows for faster loading and is the recommended format. Binary files are the only usable format when generating [hardware checkpoints](./export_inference_checkpoint.md), though providing an n-gram is optional.

## Validation with an N-gram <a name="validation"></a>

During beam search validation, the n-gram language model generated during preprocessing is used by default, by reading from the following entries in the model configuration file:

```yaml
ngram:
  ngram_path: /datasets/ngrams/NGRAM_SUBDIR
  scale_factor: 0.05
```

First, a binary file, named `ngram.binary`, in `NGRAM_SUBDIR` is searched for. If not found, an ARPA file - `ngram.arpa` - is searched for. If neither file exists, the process will crash with an error. To prevent this, use the `--skip_ngram` flag to disable the use of an n-gram during validation with beam search:

```bash
scripts/val.sh --decoder=beam --skip_ngram
```

The `scale_factor` adjusts the scores from the n-gram language model, and this will require tuning for your dataset. Values between 0.05 and 0.1 are empirically effective for improving WER. See the [Sweep Scale Factor](#sweep-scale-factor) section below for details on running a sweep across the scale factor.

To use an n-gram that was trained on a different dataset, use the `--override_ngram_path` argument, which will take precedence over any n-grams in `NGRAM_SUBDIR`:

```bash
scripts/val.sh --decoder=beam --override_ngram_path /path/to/ngram.binary
```

## Sweep Scale Factor

To optimize the `scale_factor` for your n-gram language model, use the `sweep_scale_factor.py` script.
This script iterates over multiple `scale_factor` values, performs validation, and updates your model config YAML with the best one based on WER.

Run the following command to perform a sweep:

```bash
python caiman_asr_train/lm/sweep_scale_factor.py --checkpoint /path/to/checkpoint.pt --model_config configs/config.yaml --val_manifests /path/to/manifest.json
```

By default, a sweep is performed across `[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]`.
To specify custom values, use the `--scale_factors` argument:

```bash
python caiman_asr_train/lm/sweep_scale_factor.py --scale_factors 0.1 0.2 0.3 --checkpoint /path/to/checkpoint.pt --model_config configs/config.yaml --val_manifests /path/to/manifest.json
```
