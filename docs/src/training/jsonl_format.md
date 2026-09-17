# `JSONL` format

The `JSONL` format is the default in this repository and if you are training on your own data it is recommended to manipulate it into this format. Note that the data preparation steps are slightly different given the model you have decided to train so please refer to the [model configuration](model_yaml_configurations.md) page first.

The `JSONL` format stores a [Lhotse
CutSet](https://lhotse.readthedocs.io/en/latest/getting-started.html), each cut
must have a recording and a single supervision.

## Prepare LibriSpeech in `JSONL` format <a name="librispeech_json"></a>

This page takes LibriSpeech as it is distributed from the
<https://www.openslr.org> website and prepares it into a JSONL manifest format.

### Quick Start <a name="quick_start"></a>

To run the data preparation steps for LibriSpeech and the `base` model run the following from the `training/` directory:

```bash
# Download data to /datasets/LibriSpeech: requires 120GB of disk
./scripts/prepare_librispeech.sh
```

To run preprocessing for the `testing` or `large` configurations, instead run:

```bash
SPM_SIZE=1023 CONFIG_NAME=testing-1023sp ./scripts/prepare_librispeech.sh
SPM_SIZE=17407 CONFIG_NAME=large-17407sp ./scripts/prepare_librispeech.sh
```

```admonish
If `~/datasets` on the host is mounted to `/datasets`, the downloaded data will be accessible outside the container at `~/datasets/LibriSpeech`.
```

### Further detail: `prepare_librispeech.sh` <a name="prepare_librispeech_sh"></a>

The script will:

1. Download data
2. Create `JSON` manifests for each subset of LibriSpeech
3. Convert the manifests into end-pointed manifests
4. Convert the `JSON` manifests into 'JSONL' manifests
5. Create a sentencepiece tokenizer from the train-960h subset
6. Record log-mel stats for the train-960h subset
7. Populate the [missing fields](model_yaml_configurations.md#missing_yaml_fields) of a YAML configuration template
8. Generate an n-gram language model with KenLM from the train-960h subset

#### 1. Data download

Having run the script, the following folders should exist inside the container:

- `/datasets/LibriSpeech`
  - `train-clean-100/`
  - `train-clean-360/`
  - `train-other-500/`
  - `dev-clean/`
  - `dev-other/`
  - `test-clean/`
  - `test-other/`

#### 2. JSON manifests

- `/datasets/LibriSpeech/`
  - `librispeech-train-clean-100-flac.json`
  - `librispeech-train-clean-360-flac.json`
  - `librispeech-train-other-500-flac.json`
  - `librispeech-train-clean-100-flac.eos.json`
  - `librispeech-train-clean-360-flac.eos.json`
  - `librispeech-train-other-500-flac.eos.json`
  - `librispeech-dev-clean-flac.json`
  - `librispeech-dev-other-flac.json`
  - `librispeech-test-clean-flac.json`
  - `librispeech-test-other-flac.json`

#### 3. Add EOS

- `/datasets/LibriSpeech/`
  - `librispeech-train-clean-100-flac.eos.json`
  - `librispeech-train-clean-360-flac.eos.json`
  - `librispeech-train-other-500-flac.eos.json`

#### 4. JSONL manifests

- `/datasets/LibriSpeech/`
  - `librispeech-train-clean-100-flac.cuts.jsonl.gz`
  - `librispeech-train-clean-360-flac.cuts.jsonl.gz`
  - `librispeech-train-other-500-flac.cuts.jsonl.gz`
  - `librispeech-train-clean-100-flac.eos.jsonl.gz`
  - `librispeech-train-clean-360-flac.eos.jsonl.gz`
  - `librispeech-train-other-500-flac.eos.jsonl.gz`
  - `librispeech-dev-clean-flac.cuts.jsonl.gz`
  - `librispeech-dev-other-flac.cuts.jsonl.gz`
  - `librispeech-test-clean-flac.cuts.jsonl.gz`
  - `librispeech-test-other-flac.cuts.jsonl.gz`

#### 5. Sentencepiece tokenizer

- `/datasets/sentencepieces/`
  - `librispeech8703.model`
  - `librispeech8703.vocab`

#### 6. Log-mel stats

- `/datasets/stats/STATS_SUBDIR`:
  - `melmeans.pt`
  - `meln.pt`
  - `melvars.pt`

The `STATS_SUBDIR` will differ depending on the model since these stats are affected by the feature extraction window size. They are:

- `testing`: `/datasets/stats/librispeech-winsz0.02`
- {`base`, `large`}: `/datasets/stats/librispeech-winsz0.025`

#### 7. `_run.yaml` config

In the `configs/` directory. Depending on the model you are training you will have one of:

- `testing`: `configs/testing-1023sp_run.yaml`
- `base`: `configs/base-8703sp_run.yaml`
- `large`: `configs/large-17407sp_run.yaml`

`_run` indicates that this is a complete config, not just a template.

#### 8. N-gram language model

- `/datasets/ngrams/librispeech8703/`
  - `transcripts.txt`
  - `ngram.arpa`
  - `ngram.binary`

## Selecting transcripts

Each supervision must include a "text" field, which contains the transcript
of the corresponding audio file.

If multiple transcripts exist for the same audio file, the model can be trained
using a specific transcript by specifying it in the use_transcripts field of
the YAML configuration:

```yaml
use_transcripts:
  to_use: ["<transcript_key_1>", "<transcript_key_2>", ...]
  on_missing: "raise_error" # or "use_default"
```

These extra transcripts are stored as custom data inside each supervision.

The first transcript key that is found in the JSONL manifest will be used.

If no key is found in the JSONL manifest, the behaviour will depend on the
value of the on_missing field:

- "raise_error": an error will be raised.
- "use_default": the default transcript will be used.

## Standardizing transcripts

Training on multiple datasets can negatively impact performance, as identical words may
be transcribed differently across datasets - for example, "colour" (British English)
vs "color" (American English). This inconsistency splits the model's probability distribution
across multiple spellings, reducing prediction confidence.

To address this, we standardize the training transcripts using Whisper EnglishTextNormalizer.
This process follows the same approach as WER standardization
(see [WER calculation docs](./wer_calculation.md) for more details).

There are two methods to do this which are described below.

### Recommended approach: Pre-standardizing manifests

<!-- TODO: This script doesn't work, in experiments on-the-fly is now as fast to maybe we can retire this? -->

The recommended approach is to standardize your manifests before training using the `standardize_manifest.py` script:

```bash
python caiman_asr_train/data/text/standardize_manifest.py \
  --manifests /path/to/shar/manifest \
  --output_dirs /path/to/save/output \
  --preserve-case \
  --preserve-punctuation
```

This creates standardized versions of each transcript field as `transcript-standardized`. You can also choose to standardize
transcripts other than the default text using `--transcript_fields`:

```bash
python caiman_asr_train/data/text/standardize_manifest.py \
  --manifests /path/to/shar/manifest \
  --transcript-fields field1-name field2-name
```

The standardized versions of these will be stored as `field1-name-standardized`, `field2-name-standardized`. If any of the provided
fields are not found, they will be ignored.

You can then use these standardized fields during training by configuring your model's YAML:

```yaml
use_transcripts:
  to_use: ["transcript-standardized"]
  on_missing: "raise_error" # or "use_default"
```

This approach offers several advantages:

- Faster training startup (standardization is done only once)
- Ability to standardize multiple transcript fields in the same manifest

Multiple manifests and fields can be standardized at once:

```bash
python caiman_asr_train/data/text/standardize_manifest.py \
  --manifests /path/to/shar/manifest1 /path/to/shar/manifest2 \
  --output_dirs /path/to/output/for/manifest1 /path/to/output/for/manifest2
```

This will standardize the `text` field in each of the jsonl in `manifest1` and `manifest2` with a `transcript-standardized` custom field

All arguments:

- `--manifests [str] ...`: Absolute path to dataset manifest files [required].
- `--output_dirs [str] ...`: Absolute path to save standardized manifest files [required].
- `--lang_code [str]`: ISO code of the language of the transcripts to be standardized [required].
- `--preserve-case`: Flag to preserve the original casing of the text [optional].
- `--preserve-punctuation`: Flag to preserve the original punctuation of the text [optional].
- `--expand-contractions`: Flag to enable expansion of contractions (e.g. "don't" -> "do not")[optional].
- `--model-config [str]`: Path to model config to extract user symbols to preserve e.g. `<EOS>`
- `--num-workers [int]`: Number of worker processes to use [optional][default: CPU count].
- `--overwrite`: Flag to enable overwrite of existing output files [optional].

```admonish
If using offline standardization of manifests, please ensure to disable on-the-fly standardization by setting
`standardize_text` to false in the model config (.yaml) file.
```

```admonish
By default, standardization expands all contractions (e.g "don't" -> "do not").
As a result, if you train a model on standardized transcripts,
it will always predict the expanded form e.g. it will output "do not"
whether the speaker says "don't" or "do not".

If this behaviour is not desired,
you can disable contraction expansion during train-time standardization
by passing the `--no-expand-contractions` flag to the `standardize_manifest.py` script.
This will however worsen WERs by ~1-2% relative.
```

### Alternative: On-the-fly standardization

Alternatively, the YAML config includes the entry `standardize_text: true|false`,
enabling on-the-fly standardization of training transcripts during training.

When using the SHAR data format on-the-fly standardization should be fully
overlapped with GPU compute, resulting in minimal training speed impact.

### Preserving casing and punctuation

By default, during training the original casing and punctuation of the transcripts
will be preserved. This is is controlled by the `preserve_case` and `preserve_punctuation`
parameters in the YAML config.

The default settings are:

```yaml
standardize_text: true
preserve_case: true
preserve_punctuation: true
```

```admonish
This is irrelevant to the WER calculation, which always lowercases and standardizes
the transcripts before calculating WER.
```

## Next steps

Having run the data preparation steps, go to the [training docs](./training.md) to start training.

### See also

- [Supported dataset formats](supported_dataset_formats.md)
- [Input activation normalization](log_mel_feature_normalization.md)
