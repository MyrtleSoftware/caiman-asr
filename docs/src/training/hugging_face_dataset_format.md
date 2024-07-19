# Hugging Face Dataset Format

## Validating directly on a dataset from the Hugging Face Hub

```admonish
Validating on a Hugging Face dataset is supported in `val.sh` and `train.sh`.
To train on a Hugging Face dataset, you will need to convert it to JSON format,
as described in the next section.
```

This command will run validation on distil-whisper's [version](https://huggingface.co/datasets/distil-whisper/librispeech_asr) of LibriSpeech dev-other:

```bash
./scripts/val.sh --num_gpus 8 \
  --checkpoint /path/to/checkpoint.pt  \
  --use_hugging_face \
  --hugging_face_val_dataset distil-whisper/librispeech_asr \
  --hugging_face_val_split validation.other
```

This will download the dataset and cache it in `~/.cache/huggingface`, which will persist between containers.

Since datasets are large, you may wish to change the Hugging Face cache location via `HF_CACHE=[path] ./scripts/docker/launch.sh ...`.

For some datasets, you may need to set more options. The following command will validate on the first 10 utterance of [google/fleurs](https://huggingface.co/datasets/google/fleurs):

```bash
./scripts/val.sh --num_gpus 8 \
  --checkpoint /path/to/checkpoint.pt \
  --use_hugging_face \
  --hugging_face_val_dataset google/fleurs \
  --hugging_face_val_config en_us \
  --hugging_face_val_transcript_key raw_transcription \
  --hugging_face_val_split validation[0:10]
```

See the [docstrings](https://github.com/MyrtleSoftware/caiman-asr/blob/main/training/caiman_asr_train/args/hugging_face.py) for more information.

## Converting a Hugging Face dataset to JSON format

The following command will download the `train.clean.100` split of
[distil-whisper/librispeech_asr](https://huggingface.co/datasets/distil-whisper/librispeech_asr)
and convert it to JSON format,
putting the result in `/datasets/LibriSpeechHuggingFace`:

```bash
python caiman_asr_train/data/make_datasets/hugging_face_to_json.py \
  --hugging_face_dataset distil-whisper/librispeech_asr \
  --data_dir /datasets/LibriSpeechHuggingFace \
  --hugging_face_split train.clean.100
```
