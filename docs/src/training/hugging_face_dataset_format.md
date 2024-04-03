# Loading a dataset from the Hugging Face Hub

```admonish
This format is supported in `val.sh` but not in `train.sh`.
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
