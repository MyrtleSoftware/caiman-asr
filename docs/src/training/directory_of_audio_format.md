# Directory of audio format

```admonish
This format is supported for validation but not training
```

It is possible to run validation on all audio files (and their respective `.txt` transcripts)
found recursively in two directories `--val_audio_dir` and `--val_txt_dir`.

### Directory Structure

The audio and transcripts directories should contain the same number of files, and the file names should match.
For example, the structure of the directories could be:

```
audio_dir/
  dir1/
    file1.wav
    file2.wav
txt_dir/
  dir1/
    file1.txt
    file2.txt
```

The audio and transcript files can be under the same directory.

### Running Validation

Using data from directories for validation can be done by parsing the argument
`--val_from_dir` along with the audio and transcript directories as follows:

```bash
scripts/val.sh --val_from_dir --val_audio_dir audio_dir --val_txt_dir txt_dir --dataset_dir /path/to/dataset/dir
```

where the `audio_dir` and `txt_dir` are relative to the `--dataset_dir`.

When training on webdataset files (`--read_from_tar=True` in the `train.py`), validation on directories is not supported.
